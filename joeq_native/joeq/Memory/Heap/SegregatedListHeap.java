package Memory.Heap;

import Allocator.HeapAllocator;
import Allocator.ObjectLayout;
import Allocator.ObjectLayoutMethods;
import Clazz.jq_Array;
import Main.jq;
import Memory.Address;
import Memory.Debug;
import Memory.HeapAddress;
import Memory.Manager.CollectorThread;
import Memory.Manager.GCConstants;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;

/**
 * @author John Whaley
 */
public class SegregatedListHeap extends Heap implements GCConstants {

    public static final boolean LITTLE_ENDIAN = true;

    private final static int OUT_OF_BLOCKS = -1;

    // value below is a tuning parameter: for single threaded appl, on multiple processors
    private static final int numBlocksToKeep = 10; // GSC 

    private Object sysLockFree = new Object();
    private Object sysLockBlock = new Object();

    // 1 BlockControl per GC-SIZES for initial use, before heap setup
    private BlockControl[] init_blocks;
    // BlockControl[] 1 per BLKSIZE block of the heap

    private BlockControl[] blocks; // 1 per BLKSIZE block of the heap

    private int[] partialBlockList; // GSC

    private int num_blocks; // number of blocks in the heap
    private int first_freeblock; // number of first available block
    private int highest_block; // number of highest available block
    private int blocks_available; // number of free blocks for small obj's

    /**
     * backing store for heap metadata
     */
    private MallocHeap mallocHeap;

    public SegregatedListHeap(String s, MallocHeap mh) {
        super(s);
        mallocHeap = mh;
    }

    /**
     * Setup done during bootimage writing
     */
    public void init(jq_NativeThread st) {
        partialBlockList = new int[GC_SIZES]; // GSC
        for (int i = 0; i < GC_SIZES; i++) {
            partialBlockList[i] = OUT_OF_BLOCKS;
        }

        st.sizes = new SizeControl[GC_SIZES];
        init_blocks = new BlockControl[GC_SIZES];

        // On the jdk side, we allocate an array of SizeControl Blocks, 
        // one for each size.
        // We also allocate init_blocks array within the boot image.  
        // At runtime we allocate the rest of the BLOCK_CONTROLS, whose number 
        // depends on the heapsize, and copy the contents of init_blocks 
        // into the first GC_SIZES of them.

        for (int i = 0; i < GC_SIZES; i++) {
            st.sizes[i] = new SizeControl();
            init_blocks[i] = new BlockControl();
            st.sizes[i].first_block = i; // 1 block/size initially
            st.sizes[i].current_block = i;
            st.sizes[i].ndx = i;
            init_blocks[i].mark = new byte[GC_BLOCKSIZE / GC_SIZEVALUES[i]];
            for (int ii = 0; ii < GC_BLOCKSIZE / GC_SIZEVALUES[i]; ii++) {
                init_blocks[i].mark[ii] = 0;
            }
            init_blocks[i].nextblock = OUT_OF_BLOCKS;
            init_blocks[i].slotsize = GC_SIZEVALUES[i];
        }

        // set up GC_INDEX_ARRAY for this Processor
        // NOTE: we exploit the fact that all allocations are word aligned to
        //       reduce the size of the index array by 4x....
        st.GC_INDEX_ARRAY = new SizeControl[(GC_MAX_SMALL_SIZE >> 2) + 1];
        st.GC_INDEX_ARRAY[0] = st.sizes[0]; // for size = 0
        for (int i = 0, j = 4; i < GC_SIZES; i++) {
            for (; j <= GC_SIZEVALUES[i]; j += 4) {
                st.GC_INDEX_ARRAY[j >> 2] = st.sizes[i];
            }
        }

        st.backingSLHeap = this;
    }

    public void boot(jq_NativeThread st, ImmortalHeap immortalHeap) {
        
        int k = 1024 + GC_SIZES;
        start = mallocHeap.allocateZeroedMemory(k * GC_BLOCKSIZE);
        if (start.isNull()) {
            Debug.writeln("Panic!  Cannot allocate ", k * GC_BLOCKSIZE, "bytes.");
            jq.UNREACHABLE();
        }
        end = (HeapAddress) start.offset(k * GC_BLOCKSIZE);
        
        blocks = init_blocks;

        // Now set the beginning address of each block into each BlockControl.
        // Note that init_blocks is in the boot image, but heap pages are controlled by it.
        for (int i = 0; i < GC_SIZES; i++) {
            init_blocks[i].baseAddr = (HeapAddress) start.offset(i * GC_BLOCKSIZE);
            build_list_for_new_block(init_blocks[i], st.sizes[i]);
        }

        // Now allocate the blocks array - which will be used to allocate blocks to sizes
        int size = getSize();
        num_blocks = size / GC_BLOCKSIZE;
        blocks = (BlockControl[])
            immortalHeap.allocateArray(BlockControl._array, num_blocks);

        // index for highest page in heap
        highest_block = num_blocks - 1;
        blocks_available = highest_block - GC_SIZES; // available to allocate

        // Now fill in blocks with values from blocks_init
        for (int i = 0; i < GC_SIZES; i++) {
            // NOTE: if blocks are identified by index, st.sizes[] need not be changed; if
            // blocks are identified by address, then updates st.sizes[0-GC_SIZES] here
            blocks[i] = init_blocks[i];
        }

        // At this point we have assigned the first GC_SIZES blocks, 
        // 1 per, to each GC_SIZES bin
        // and are prepared to allocate from such, or from large object space:
        // BlockControl blocks are not used to manage large objects - 
        // they are unavailable by special logic for allocation of small objs
        //
        first_freeblock = GC_SIZES; // next to be allocated
        init_blocks = null; // these are currently live through blocks

        // Now allocate the rest of the BlockControls
        for (int i = GC_SIZES; i < num_blocks; i++) {
            BlockControl bc =
                (BlockControl) immortalHeap.allocateObject(BlockControl._class);
            blocks[i] = bc;
            bc.baseAddr = (HeapAddress) start.offset(i * GC_BLOCKSIZE);
            bc.nextblock = (i == num_blocks - 1) ? OUT_OF_BLOCKS : i + 1;
        }

    }

    /**
     * Allocate size bytes of raw memory.
     * Size is a multiple of wordsize, and the returned memory must be word aligned
     * 
     * @param size Number of bytes to allocate
     * @return Address of allocated storage
     */
    protected HeapAddress allocateZeroedMemory(int size) {
        return allocateFastPath(size);
    }

    /**
     * Hook to allow heap to perform post-allocation processing of the object.
     * For example, setting the GC state bits in the object header.
     */
    protected void postAllocationProcessing(Object newObj) {
        // nothing to do in this heap
    }

    /**
     * Fast path (inlined) allocation path for SegregatedListHeap.
     * Processor local allocation via fields on the jq_NativeThread object.
     * 
     * @param size Number of bytes to allocate
     * @return Address of allocated storage
     */
    public static HeapAddress allocateFastPath(int size)
        throws OutOfMemoryError {

        // Use direct addressing to avoid spurious array bounds check on common case allocation path.
        // NOTE: we exploit the fact that all allocations are word aligned to
        //       reduce the size of the index array by 4x....
        size = Address.align(size, HeapAddress.logSize());
        jq_NativeThread nt = Unsafe.getThreadBlock().getNativeThread();
        HeapAddress loc = HeapAddress.addressOf(nt.GC_INDEX_ARRAY);
        loc = (HeapAddress) loc.offset(size);
        HeapAddress rs = (HeapAddress) loc.peek();
        SizeControl the_size = (SizeControl) rs.asObject();
        HeapAddress next_slot = the_size.next_slot;
        if (next_slot.isNull()) {
            // slow path: find a new block to allocate from
            return nt.backingSLHeap.allocateSlot(the_size, size);
        } else {
            // inlined fast path: get slot from block
            return allocateSlotFast(the_size, next_slot);
        }
    }

    /** 
     * given an address in the heap 
     *  set the corresponding mark byte on
     */
    public boolean mark(HeapAddress ref) {
        HeapAddress tref = ref;
        int blkndx = (tref.difference(start)) >> LOG_GC_BLOCKSIZE;
        BlockControl this_block = blocks[blkndx];
        int offset = tref.difference(this_block.baseAddr);
        int slotndx = offset / this_block.slotsize;

        if (this_block.mark[slotndx] != 0)
            return false;

        if (false) {
            // store byte into byte array
            this_block.mark[slotndx] = 1;
        } else {
            byte tbyte = (byte) 1;
            int temp, temp1;
            for (;;) {
                // get word with proper byte from map
                HeapAddress addr = (HeapAddress) HeapAddress.addressOf(this_block.mark).offset((slotndx >> 2) << 2);
                temp1 = addr.peek4();
                if (this_block.mark[slotndx] != 0)
                    return false;
                int index;
                if (LITTLE_ENDIAN)
                    index = slotndx % 4; // get byte in word - little Endian
                else
                    index = 3 - (slotndx % 4); // get byte in word - big Endian
                int mask = tbyte << (index * 8); // convert to bit in word
                temp = temp1 | mask; // merge into existing word
                addr.atomicCas4(temp1, temp);
                if (Unsafe.isEQ()) break;
            }
        }

        this_block.live = true;
        return true;
    }

    public boolean isLive(HeapAddress ref) {
        HeapAddress tref = ref;
        //  check for live small object
        int blkndx, slotno, size, ij;
        blkndx = (tref.difference(start)) >> LOG_GC_BLOCKSIZE;
        BlockControl this_block = blocks[blkndx];
        int offset = tref.difference(this_block.baseAddr);
        int slotndx = offset / this_block.slotsize;
        return (this_block.mark[slotndx] != 0);
    }

    /**
     * Allocate a fixed-size small chunk when we know one is available.  Always inlined.
     * 
     * @param the_size Header record for the given slot size
     * @param next_slot contents of the_size.next_slot
     * @return Address of free, zero-filled storage
     */
    protected static HeapAddress allocateSlotFast(
        SizeControl the_size,
        HeapAddress next_slot)
        throws OutOfMemoryError {

        // Get the next object from the head of the list
        HeapAddress objaddr = next_slot;

        // Update the next pointer
        the_size.next_slot = (HeapAddress) objaddr.peek();

        // Zero out the old next pointer 
        // NOTE: Possible MP bug on machines with relaxed memory models....
        //       technically we need a store barrier after we zero this word!
        objaddr.poke(HeapAddress.getNull());

        // Return zero-filled storage
        return objaddr;
    }

    /**
     * Allocate a fixed-size small chunk when we have run off the end of the 
     * current block.  This will either use a partially filled block of the 
     * given size, or a completely empty block, or it may trigger GC.
     *   @param the_size Header record for the given slot size
     *   @param size Size in bytes to allocate
     *   @return Address of free, zero-filled storage
     */
    protected HeapAddress allocateSlot(SizeControl the_size, int size)
        throws OutOfMemoryError {

        int count = 0;
        while (true) {
            HeapAddress objaddr = allocateSlotFromBlocks(the_size, size);
            if (!objaddr.isNull())
                return objaddr;

            // Didn't get any memory; ask allocator to deal with this
            // This may not return (VM exits with OutOfMemoryError)
            HeapAllocator.heapExhausted(this, size, count++);

            // An arbitrary amount of time may pass until this thread runs again,
            // and in fact it could be running on a different virtual processor.
            // Therefore, reacquire the processor local size control.
            // reset the_size in case we are on a different processor after GC
            // NOTE: we exploit the fact that all allocations are word aligned to
            //       reduce the size of the index array by 4x....
            the_size =
                Unsafe.getThreadBlock().getNativeThread().GC_INDEX_ARRAY[size >> 2];

            // At this point allocation might or might not succeed, since the
            // thread which originally requested the collection is usually not
            // the one to run after gc is finished; therefore failing here is
            // not a permanent failure necessarily
            HeapAddress next_slot = the_size.next_slot;
            if (!next_slot.isNull()) // try fast path again
                return allocateSlotFast(the_size, next_slot);
        }
    }

    /**
     * Find a new block to use for the given slot size, format the 
     * free list, and allocate an object from that list.  First tries to
     * allocate from the processor-local list for the given size, then from 
     * the global list of partially filled blocks of the given size, and
     * finally tries to get an empty block and format it to the given size.
     *   @param the_size Header record for the given slot size
     *   @param size Size in bytes to allocate
     *   @return Address of free storage or 0 if none is available
     */
    protected HeapAddress allocateSlotFromBlocks(
        SizeControl the_size,
        int size) {
        BlockControl the_block = blocks[the_size.current_block];

        // First, look for a slot in the blocks on the existing list
        while (the_block.nextblock != OUT_OF_BLOCKS) {
            the_size.current_block = the_block.nextblock;
            the_block = blocks[the_block.nextblock];
            if (build_list(the_block, the_size))
                return allocateSlotFast(the_size, the_size.next_slot);
        }

        // Next, try to get a partially filled block of the given size from the global pool
        while (getPartialBlock(the_size.ndx) == 0) {
            the_size.current_block = the_block.nextblock;
            the_block = blocks[the_block.nextblock];

            if (build_list(the_block, the_size))
                return allocateSlotFast(the_size, the_size.next_slot);
        }

        // Finally, try to allocate a free block and format it to the given size
        if (getnewblock(the_size.ndx) == 0) {
            the_size.current_block = the_block.nextblock;
            int idx = the_size.current_block;
            build_list_for_new_block(blocks[idx], the_size);
            return allocateSlotFast(the_size, the_size.next_slot);
        }

        // All attempts failed; heap is currently exhausted.
        return HeapAddress.getNull();
    }

    // build, in the block, the list of free slot pointers, and update the
    // associated SizeControl.next_slot; return true if a free slot was found,
    // or false if not
    //
    protected boolean build_list(
        BlockControl the_block,
        SizeControl the_size) {
        byte[] the_mark = the_block.mark;
        int first_free = 0, i = 0, j;
        HeapAddress current, next;

        for (; i < the_mark.length; i++) {
            if (the_mark[i] == 0)
                break;
        }

        if (i == the_mark.length) {
            // no free slot was found

            // Reset control info for this block, for next collection 
            HeapAddress a = HeapAddress.addressOf(the_mark);
            SystemInterface.mem_set(a, the_mark.length, (byte)0);
            the_block.live = false;
            return false; // no free slots in this block
        } else {
            // here is the first
            current =
                (HeapAddress) the_block.baseAddr.offset(i * the_block.slotsize);
        }

        SystemInterface.mem_set(current.offset(4), the_block.slotsize-4, (byte)0);
        the_size.next_slot = current;

        // now find next free slot
        i++;
        for (; i < the_mark.length; i++) {
            if (the_mark[i] == 0)
                break;
        }

        if (i == the_mark.length) {
            // this block has only 1 free slot, so..
            current.poke(HeapAddress.getNull()); // null pointer to next

            // Reset control info for this block, for next collection 
            HeapAddress a = HeapAddress.addressOf(the_mark);
            SystemInterface.mem_set(a, the_mark.length, (byte)0);
            the_block.live = false;
            return true;
        }

        next = (HeapAddress) the_block.baseAddr.offset(i * the_block.slotsize);
        current.poke(next);
        current = next;
        SystemInterface.mem_set(current.offset(4), the_block.slotsize-4, (byte)0);

        // build the rest of the list; there is at least 1 more free slot
        for (i = i + 1; i < the_mark.length; i++) {
            if (the_mark[i] == 0) { // This slot is free
                next =
                    (HeapAddress) the_block.baseAddr.offset(
                        i * the_block.slotsize);
                current.poke(next);
                current = next;
                SystemInterface.mem_set(current.offset(4), the_block.slotsize-4, (byte)0);
            }
        }

        current.poke(HeapAddress.getNull()); // set the end of the list
        // Reset control info for this block, for next collection 
        HeapAddress a = HeapAddress.addressOf(the_mark);
        SystemInterface.mem_set(a, the_mark.length, (byte)0);
        the_block.live = false;
        return true;
    }

    // A debugging routine: called to validate the result of build_list 
    // and build_list_for_new_block
    // 
    protected void do_check(BlockControl the_block, SizeControl the_size) {
        int count = 0;
        if (blocks[the_size.current_block] != the_block) {
            //VM_Scheduler.trace("do_check", "BlockControls don't match");
            jq.UNREACHABLE("BlockControl Inconsistency");
        }
        /*
        if (the_size.next_slot.isNull())
            VM_Scheduler.trace("do_check", "no free slots in block");
            */
        HeapAddress temp = the_size.next_slot;
        while (!temp.isNull()) {
            if ((temp.difference(the_block.baseAddr) < 0)
                || (temp.difference(the_block.baseAddr.offset(GC_BLOCKSIZE)) > 0)) {
                //VM_Scheduler.trace("do_check: TILT:", "invalid slot ptr", temp);
                jq.UNREACHABLE("Bad freelist");
            }
            count++;
            temp = (HeapAddress) temp.peek();
        }

        if (count > the_block.mark.length) {
            //VM_Scheduler.trace("do_check: TILT:", "too many slots in block");
            jq.UNREACHABLE("too many slots");
        }
    }

    // Input: a BlockControl that was just assigned to a size; the SizeControl
    // associated with the block
    //
    protected void build_list_for_new_block(
        BlockControl the_block,
        SizeControl the_size) {
        byte[] the_mark = the_block.mark;
        int i, delta;
        HeapAddress current = the_block.baseAddr;
        SystemInterface.mem_set(current, GC_BLOCKSIZE, (byte)0);
        delta = the_block.slotsize;
        the_size.next_slot = current; // next one to allocate
        for (i = 0; i < the_mark.length - 1; i++) {
            current.poke(current.offset(delta));
            current = (HeapAddress) current.offset(delta);
        }
        // last slot does not point forward - already zeroed
        //  

        // Reset control info for this block, for next collection 
        HeapAddress a = HeapAddress.addressOf(the_mark);
        SystemInterface.mem_set(a, the_mark.length, (byte)0);
        the_block.live = false;
    }

    // A routine to obtain a free BlockControl and return it
    // to the caller.  First use is for the VM_Processor constructor: 
    protected int getnewblockx(int ndx) {
        BlockControl alloc_block;
        int theblock;
        synchronized (sysLockBlock) {
            if (first_freeblock == OUT_OF_BLOCKS) {
                HeapAllocator.heapExhausted(this, 0, 0);
            }
            alloc_block = blocks[first_freeblock];
            theblock = first_freeblock;
            first_freeblock = alloc_block.nextblock;
        }

        alloc_block.nextblock = OUT_OF_BLOCKS;
        // this is last block in list for thissize
        alloc_block.slotsize = GC_SIZEVALUES[ndx];
        int size = GC_BLOCKSIZE / GC_SIZEVALUES[ndx];

        if (alloc_block.mark != null) {
            if (size <= alloc_block.alloc_size) {
                ObjectLayoutMethods.setArrayLength(alloc_block.mark, size);
                return theblock;
            } else { // free the existing array space
                mallocHeap.atomicFreeArray(alloc_block.mark);
            }
        }
        // allocate a mark array from the malloc heap.
        alloc_block.mark = (byte[]) mallocHeap.atomicAllocateArray(jq_Array.BYTE_ARRAY, size);
        return theblock;
    }

    protected int getPartialBlock(int ndx) {
        jq_NativeThread st = Unsafe.getThreadBlock().getNativeThread();
        SizeControl this_size = st.sizes[ndx];
        BlockControl currentBlock = blocks[this_size.current_block];

        synchronized (sysLockBlock) {

            if (partialBlockList[ndx] == OUT_OF_BLOCKS) {
                return -1;
            }

            // get first partial block of same slot size
            //
            currentBlock.nextblock = partialBlockList[ndx];
            BlockControl allocBlock = blocks[partialBlockList[ndx]];

            partialBlockList[ndx] = allocBlock.nextblock;
            allocBlock.nextblock = OUT_OF_BLOCKS;
        }

        return 0;
    }

    protected int getnewblock(int ndx) {
        int i, save, size;
        jq_NativeThread st = Unsafe.getThreadBlock().getNativeThread();

        SizeControl this_size = st.sizes[ndx];
        BlockControl alloc_block = blocks[this_size.current_block];

        synchronized (sysLockBlock) {
            if (first_freeblock == OUT_OF_BLOCKS) {
                return -1;
            }

            alloc_block.nextblock = first_freeblock;
            alloc_block = blocks[first_freeblock];
            first_freeblock = alloc_block.nextblock; // new first_freeblock
        }

        alloc_block.nextblock = OUT_OF_BLOCKS;
        // this is last block in list for thissize
        alloc_block.slotsize = GC_SIZEVALUES[ndx];
        size = GC_BLOCKSIZE / GC_SIZEVALUES[ndx];

        // on first assignment of this block, get space from mallocHeap
        // for alloc array, for the size requested.  
        // If not first assignment, if the existing array is large enough for 
        // the new size, use them; else free the existing one, and get space 
        // for new one.  Keep the size for the currently allocated array in
        // alloc_block.alloc_size.  This value only goes up during the running
        // of the VM.

        if (alloc_block.mark != null) {
            if (size <= alloc_block.alloc_size) {
                ObjectLayoutMethods.setArrayLength(alloc_block.mark, size);
                return 0;
            } else { // free the existing make array
                mallocHeap.atomicFreeArray(alloc_block.mark);
            }
        }

        alloc_block.mark = (byte[]) mallocHeap.atomicAllocateArray(jq_Array.BYTE_ARRAY, size);
        return 0;
    }

    protected int getndx(int size) {
        if (size <= GC_SIZEVALUES[0])
            return 0; // special case most common
        if (size <= GC_SIZEVALUES[1])
            return 1; // special case most common
        if (size <= GC_SIZEVALUES[2])
            return 2; // special case most common
        if (size <= GC_SIZEVALUES[3])
            return 3; // special case most common
        if (size <= GC_SIZEVALUES[4])
            return 4; // special case most common
        if (size <= GC_SIZEVALUES[5])
            return 5; // special case most common
        if (size <= GC_SIZEVALUES[6])
            return 6; // special case most common
        if (size <= GC_SIZEVALUES[7])
            return 7; // special case most common
        for (int i = 8; i < GC_SIZES; i++)
            if (size <= GC_SIZEVALUES[i])
                return i;
        return -1;
    }

    //  a debugging routine: to make sure a pointer is into the give block
    protected boolean isPtrInBlock(HeapAddress ptr, SizeControl the_size) {
        BlockControl the_block = blocks[the_size.current_block];
        HeapAddress base = the_block.baseAddr;
        int offset = ptr.difference(base);
        HeapAddress endofslot = (HeapAddress) ptr.offset(the_block.slotsize);
        if (offset % the_block.slotsize != 0)
            jq.UNREACHABLE("Ptr not to beginning of slot");
        HeapAddress bound = (HeapAddress) base.offset(GC_BLOCKSIZE);
        return ptr.difference(base) >= 0 && endofslot.difference(bound) <= 0;
    }

    void dumpblocks(jq_NativeThread st) {
        Debug.writeln("\n-- Processor ", st.getIndex(), " --");
        for (int i = 0; i < GC_SIZES; i++) {
            Debug.write(" Size ", GC_SIZEVALUES[i], "  ");
            BlockControl the_block = blocks[st.sizes[i].first_block];
            Debug.write(st.sizes[i].first_block);
            while (true) {
                Debug.write("  ", the_block.nextblock);
                if (the_block.nextblock == OUT_OF_BLOCKS)
                    break;
                the_block = blocks[the_block.nextblock];
            }
            Debug.writeln();
        }
    }

    void dumpblocks() {
        jq_NativeThread st = Unsafe.getThreadBlock().getNativeThread();
        Debug.write(first_freeblock, "  is the first freeblock index \n");
        for (int i = 0; i < GC_SIZES; i++) {
            Debug.write(
                i,
                "th SizeControl first_block = ",
                st.sizes[i].first_block);
            Debug.write(" current_block = ", st.sizes[i].current_block, "\n\n");
        }

        for (int i = 0; i < num_blocks; i++) {
            Debug.write(i, "th BlockControl   ");
            Debug.writeln((blocks[i].live) ? "   live  " : "not live  ");
            Debug.writeln("baseaddr = ", blocks[i].baseAddr);
            Debug.writeln("nextblock = ", blocks[i].nextblock);
        }
    }

    void clobber(HeapAddress addr, int length) {
        int value = 0xdeaddead;
        int i;
        for (i = 0; i + 3 < length; i = i + 4)
            addr.offset(i).poke4(value);
    }

    public void clobberfree() {
        jq_NativeThread st = Unsafe.getThreadBlock().getNativeThread();
        for (int i = 0; i < GC_SIZES; i++) {
            BlockControl this_block = blocks[st.sizes[i].first_block];
            byte[] this_alloc = this_block.mark;
            for (int ii = 0; ii < this_alloc.length; ii++) {
                if (this_alloc[ii] == 0)
                    clobber(
                        (HeapAddress) this_block.baseAddr.offset(ii * GC_SIZEVALUES[i]),
                        GC_SIZEVALUES[i]);
            }
            int next = this_block.nextblock;
            while (next != OUT_OF_BLOCKS) {
                this_block = blocks[next];
                this_alloc = this_block.mark;
                for (int ii = 0; ii < this_alloc.length; ii++) {
                    if (this_alloc[ii] == 0)
                        clobber(
                            (HeapAddress) this_block.baseAddr.offset(ii * GC_SIZEVALUES[i]),
                            GC_SIZEVALUES[i]);
                }
                next = this_block.nextblock;
            }
        }

        // Now clobber the free list
        synchronized (sysLockBlock) {
            BlockControl block;
            for (int freeIndex = first_freeblock;
                freeIndex != OUT_OF_BLOCKS;
                freeIndex = block.nextblock) {
                block = blocks[freeIndex];
                clobber(block.baseAddr, GC_BLOCKSIZE);
            }
        }
    }

    public long freeMemory() {
        return freeBlocks() * GC_BLOCKSIZE;
    }

    public long partialBlockFreeMemory() {
        if (verbose >= 2)
            Debug.write(
                "WARNING: partialBlockFreeMemory not implemented; returning 0\n");
        return 0;
    }

    protected int emptyOfCurrentBlock(
        BlockControl the_block,
        HeapAddress current_pointer) {
        int sum = 0;
        while (!current_pointer.isNull()) {
            sum += the_block.slotsize;
            current_pointer = (HeapAddress) current_pointer.peek();
        }
        return sum;
    }

    //  calculate the number of free bytes in a block of slotsize size
    protected int emptyof(int size, byte[] alloc) {
        int total = 0;
        for (int i = 0; i < alloc.length; i++) {
            if (alloc[i] == 0)
                total += GC_SIZEVALUES[size];
        }
        return total;
    }

    // Count all VM_blocks in the chain from the input to the end
    //
    protected int blocksInChain(BlockControl the_block) {
        int next = the_block.nextblock;
        int count = 1;
        while (next != OUT_OF_BLOCKS) {
            count++;
            the_block = blocks[next];
            next = the_block.nextblock;
        }
        return count;
    }

    // Count the blocks in the size_control input from first to current
    // 
    protected int blocksToCurrent(SizeControl the_size) {
        int count = 1;
        if (the_size.first_block == the_size.current_block)
            return 1;
        BlockControl the_block = blocks[the_size.first_block];
        int next = the_block.nextblock;
        while (next != the_size.current_block) {
            count++;
            the_block = blocks[next];
            next = the_block.nextblock;
        }
        return count;
    }

    public void setupProcessor(jq_NativeThread st) {
        int scArraySize = SizeControl._array.getInstanceSize(GC_SIZES);
        int scSize = SizeControl._class.getInstanceSize();
        int regionSize = scArraySize + scSize * GC_SIZES;

        // Allocate objects for processor-local meta data from backing malloc heap.
        //VM.disableGC();
        HeapAddress region = mallocHeap.allocateZeroedMemory(regionSize);
        st.sizes =
            (SizeControl[]) ObjectLayoutMethods.initializeArray(
                region,
                SizeControl._array.getVTable(),
                GC_SIZES,
                scArraySize);
        region = (HeapAddress) region.offset(scArraySize);
        for (int i = 0; i < GC_SIZES; i++) {
            st.sizes[i] =
                (SizeControl) ObjectLayoutMethods.initializeObject(
                    region,
                    SizeControl._class.getVTable(),
                    scSize);
            region = (HeapAddress) region.offset(scSize);
        }

        regionSize = SizeControl._array.getInstanceSize(GC_MAX_SMALL_SIZE + 1);
        region = mallocHeap.allocateZeroedMemory(regionSize);
        st.GC_INDEX_ARRAY =
            (SizeControl[]) ObjectLayoutMethods.initializeArray(
                region,
                SizeControl._array.getVTable(),
                (GC_MAX_SMALL_SIZE >> 2) + 1,
                regionSize);

        //VM.enableGC();

        // Finish setting up the size controls and index arrays
        for (int i = 0; i < GC_SIZES; i++) {
            int ii = getnewblockx(i);
            st.sizes[i].first_block = ii; // 1 block/size initially
            st.sizes[i].current_block = ii;
            st.sizes[i].ndx = i; // to fit into old code
            build_list_for_new_block(blocks[ii], st.sizes[i]);
        }

        // NOTE: we exploit the fact that all allocations are word aligned to
        //       reduce the size of the index array by 4x....
        st.GC_INDEX_ARRAY[0] = st.sizes[0]; // for size = 0
        for (int i = 0, j = 4; i < GC_SIZES; i++) {
            for (; j <= GC_SIZEVALUES[i]; j += 4) {
                st.GC_INDEX_ARRAY[j >> 2] = st.sizes[i];
            }
        }

        st.backingSLHeap = this;
    }

    public int freeBlocks() {
        int i;
        synchronized (sysLockBlock) {
            if (first_freeblock == OUT_OF_BLOCKS) {
                return 0;
            }
            BlockControl the_block = blocks[first_freeblock];
            i = 1;
            int next = the_block.nextblock;
            while (next != OUT_OF_BLOCKS) {
                the_block = blocks[next];
                i++;
                next = the_block.nextblock;
            }
        }

        return i;
    }

    public void postCollectionReport() {
    }

    /**
     * Done by 1 collector thread at start of collection
     */
    public void startCollect() {

        blocks_available = 0; // not decremented during allocation

        for (int i = 0; i < GC_SIZES; i++) {
            int counter = 0;
            int index = partialBlockList[i];
            while (index != OUT_OF_BLOCKS) {
                BlockControl this_block = blocks[index];
                HeapAddress a = HeapAddress.addressOf(this_block.mark);
                SystemInterface.mem_set(a, this_block.mark.length, (byte)0);
                this_block.live = false;
                index = this_block.nextblock;
            }
        }
    }

    public void zeromarks(jq_NativeThread st) {
        int block_count = 0;
        for (int i = 0; i < GC_SIZES; i++) {

            //  NEED TO INITIALIZE THE BLOCK AFTER CURRENT_BLOCK, FOR
            //  EACH SIZE, SINCE THIS WAS NOT DONE DURING MUTATOR EXECUTION
            BlockControl this_block = blocks[st.sizes[i].current_block];

            int next = this_block.nextblock;
            while (next != OUT_OF_BLOCKS) {
                this_block = blocks[next];
                HeapAddress a = HeapAddress.addressOf(this_block.mark);
                SystemInterface.mem_set(a, this_block.mark.length, (byte)0);
                this_block.live = false;
                next = this_block.nextblock;
            }
        }

    }

    void setupallocation(jq_NativeThread st) {
        for (int i = 0; i < GC_SIZES; i++) {
            BlockControl this_block = blocks[st.sizes[i].first_block];
            SizeControl this_size = st.sizes[i];
            // begin scan in 1st block again
            this_size.current_block = this_size.first_block;
            if (!build_list(this_block, this_size))
                this_size.next_slot = HeapAddress.getNull();
        }
    }

    /**
     * Parallel sweep
     */
    public void sweep(CollectorThread mylocal) {
        // Following local variables for free_block logic
        int local_first_free_ndx = OUT_OF_BLOCKS;
        int local_blocks_available = 0;
        int temp;
        BlockControl local_first_free_block = null;

        jq_NativeThread st = Unsafe.getThreadBlock().getNativeThread();
        for (int i = 0; i < GC_SIZES; i++) {
            BlockControl this_block = blocks[st.sizes[i].first_block];
            SizeControl this_size = st.sizes[i];
            // begin scan in 1st block again
            this_size.current_block = this_size.first_block;
            if (!build_list(this_block, this_size))
                this_size.next_slot = HeapAddress.getNull();
            int next = this_block.nextblock;
            this_size.lastBlockToKeep = -1; // GSC
            int blockCounter = 0; // GSC
            int counter = 0;
            while (next != OUT_OF_BLOCKS) {

                BlockControl next_block = blocks[next];
                if (!next_block.live) {
                    if (local_first_free_block == null) {
                        local_first_free_block = next_block;
                    }
                    //  In this stanza, we make the next's next the next of this_block, and put
                    //  original next on the freelist
                    this_block.nextblock = next_block.nextblock;
                    // out of live list
                    next_block.nextblock = local_first_free_ndx;
                    local_first_free_ndx = next;
                    local_blocks_available++;
                } else {
                    // found that next block is live
                    if (++blockCounter == numBlocksToKeep) {
                        // GSC
                        this_size.lastBlockToKeep =
                            this_block.baseAddr.difference(start)
                                / GC_BLOCKSIZE;
                    }
                    this_block = next_block;
                }
                next = this_block.nextblock;
            }
            // this_block -> last block in list, with next==0. remember its
            // index for possible moving of partial blocks to global lists below
            //
            this_size.last_allocated =
                this_block.baseAddr.difference(start) / GC_BLOCKSIZE;
        }

        // Now scan through all blocks on the partial blocks list
        // putting empty ones onto the local list of free blocks
        //
        for (int i = mylocal.gcOrdinal - 1;
            i < GC_SIZES;
            i += CollectorThread.numCollectors()) {
            if (partialBlockList[i] == OUT_OF_BLOCKS)
                continue;
            BlockControl this_block = blocks[partialBlockList[i]];
            temp = this_block.nextblock;
            while (!this_block.live) {
                local_blocks_available++;
                if (local_first_free_block == null) {
                    local_first_free_block = this_block;
                }
                temp = this_block.nextblock;
                this_block.nextblock = local_first_free_ndx;
                local_first_free_ndx =
                    (this_block.baseAddr.difference(start)) / GC_BLOCKSIZE;
                partialBlockList[i] = temp;
                if (temp == OUT_OF_BLOCKS)
                    break;
                this_block = blocks[temp];
            }

            if (temp == OUT_OF_BLOCKS)
                continue;
            int next = this_block.nextblock;
            while (next != OUT_OF_BLOCKS) {
                BlockControl next_block = blocks[next];
                if (!next_block.live) {
                    // In this stanza, we make the next's next the next of this_block, and put
                    // original next on the freelist
                    if (local_first_free_block == null) {
                        local_first_free_block = next_block;
                    }
                    this_block.nextblock = next_block.nextblock;
                    // out of live list
                    next_block.nextblock = local_first_free_ndx;
                    local_first_free_ndx = next;
                    local_blocks_available++;
                } else {
                    this_block = next_block; // live block done
                }
                next = this_block.nextblock;
            }
        }

        // Rendezvous here because below and above, partialBlocklist can be modified
        //CollectorThread.gcBarrier.rendezvous(CollectorThread.MEASURE_WAIT_TIMES);

        synchronized (sysLockFree) { // serialize access to global block data

            // If this processor found empty blocks, add them to global free list
            //
            if (local_first_free_block != null) {
                local_first_free_block.nextblock = first_freeblock;
                first_freeblock = local_first_free_ndx;
                blocks_available += local_blocks_available;
            }

            // Add excess partially full blocks (maybe full ???) blocks
            // of each size to the global list for that size
            //

            for (int i = 0; i < GC_SIZES; i++) {
                SizeControl this_size = st.sizes[i];
                if (this_size.lastBlockToKeep != -1) {
                    BlockControl lastToKeep = blocks[this_size.lastBlockToKeep];
                    int firstToGiveUp = lastToKeep.nextblock;
                    if (firstToGiveUp != OUT_OF_BLOCKS) {
                        blocks[this_size.last_allocated].nextblock =
                            partialBlockList[i];
                        partialBlockList[i] = firstToGiveUp;
                        lastToKeep.nextblock = OUT_OF_BLOCKS;
                    }
                }
            }
        } // release lock on global block data
    }

}

// SimpleAllocator.java, created Mon Feb  5 23:23:19 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import java.lang.reflect.Array;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_InstanceMethod;
import joeq.Class.jq_Reference;
import joeq.Class.jq_Type;
import joeq.Memory.Address;
import joeq.Memory.HeapAddress;
import joeq.Runtime.Debug;
import joeq.Runtime.SystemInterface;
import joeq.Util.Assert;

/**
 * SimpleAllocator
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class SimpleAllocator extends HeapAllocator {

    /**
     * Size of blocks allocated from the OS.
     */
    public static final int BLOCK_SIZE = 2097152;

    /**
     * Maximum memory, in bytes, to be allocated from the OS.
     */
    public static /*final*/ int MAX_MEMORY = 67108864;

    /**
     * Threshold for direct OS allocation.  When an array overflows the current block
     * and is larger than this size, it is allocated directly from the OS.
     */
    public static final int LARGE_THRESHOLD = 262144;

    /**
     * Pointers to the start, current, and end of the heap.
     */
    private HeapAddress heapFirst, heapCurrent, heapEnd;

    /**
     * Pointer to the start of the free list.
     */
    private HeapAddress firstFree;
    
    /**
     * Pointer to the start and current of the large object list.
     */
    private HeapAddress firstLarge, currLarge;
    
    /**
     * Simple work queue for GC.
     */
    private SimpleGCWorkQueue gcWorkQueue = new SimpleGCWorkQueue();
    
    /**
     * Are we currently doing a GC?  For debugging purposes.
     */
    private boolean inGC;

    /**
     * Are we currently doing an allocation?  For debugging purposes.
     */
    private boolean inAlloc;
    
    /**
     * Smallest object size allowed in free list.
     */
    public static final int MIN_SIZE = Math.max(ObjectLayout.OBJ_HEADER_SIZE, 8);
    
    static final HeapAddress allocNewBlock() {
        HeapAddress block = (HeapAddress) SystemInterface.syscalloc(BLOCK_SIZE);
        if (block.isNull()) {
            HeapAllocator.outOfMemory();
        }
        return block;
    }
    
    static final HeapAddress getBlockEnd(HeapAddress block) {
        // At end of memory block:
        //  - one word for pointer to start of next memory block.
        return (HeapAddress) block.offset(BLOCK_SIZE - HeapAddress.size());
    }
    
    static final int getBlockSize(HeapAddress block) {
        return BLOCK_SIZE;
    }
    
    static final void setBlockNext(HeapAddress block, HeapAddress next) {
        block.offset(BLOCK_SIZE - HeapAddress.size()).poke(next);
    }
    
    static final HeapAddress getBlockNext(HeapAddress block) {
        return (HeapAddress) block.offset(BLOCK_SIZE - HeapAddress.size()).peek();
    }
    
    static final HeapAddress getFreeNext(HeapAddress free) {
        return (HeapAddress) free.peek();
    }
    
    static final void setFreeNext(HeapAddress free, HeapAddress next) {
        free.poke(next);
    }
    
    static final int getFreeSize(HeapAddress free) {
        return free.offset(HeapAddress.size()).peek4();
    }
    
    static final void setFreeSize(HeapAddress free, int size) {
        free.offset(HeapAddress.size()).poke4(size);
    }
    
    static final HeapAddress getFreeEnd(HeapAddress free) {
        int size = getFreeSize(free);
        return (HeapAddress) free.offset(size);
    }
    
    static final HeapAddress getLargeNext(HeapAddress large) {
        int size = getLargeSize(large);
        return (HeapAddress) large.offset(size).peek();
    }
    
    static final void setLargeNext(HeapAddress large, HeapAddress next) {
        int size = getLargeSize(large);
        large.offset(size).poke(next);
    }
    
    static final Object getLargeObject(HeapAddress large) {
        Object o = ((HeapAddress) large.offset(ObjectLayout.ARRAY_HEADER_SIZE)).asObject();
        return o;
    }
    
    static final int getLargeSize(HeapAddress large) {
        Object o = ((HeapAddress) large.offset(ObjectLayout.ARRAY_HEADER_SIZE)).asObject();
        int size = getObjectSize(o);
        return size;
    }
    
    /**
     * Perform initialization for this allocator.  This will be called before any other methods.
     * This allocates an initial block of memory from the OS and sets up relevant pointers.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    public void init() throws OutOfMemoryError {
        Assert._assert(!inGC);
        Assert._assert(!inAlloc);
        heapCurrent = heapFirst = allocNewBlock();
        heapEnd = getBlockEnd(heapCurrent);
        firstFree = firstLarge = currLarge = HeapAddress.getNull();
    }

    static boolean TRACE = false;
    
    /**
     * Allocates a new block of memory from the OS, sets the current block to
     * point to it, and makes the new block the current block.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    private void allocateNewBlock() {
        if (TRACE) Debug.writeln("Allocating new memory block.");
        if (totalMemory() >= MAX_MEMORY) {
            HeapAllocator.outOfMemory();
        }
        int size = heapEnd.difference(heapCurrent);
        if (TRACE) Debug.writeln("Size of remaining: ", size);
        if (size >= MIN_SIZE) {
            // Add remainder of this block to the free list.
            setFreeSize(heapCurrent, size);
            HeapAddress curr_p = firstFree;
            HeapAddress prev_p = HeapAddress.getNull();
            if (TRACE) Debug.writeln("Adding free block ", heapCurrent);
            for (;;) {
                if (TRACE) Debug.writeln("Checking ", curr_p);
                if (curr_p.isNull() || heapCurrent.difference(curr_p) < 0) {
                    if (prev_p.isNull()) {
                        firstFree = heapCurrent;
                        if (TRACE) Debug.writeln("New head of free list ", firstFree);
                    } else {
                        setFreeNext(prev_p, heapCurrent);
                        if (TRACE) Debug.writeln("Inserting after ", prev_p);
                    }
                    setFreeNext(heapCurrent, curr_p);
                    break;
                }
                HeapAddress next_p = getFreeNext(curr_p);
                prev_p = curr_p;
                curr_p = next_p;
            }
        }
        heapCurrent = allocNewBlock();
        setBlockNext(heapEnd, heapCurrent);
        heapEnd = getBlockEnd(heapCurrent);
    }

    /**
     * Returns the number of bytes available in the free list.
     * 
     * @return bytes available in the free list
     */
    int getFreeListBytes() {
        int freeListMem = 0;
        HeapAddress p = firstFree;
        while (!p.isNull()) {
            freeListMem += getFreeSize(p);
            p = getFreeNext(p);
        }
        return freeListMem;
    }
    
    /**
     * Returns the number of bytes allocated in the large object list.
     * 
     * @return bytes allocated for large objects
     */
    int getLargeBytes() {
        int largeMem = 0;
        HeapAddress p = firstLarge;
        while (!p.isNull()) {
            largeMem += getLargeSize(p);
            p = getLargeNext(p);
        }
        return largeMem;
    }
    
    /**
     * Returns an estimate of the amount of free memory available.
     *
     * @return bytes of free memory
     */
    public int freeMemory() {
        return getFreeListBytes() + heapEnd.difference(heapCurrent);
    }

    /**
     * Returns an estimate of the total memory allocated (both used and unused).
     *
     * @return bytes of memory allocated
     */
    public int totalMemory() {
        int total = 0;
        HeapAddress ptr = heapFirst;
        while (!ptr.isNull()) {
            total += getBlockSize(ptr);
            ptr = getBlockNext(ptr);
        }
        total += getLargeBytes();
        return total;
    }

    /**
     * Allocate an object with the default alignment.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateObject(int size, Object vtable) throws OutOfMemoryError {
        if (TRACE) {
            Debug.write("Allocating object of size ", size);
            jq_Type type = (jq_Type) ((HeapAddress) HeapAddress.addressOf(vtable).peek()).asObject();
            Debug.write(" type ");
            Debug.write(type.getDesc());
            Debug.writeln(" vtable ", HeapAddress.addressOf(vtable));
        }
        if (inGC) {
            Debug.writeln("BUG! Trying to allocate during GC!");
        }
        if (inAlloc) {
            Debug.writeln("BUG! Trying to allocate during another allocation!");
        }
        inAlloc = true;
        if (size < ObjectLayout.OBJ_HEADER_SIZE) {
            // size overflow! become minus!
            inAlloc = false;
            HeapAllocator.outOfMemory();
        }
        //jq.Assert((size & 0x3) == 0);
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            if (TRACE) Debug.writeln("Not enough free space: ", heapEnd.difference(heapCurrent));
            // try to allocate the object from the free list.
            Object o = allocObjectFromFreeList(size, vtable);
            if (o != null) {
                inAlloc = false;
                return o;
            }
            // not enough space on free list, allocate another memory block.
            allocateNewBlock();
            addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
            heapCurrent = (HeapAddress) heapCurrent.offset(size);
            Assert._assert(heapEnd.difference(heapCurrent) >= 0);
        } else {
            if (TRACE) Debug.writeln("Fast path object allocation: ", addr);
        }
        // fast path
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        inAlloc = false;
        return addr.asObject();
    }

    private HeapAddress allocFromFreeList(int size) {
        TRACE = true;
        // Search free list to find if there is an area that will fit the object.
        HeapAddress prev_p = HeapAddress.getNull();
        HeapAddress curr_p = firstFree;
        if (TRACE) Debug.writeln("Searching free list for block of size: ", size);
        while (!curr_p.isNull()) {
            if (TRACE) Debug.writeln("Looking at block ", curr_p);
            HeapAddress next_p = getFreeNext(curr_p);
            int areaSize = getFreeSize(curr_p);
            if (TRACE) Debug.writeln("Block size ", areaSize);
            if (areaSize >= size) {
                // This area fits!
                if (TRACE) Debug.writeln("Block fits, zeroing ", curr_p);
                // Zero out the memory.
                SystemInterface.mem_set(curr_p, areaSize, (byte) 0);
                // Fix up free list.
                int newSize = areaSize - size;
                if (TRACE) Debug.writeln("New size of block: ", newSize);
                HeapAddress new_next_p;
                if (newSize >= MIN_SIZE) {
                    // Still some space left here.
                    Assert._assert(newSize >= HeapAddress.size() * 2);
                    new_next_p = (HeapAddress) curr_p.offset(size);
                    setFreeNext(new_next_p, next_p);
                    setFreeSize(new_next_p, newSize);
                    if (TRACE) Debug.writeln("Block shrunk, now ", new_next_p);
                } else {
                    // Remainder is too small, skip it.
                    new_next_p = next_p;
                    if (TRACE) Debug.writeln("Result too small, new next ", new_next_p);
                }
                if (prev_p.isNull()) {
                    // New start of free list.
                    firstFree = new_next_p;
                    if (TRACE) Debug.writeln("New start of free list: ", firstFree);
                } else {
                    // Patch previous in free list to point to new location.
                    setFreeNext(prev_p, new_next_p);
                    if (TRACE) Debug.writeln("Inserted after ", prev_p);
                }
                return curr_p;
            }
            prev_p = curr_p;
            curr_p = next_p;
        }
        // Nothing in the free list is big enough!
        if (TRACE) Debug.writeln("Nothing in free list is big enough.");
        return HeapAddress.getNull();
    }
    
    /**
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocObjectFromFreeList(int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = allocFromFreeList(size);
        if (addr.isNull()) {
            // Not enough space in free list, try a GC.
            collect();
            addr = allocFromFreeList(size);
            if (addr.isNull()) {
                // need to allocate new block of memory.
                return null;
            }
        }
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    /**
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocArrayFromFreeList(int length, int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = allocFromFreeList(size);
        if (addr.isNull()) {
            // Not enough space in free list, try a GC.
            collect();
            addr = allocFromFreeList(size);
            if (addr.isNull()) {
                // need to allocate new block of memory.
                return null;
            }
        }
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    /**
     * @param length
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocLargeArray(int length, int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = (HeapAddress) SystemInterface.syscalloc(size + HeapAddress.size());
        if (addr.isNull())
            outOfMemory();
        if (firstLarge.isNull()) {
            firstLarge = currLarge = addr;
        } else {
            setLargeNext(currLarge, addr);
            currLarge = addr;
        }
        addr = (HeapAddress) addr.offset(ObjectLayout.ARRAY_HEADER_SIZE);
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    public static final int getObjectSize(Object o) {
        jq_Reference t = jq_Reference.getTypeOf(o);
        int size;
        if (t.isArrayType()) {
            jq_Array a = (jq_Array) t;
            int length = Array.getLength(o);
            size = a.getInstanceSize(length);
        } else {
            jq_Class c = (jq_Class) t;
            size = c.getInstanceSize();
        }
        size = (size + 3) & ~3; // align size
        return size;
    }
    
    /**
     * Allocate an object such that the first field is 8-byte aligned.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateObjectAlign8(int size, Object vtable) throws OutOfMemoryError {
        heapCurrent = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE).align(3).offset(-ObjectLayout.OBJ_HEADER_SIZE);
        return allocateObject(size, vtable);
    }

    /**
     * Allocate an array with the default alignment.
     * If length is negative, throws NegativeArraySizeException.
     * If the array cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param length length of new array
     * @param size size of array to allocate (including array header), in bytes
     * @param vtable vtable pointer for new array
     * @return new array
     * @throws NegativeArraySizeException if length is negative
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateArray(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        if (length < 0) throw new NegativeArraySizeException(length + " < 0");
        if (TRACE) {
            Debug.write("Allocating array of size ", size);
            jq_Type type = (jq_Type) ((HeapAddress) HeapAddress.addressOf(vtable).peek()).asObject();
            Debug.write(" type ");
            Debug.write(type.getDesc());
            Debug.write(" length ", length);
            Debug.writeln(" vtable ", HeapAddress.addressOf(vtable));
        }
        if (inGC) {
            Debug.writeln("BUG! Trying to allocate during GC!");
        }
        if (inAlloc) {
            Debug.writeln("BUG! Trying to allocate during another allocation!");
        }
        inAlloc = true;
        if (size < ObjectLayout.ARRAY_HEADER_SIZE) {
            // size overflow!
            inAlloc = false;
            HeapAllocator.outOfMemory();
        }
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            if (size > LARGE_THRESHOLD) {
                // special large-object allocation
                return allocLargeArray(length, size, vtable);
            } else {
                Object o = allocArrayFromFreeList(length, size, vtable);
                if (o != null) {
                    inAlloc = false;
                    return o;
                }
                // not enough space on free list, allocate another memory block.
                allocateNewBlock();
                addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
            }
        } else {
            if (TRACE) Debug.writeln("Fast path array allocation: ", addr);
        }
        // fast path
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        inAlloc = false;
        return addr.asObject();
    }

    /**
     * Allocate an array such that the elements are 8-byte aligned.
     * If length is negative, throws NegativeArraySizeException.
     * If the array cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param length length of new array
     * @param size size of array to allocate (including array header), in bytes
     * @param vtable vtable pointer for new array
     * @return new array
     * @throws NegativeArraySizeException if length is negative
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateArrayAlign8(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        heapCurrent = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE).align(3).offset(-ObjectLayout.ARRAY_HEADER_SIZE);
        return allocateArray(length, size, vtable);
    }

    public void collect() {
        TRACE = true;
        if (inGC) {
            if (TRACE) Debug.writeln("BUG! Recursively calling GC!");
            //allocateNewBlock();
            return;
        }
        inGC = true;
        SemiConservative.collect();
        inGC = false;
    }
    
    public void sweep() {
        updateFreeList();
        updateLargeObjectList();
    }

    void scanGCQueue(boolean b) {
        for (;;) {
            HeapAddress o = gcWorkQueue.pull();
            if (SimpleAllocator.TRACE) Debug.writeln("Pulled object from queue: ", HeapAddress.addressOf(o));
            if (o.isNull()) break;
            scanObject(o.asObject(), b);
        }
    }
    
    void updateLargeObjectList() {
        HeapAddress prev_p = HeapAddress.getNull();
        HeapAddress curr_p = firstLarge;
        while (!curr_p.isNull()) {
            Object o = getLargeObject(curr_p);
            HeapAddress next_p = getLargeNext(curr_p);
            int status = HeapAddress.addressOf(o).offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
            if ((status & ObjectLayout.GC_BIT) == 0) {
                if (prev_p.isNull()) {
                    firstLarge = next_p;
                } else {
                    setLargeNext(prev_p, next_p);
                }
                if (curr_p.difference(currLarge) == 0) {
                    currLarge = prev_p;
                }
            }
            prev_p = curr_p;
            curr_p = next_p;
        }
    }
    
    void updateFreeList() {
        if (TRACE) Debug.writeln("Updating free list.");
        // Scan forward through heap, finding unmarked objects and merging free spaces.
        HeapAddress currBlock = heapFirst;
        while (!currBlock.isNull()) {
            HeapAddress currBlockEnd = getBlockEnd(currBlock);
            HeapAddress p = currBlock;
            
            HeapAddress currFree, prevFree;
            prevFree = HeapAddress.getNull();
            currFree = firstFree;
            // Seek to the right place in the free list.
            while (!currFree.isNull() && currFree.difference(p) <= 0) {
                prevFree = currFree;
                currFree = getFreeNext(currFree);
            }
            
            if (TRACE) {
                Debug.write("Visiting block ", currBlock);
                Debug.write("-", currBlockEnd);
                Debug.write(" free list ptr ", prevFree);
                Debug.writeln(",", currFree);
            }
            
            // Walk over current block.
            while (currBlockEnd.difference(p) > 0) {
                
                if (TRACE) Debug.writeln("ptr: ", p);
                
                if (p.difference(currFree) == 0) {
                    // Skip over free chunk.
                    p = getFreeEnd(currFree);
                    prevFree = currFree;
                    currFree = getFreeNext(currFree);
                    if (TRACE) Debug.writeln("Skipped over free, next free=", currFree);
                    continue;
                }
                
                HeapAddress obj;
                
                // This is complicated by the fact that there may be a one-word
                // gap between objects due to 8-byte alignment.
                HeapAddress p1 = (HeapAddress) p.offset(ObjectLayout.OBJ_HEADER_SIZE);
                HeapAddress p3 = (HeapAddress) p.offset(ObjectLayout.ARRAY_HEADER_SIZE);
                
                Address vt1 = (Address) p1.offset(ObjectLayout.VTABLE_OFFSET).peek();
                Address vt3 = (Address) p3.offset(ObjectLayout.VTABLE_OFFSET).peek();
                
                boolean b1 = isValidVTable(vt1);
                boolean b3 = isValidArrayVTable(vt3);
                
                boolean skipWord;
                
                if (b1) {
                    // Looks like a scalar object at p1.
                    if (TRACE) Debug.writeln("Looks like scalar object at ", p1);
                    if (b3) {
                        // p1 could also be status word of array at p3.
                        if (TRACE) Debug.writeln("More likely array at ", p3);
                        obj = p3;
                        skipWord = false;
                    } else {
                        // todo: p1 could also possibly be status word of aligned object.
                        obj = p1;
                        skipWord = false;
                    }
                } else {
                    HeapAddress p4 = (HeapAddress) p.offset(4+ObjectLayout.ARRAY_HEADER_SIZE);
                    Address vt4 = (Address) p4.offset(ObjectLayout.VTABLE_OFFSET).peek();
                    boolean b4 = isValidArrayVTable(vt4);
                    if (b3) {
                        // Looks like array object at p3.
                        // todo: p3 could also possibly be status word of aligned array.
                        if (TRACE) Debug.writeln("Looks like array at ", p3);
                        obj = p3;
                        skipWord = false;
                    } else if (b4) {
                        // Looks like aligned array at p4.
                        if (TRACE) Debug.writeln("Looks like aligned array at ", p4);
                        obj = p4;
                        skipWord = true;
                    } else {
                        // Heap is corrupted or we reached the end of this block.
                        break;
                    }
                }
                
                Object o = obj.asObject();
                int size = getObjectSize(o);
                
                HeapAddress next_p = (HeapAddress) p.offset(size + (skipWord?4:0)).align(2);
                if (TRACE) Debug.writeln("Next ptr ", next_p);
                
                int status = obj.offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
                if ((status & ObjectLayout.GC_BIT) == 0) {
                    if (TRACE) Debug.writeln("Not marked, adding to free list.");
                    HeapAddress prevFreeEnd = getFreeEnd(prevFree);
                    if (p.difference(prevFreeEnd) == 0) {
                        // Just extend size of this free area.
                        int newSize = next_p.difference(prevFree);
                        setFreeSize(prevFree, newSize);
                        if (TRACE) Debug.writeln("Free area extended to size ", newSize);
                    } else {
                        // Insert into free list.
                        setFreeNext(p, currFree);
                        int newSize = next_p.difference(p);
                        setFreeSize(p, newSize);
                        if (prevFree.isNull()) {
                            firstFree = p;
                            if (TRACE) Debug.writeln("New first free area ", p);
                        } else {
                            setFreeNext(prevFree, p);
                            if (TRACE) Debug.writeln("Inserted free area ", p);
                        }
                    }
                }
                
                p = next_p;
            }
            currBlock = (HeapAddress) currBlock.offset(BLOCK_SIZE - 2 * HeapAddress.size()).peek();
        }
    }
    
    void scanObject(Object obj, boolean b) {
        jq_Reference type = jq_Reference.getTypeOf(obj);
        if (type.isClassType()) {
            if (SimpleAllocator.TRACE) Debug.writeln("Scanning object ", HeapAddress.addressOf(obj));
            int[] referenceOffsets = ((jq_Class)type).getReferenceOffsets();
            for (int i = 0, n = referenceOffsets.length; i < n; i++) {
                HeapAddress objRef = HeapAddress.addressOf(obj);
                if (SimpleAllocator.TRACE) Debug.writeln("Scanning offset ", referenceOffsets[i]);
                DefaultHeapAllocator.processPtrField(objRef.offset(referenceOffsets[i]), b);
            }
        } else {
            if (SimpleAllocator.TRACE) Debug.writeln("Scanning array ", HeapAddress.addressOf(obj));
            jq_Type elementType = ((jq_Array)type).getElementType();
            if (elementType.isReferenceType()) {
                int num_elements = Array.getLength(obj);
                int numBytes = num_elements * HeapAddress.size();
                HeapAddress objRef = HeapAddress.addressOf(obj);
                HeapAddress location = (HeapAddress) objRef.offset(ObjectLayout.ARRAY_ELEMENT_OFFSET);
                HeapAddress end = (HeapAddress) location.offset(numBytes);
                while (location.difference(end) < 0) {
                    if (SimpleAllocator.TRACE) Debug.writeln("Scanning address ", location);
                    DefaultHeapAllocator.processPtrField(location, b);
                    location =
                        (HeapAddress) location.offset(HeapAddress.size());
                }
            }
        }
    }
    
    public void processPtrField(Address a, boolean b) {
        if (!DefaultHeapAllocator.isValidAddress(a)) {
            if (SimpleAllocator.TRACE) Debug.writeln("Address not valid, skipping: ", a);
            return;
        }
        a = a.peek();
        if (SimpleAllocator.TRACE) Debug.writeln("Checking if valid object ref::: ", a);
        if (!isValidObjectRef(a)) {
            if (SimpleAllocator.TRACE) Debug.writeln("Not a valid object, skipping: ", a);
            return;
        }
        if (SimpleAllocator.TRACE) Debug.writeln("Adding object to queue: ", a);
        gcWorkQueue.addToQueue(a, b);
    }
    
    public boolean isInHeap(Address a) {
        HeapAddress p = heapFirst;
        while (!p.isNull()) {
            int diff = a.difference(p);
            if (diff >= 0 && diff < BLOCK_SIZE - 2 * HeapAddress.size())
                return true;
            p = (HeapAddress) p.offset(BLOCK_SIZE - HeapAddress.size()).peek();
        }
        return false;
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceMethod _allocateObject;
    public static final jq_InstanceMethod _allocateObjectAlign8;
    public static final jq_InstanceMethod _allocateArray;
    public static final jq_InstanceMethod _allocateArrayAlign8;

    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Allocator/SimpleAllocator;");
        _allocateObject = _class.getOrCreateInstanceMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = _class.getOrCreateInstanceMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = _class.getOrCreateInstanceMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = _class.getOrCreateInstanceMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }

}

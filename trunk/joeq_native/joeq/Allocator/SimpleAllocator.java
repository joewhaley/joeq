/*
 * SimpleAllocator.java
 *
 * Created on January 1, 2001, 9:41 PM
 *
 */

package Allocator;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceMethod;
import GC.GCBits;
import Main.jq;
import Memory.Address;
import Memory.HeapAddress;
import Run_Time.SystemInterface;

/*
 * @author  John Whaley
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
     * GC information for the current block.
     */
    private GCBits gcBits;

    private FreeMemManager fmm = new FreeMemManager();

    /**
     * Perform initialization for this allocator.  This will be called before any other methods.
     * This allocates an initial block of memory from the OS and sets up relevant pointers.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    public final void init() throws OutOfMemoryError {
        heapCurrent = heapFirst = (HeapAddress) SystemInterface.syscalloc(BLOCK_SIZE);
        if (heapCurrent.isNull())
            HeapAllocator.outOfMemory();
        heapEnd = (HeapAddress) heapFirst.offset(BLOCK_SIZE - 2 * HeapAddress.size());
        installGCBits();
    }

    /**
     * Allocates a new block of memory from the OS, sets the current block to
     * point to it, makes the new block the current block, and allocates and
     * installs the GC structures.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    private void allocateNewBlock() {
        if (totalMemory() >= MAX_MEMORY) HeapAllocator.outOfMemory();
        heapCurrent = (HeapAddress) SystemInterface.syscalloc(BLOCK_SIZE);
        if (heapCurrent.isNull())
            HeapAllocator.outOfMemory();
        // GCBits address of current block already filled
        heapEnd.offset(HeapAddress.size()).poke(heapCurrent);
        // address for per block GCBits plus address for next block
        heapEnd = (HeapAddress) heapCurrent.offset(BLOCK_SIZE - 2 * HeapAddress.size());
        installGCBits();
    }

    /**
     * Allocates and installs the GC structures for the current block.
     */
    private void installGCBits() {
        // install gc bits for next block.
        gcBits = null;
        // NOTE: allocations caused by the next line don't have gc bits set.
        gcBits = new GCBits(heapCurrent, heapEnd);
        heapEnd.poke(HeapAddress.addressOf(gcBits));
    }

    /**
     * Returns an estimate of the amount of free memory available.
     *
     * @return bytes of free memory
     */
    public final int freeMemory() {
        return heapEnd.difference(heapCurrent);
    }

    /**
     * Returns an estimate of the total memory allocated (both used and unused).
     *
     * @return bytes of memory allocated
     */
    public final int totalMemory() {
        int total = 0;
        HeapAddress ptr = heapFirst;
        while (!ptr.isNull()) {
            total += BLOCK_SIZE;
            ptr = (HeapAddress) ptr.offset(BLOCK_SIZE - HeapAddress.size()).peek();
        }
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
    public final Object allocateObject(int size, Object vtable) throws OutOfMemoryError {
        if (size < ObjectLayout.OBJ_HEADER_SIZE) // size overflow! become minus!
            HeapAllocator.outOfMemory();
        //jq.Assert((size & 0x3) == 0);
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            jq.Assert(size < BLOCK_SIZE - 2 * HeapAddress.size());
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            // consult FreeMemManager for free mem in previous blocks
            addr = fmm.getFreeMem(size + ObjectLayout.OBJ_HEADER_SIZE);
            if (addr != null) {
                addr = (HeapAddress) addr.offset(ObjectLayout.OBJ_HEADER_SIZE);
            } else { // allocate new block and register left free mem in FMM
                //fmm.addFreeMem(new MemUnit(heapCurrent, heapEnd.difference(heapCurrent)));
                allocateNewBlock();
                addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
                heapCurrent = (HeapAddress) heapCurrent.offset(size);
            }
        }
        // fast path
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        if (gcBits != null) {
            gcBits.set((HeapAddress) addr.offset(-ObjectLayout.OBJ_HEADER_SIZE));
        }
        return addr.asObject();
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
    public final Object allocateObjectAlign8(int size, Object vtable) throws OutOfMemoryError {
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
    public final Object allocateArray(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        if (length < 0) throw new NegativeArraySizeException(length + " < 0");
        if (size < ObjectLayout.ARRAY_HEADER_SIZE) // size overflow!
            HeapAllocator.outOfMemory();
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            if (size > LARGE_THRESHOLD) {
                // special large-object allocation
                addr = (HeapAddress) SystemInterface.syscalloc(size);
                if (addr.isNull())
                    outOfMemory();
                addr = (HeapAddress) addr.offset(ObjectLayout.ARRAY_HEADER_SIZE);
                addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
                addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
                // TODO: gc ?!?
                return addr.asObject();
            } else {
                jq.Assert(size < BLOCK_SIZE - 2 * HeapAddress.size());
                // consult FreeMemManager for free mem in previous blocks
                addr = fmm.getFreeMem(size + ObjectLayout.OBJ_HEADER_SIZE);
                if (addr != null) {
                    addr = (HeapAddress) addr.offset(ObjectLayout.ARRAY_HEADER_SIZE);
                } else { // allocate new block and register left free mem in FMM
                    //fmm.addFreeMem(new MemUnit(heapCurrent, heapEnd.difference(heapCurrent)));
                    allocateNewBlock();
                    addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
                    heapCurrent = (HeapAddress) heapCurrent.offset(size);
                }
            }
        }
        // fast path
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        if (gcBits != null) {
            gcBits.set((HeapAddress) addr.offset(-ObjectLayout.ARRAY_HEADER_SIZE));
        }
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
    public final Object allocateArrayAlign8(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        heapCurrent = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE).align(3).offset(-ObjectLayout.ARRAY_HEADER_SIZE);
        return allocateArray(length, size, vtable);
    }

    public final void collect() {
        // nothing for now.
    }

    public final void processPtrField(Address a) {
        // nothing for now.
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceMethod _allocateObject;
    public static final jq_InstanceMethod _allocateObjectAlign8;
    public static final jq_InstanceMethod _allocateArray;
    public static final jq_InstanceMethod _allocateArrayAlign8;

    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/SimpleAllocator;");
        _allocateObject = _class.getOrCreateInstanceMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = _class.getOrCreateInstanceMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = _class.getOrCreateInstanceMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = _class.getOrCreateInstanceMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

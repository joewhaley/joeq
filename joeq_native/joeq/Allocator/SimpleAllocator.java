/*
 * SimpleAllocator.java
 *
 * Created on January 1, 2001, 9:41 PM
 *
 */

package Allocator;

import Clazz.jq_InstanceMethod;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Bootstrap.PrimordialClassLoader;
import Main.jq;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import java.lang.reflect.Array;

/*
 * @author  jwhaley
 * @version 
 */
public class SimpleAllocator extends HeapAllocator {

    /** Size of blocks allocated from the OS.
     */
    public static final int BLOCK_SIZE = 2097152;
    
    /** Maximum memory, in bytes, to be allocated from the OS.
     */
    public static final int MAX_MEMORY = 67108864;
    
    /** Threshold for direct OS allocation.  When an array overflows the current block
     * and is larger than this size, it is allocated directly from the OS.
     */
    public static final int LARGE_THRESHOLD = 262144;
    
    /** Pointers to the start, current, and end of the heap.
     */
    private int/*HeapAddress*/ heapFirst, heapCurrent, heapEnd;
    
    /** Perform initialization for this allocator.  This will be called before any other methods.
     * This allocates an initial block of memory from the OS and sets up relevant pointers.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    public final void init() 
    throws OutOfMemoryError {
        if (0 == (heapCurrent = heapFirst = SystemInterface.syscalloc(BLOCK_SIZE)))
            HeapAllocator.outOfMemory();
        heapEnd = heapFirst + BLOCK_SIZE - 4;
    }
    
    /** Returns an estimate of the amount of free memory available.
     *
     * @return bytes of free memory
     */
    public final int freeMemory() {
        return heapEnd - heapCurrent;
    }
    
    /** Returns an estimate of the total memory allocated (both used and unused).
     *
     * @return bytes of memory allocated
     */
    public final int totalMemory() {
        int total = 0;
        int/*HeapAddress*/ ptr = heapFirst;
        while (ptr != 0) {
            total += BLOCK_SIZE;
            ptr = Unsafe.peek(ptr+BLOCK_SIZE-4);
        }
        return total;
    }
    
    /** Allocate an object with the default alignment.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public final Object allocateObject(int size, Object vtable)
    throws OutOfMemoryError {
	if (size < OBJ_HEADER_SIZE) // size overflow!
	    HeapAllocator.outOfMemory();
        jq.Assert((size & 0x3) == 0);
        int/*HeapAddress*/ addr = heapCurrent + OBJ_HEADER_SIZE;
        heapCurrent += size;
        if (heapCurrent > heapEnd) {
            // not enough space (rare path)
            if (totalMemory() >= MAX_MEMORY) HeapAllocator.outOfMemory();
            jq.Assert(size < BLOCK_SIZE-4);
            if (0 == (heapCurrent = SystemInterface.syscalloc(BLOCK_SIZE)))
                HeapAllocator.outOfMemory();
            Unsafe.poke4(heapEnd, heapCurrent);
            heapEnd = heapCurrent + BLOCK_SIZE - 4;
            addr = heapCurrent + OBJ_HEADER_SIZE;
            heapCurrent += size;
        }
        // fast path
        Unsafe.poke4(addr+VTABLE_OFFSET, Unsafe.addressOf(vtable));
        return Unsafe.asObject(addr);
    }
    
    /** Allocate an object such that the first field is 8-byte aligned.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public final Object allocateObjectAlign8(int size, Object vtable)
    throws OutOfMemoryError {
        heapCurrent = ((heapCurrent+OBJ_HEADER_SIZE+7)&~7)-OBJ_HEADER_SIZE;
        return allocateObject(size, vtable);
    }
    
    /** Allocate an array with the default alignment.
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
    public final Object allocateArray(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException {
        if (length < 0) throw new NegativeArraySizeException(length+" < 0");
	if (size < ARRAY_HEADER_SIZE) // size overflow!
	    HeapAllocator.outOfMemory();
        size = (size+3)&~3; // align size
        int/*HeapAddress*/ addr = heapCurrent + ARRAY_HEADER_SIZE;
        heapCurrent += size;
        if (heapCurrent > heapEnd) {
            // not enough space (rare path)
            if (size > LARGE_THRESHOLD) {
                // special large-object allocation
                heapCurrent -= size;
                if (0 == (addr = SystemInterface.syscalloc(size)))
                    outOfMemory();
                addr += ARRAY_HEADER_SIZE;
            } else {
                if (totalMemory() >= MAX_MEMORY) HeapAllocator.outOfMemory();
                jq.Assert(size < BLOCK_SIZE-4);
                if (0 == (heapCurrent = SystemInterface.syscalloc(BLOCK_SIZE)))
                    outOfMemory();
                Unsafe.poke4(heapEnd, heapCurrent);
                heapEnd = heapCurrent + BLOCK_SIZE - 4;
                addr = heapCurrent + ARRAY_HEADER_SIZE;
                heapCurrent += size;
            }
        }
        // fast path
        Unsafe.poke4(addr+ARRAY_LENGTH_OFFSET, length);
        Unsafe.poke4(addr+VTABLE_OFFSET, Unsafe.addressOf(vtable));
        return Unsafe.asObject(addr);
    }
    
    /** Allocate an array such that the elements are 8-byte aligned.
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
    public final Object allocateArrayAlign8(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException {
        heapCurrent = ((heapCurrent+ARRAY_HEADER_SIZE+7)&~7)-ARRAY_HEADER_SIZE;
        return allocateArray(length, size, vtable);
    }

    public static final jq_Class _class;
    public static final jq_InstanceMethod _allocateObject;
    public static final jq_InstanceMethod _allocateObjectAlign8;
    public static final jq_InstanceMethod _allocateArray;
    public static final jq_InstanceMethod _allocateArrayAlign8;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/SimpleAllocator;");
        _allocateObject = _class.getOrCreateInstanceMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = _class.getOrCreateInstanceMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = _class.getOrCreateInstanceMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = _class.getOrCreateInstanceMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

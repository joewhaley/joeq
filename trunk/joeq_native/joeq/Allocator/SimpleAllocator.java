/*
 * SimpleAllocator.java
 *
 * Created on January 1, 2001, 9:41 PM
 *
 * @author  jwhaley
 * @version 
 */

package Allocator;

import Clazz.jq_InstanceMethod;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Bootstrap.PrimordialClassLoader;
import jq;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import java.lang.reflect.Array;

public class SimpleAllocator extends HeapAllocator {

    /** Size of blocks allocated from the OS.
     */
    public static final int BLOCK_SIZE = 2097152;
    
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
        jq.assert(size >= OBJ_HEADER_SIZE);
        int/*HeapAddress*/ addr = heapCurrent + OBJ_HEADER_SIZE;
        heapCurrent += size;
        if (heapCurrent > heapEnd) {
            // not enough space (rare path)
            jq.assert(size < BLOCK_SIZE-4);
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
        int mask = (heapCurrent+OBJ_HEADER_SIZE) & 7;
        if (mask != 0) heapCurrent += 8-mask;
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
        jq.assert(size >= ARRAY_HEADER_SIZE);
        int mask = size & 3;
        if (mask != 0) size += 3-mask;
        int/*HeapAddress*/ addr = heapCurrent + ARRAY_HEADER_SIZE;
        heapCurrent += size;
        if (heapCurrent > heapEnd) {
            // not enough space (rare path)
            if (size > LARGE_THRESHOLD) {
                // special large-object allocation
                if (0 == (addr = SystemInterface.syscalloc(BLOCK_SIZE)))
                    outOfMemory();
                addr += ARRAY_HEADER_SIZE;
            } else {
                jq.assert(size < BLOCK_SIZE-4);
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
        int mask = (heapCurrent+ARRAY_HEADER_SIZE) & 7;
        if (mask != 0) heapCurrent += 8-mask;
        return allocateArray(length, size, vtable);
    }

    public static final jq_InstanceMethod _allocateObject;
    public static final jq_InstanceMethod _allocateObjectAlign8;
    public static final jq_InstanceMethod _allocateArray;
    public static final jq_InstanceMethod _allocateArrayAlign8;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/SimpleAllocator;");
        _allocateObject = k.getOrCreateInstanceMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = k.getOrCreateInstanceMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = k.getOrCreateInstanceMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = k.getOrCreateInstanceMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

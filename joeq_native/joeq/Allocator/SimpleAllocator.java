/*
 * SimpleAllocator.java
 *
 * Created on January 1, 2001, 9:41 PM
 *
 * @author  jwhaley
 * @version 
 */

package Allocator;

import Clazz.jq_StaticMethod;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Bootstrap.PrimordialClassLoader;
import jq;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import java.lang.reflect.Array;

public abstract class SimpleAllocator implements ObjectLayout {

    public static final int BLOCK_SIZE = 2097152;
    public static final int LARGE_THRESHOLD = 262144;
    private static int/*Address*/ heapFirst, heapCurrent, heapEnd;
    private static final OutOfMemoryError outofmemoryerror = new OutOfMemoryError();
    
    private static void outOfMemory()
    throws OutOfMemoryError {
        throw outofmemoryerror;
    }
    
    public static int freeMemory() {
        return heapEnd - heapCurrent;
    }
    
    public static int totalMemory() {
        int total = 0, ptr = heapFirst;
        while (ptr != 0) {
            total += BLOCK_SIZE;
            ptr = Unsafe.peek(ptr+BLOCK_SIZE-4);
        }
        return total;
    }
    
    public static void init() 
    throws OutOfMemoryError {
        if (0 == (heapCurrent = heapFirst = SystemInterface.syscalloc(BLOCK_SIZE)))
            outOfMemory();
        heapEnd = heapFirst + BLOCK_SIZE - 4;
    }
    
    public static Object allocateObject(int size, Object vtable)
    throws OutOfMemoryError {
        jq.assert(size >= OBJ_HEADER_SIZE);
        int addr = heapCurrent + OBJ_HEADER_SIZE;
        heapCurrent += size;
        if (heapCurrent > heapEnd) {
            // not enough space (rare path)
            jq.assert(size < BLOCK_SIZE-4);
            if (0 == (heapCurrent = SystemInterface.syscalloc(BLOCK_SIZE)))
                outOfMemory();
            Unsafe.poke4(heapEnd, heapCurrent);
            heapEnd = heapCurrent + BLOCK_SIZE - 4;
            addr = heapCurrent + OBJ_HEADER_SIZE;
            heapCurrent += size;
        }
        // fast path
        Unsafe.poke4(addr+VTABLE_OFFSET, Unsafe.addressOf(vtable));
        return Unsafe.asObject(addr);
    }
    
    public static Object allocateArray(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException {
        // TODO: alignment for long/double arrays
        if (length < 0) throw new NegativeArraySizeException(length+" < 0");
        jq.assert(size >= ARRAY_HEADER_SIZE);
        int addr = heapCurrent + ARRAY_HEADER_SIZE;
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

    public static Object clone(Object o) {
        jq_Reference t = Unsafe.getTypeOf(o);
        if (t.isClassType()) {
            jq_Class k = (jq_Class)t;
            Object p = allocateObject(k.getInstanceSize(), k.getVTable());
            if (k.getInstanceSize()-OBJ_HEADER_SIZE > 0)
                SystemInterface.memcpy(Unsafe.addressOf(p), Unsafe.addressOf(o), k.getInstanceSize()-OBJ_HEADER_SIZE);
            return p;
        } else {
            jq.assert(t.isArrayType());
            jq_Array k = (jq_Array)t;
            int length = Array.getLength(o);
            Object p = allocateArray(length, k.getInstanceSize(length), k.getVTable());
            if (length > 0)
                SystemInterface.memcpy(Unsafe.addressOf(p), Unsafe.addressOf(o), k.getInstanceSize(length)-ARRAY_HEADER_SIZE);
            return p;
        }
    }
    
    public static final jq_StaticMethod _allocateObject;
    public static final jq_StaticMethod _allocateArray;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/SimpleAllocator;");
        _allocateObject = k.getOrCreateStaticMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = k.getOrCreateStaticMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

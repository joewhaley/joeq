/*
 * DefaultHeapAllocator.java
 *
 * Created on February 8, 2001, 3:26 PM
 *
 * @author  John Whaley
 * @version 
 */

package Allocator;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_StaticMethod;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;

public abstract class DefaultHeapAllocator {

    public static final HeapAllocator def() {
        return Unsafe.getThreadBlock().getNativeThread().getHeapAllocator();
    }
    
    public static final void init() throws OutOfMemoryError {
        def().init();
    }
    public static final Object allocateObject(int size, Object vtable) throws OutOfMemoryError {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = def().allocateObject(size, vtable);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }
    public static final Object allocateObjectAlign8(int size, Object vtable) throws OutOfMemoryError {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = def().allocateObjectAlign8(size, vtable);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }
    public static final Object allocateArray(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = def().allocateArray(length, size, vtable);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }
    public static final Object allocateArrayAlign8(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = def().allocateArrayAlign8(length, size, vtable);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }
    public static final int freeMemory() { return def().freeMemory(); }
    public static final int totalMemory() { return def().totalMemory(); }

    public static final jq_StaticMethod _allocateObject;
    public static final jq_StaticMethod _allocateObjectAlign8;
    public static final jq_StaticMethod _allocateArray;
    public static final jq_StaticMethod _allocateArrayAlign8;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/DefaultHeapAllocator;");
        _allocateObject = k.getOrCreateStaticMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = k.getOrCreateStaticMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = k.getOrCreateStaticMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = k.getOrCreateStaticMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

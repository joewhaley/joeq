/*
 * DefaultAllocator.java
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

public abstract class DefaultAllocator {

    public static final HeapAllocator DEFAULT = new SimpleAllocator();
    public static final void init()
    throws OutOfMemoryError { DEFAULT.init(); }
    public static final Object allocateObject(int size, Object vtable)
    throws OutOfMemoryError { return DEFAULT.allocateObject(size, vtable); }
    public static final Object allocateObjectAlign8(int size, Object vtable)
    throws OutOfMemoryError { return DEFAULT.allocateObjectAlign8(size, vtable); }
    public static final Object allocateArray(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException { return DEFAULT.allocateArray(length, size, vtable); }
    public static final Object allocateArrayAlign8(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException { return DEFAULT.allocateArrayAlign8(length, size, vtable); }
    public static final int freeMemory() { return DEFAULT.freeMemory(); }
    public static final int totalMemory() { return DEFAULT.totalMemory(); }

    public static final jq_StaticMethod _allocateObject;
    public static final jq_StaticMethod _allocateObjectAlign8;
    public static final jq_StaticMethod _allocateArray;
    public static final jq_StaticMethod _allocateArrayAlign8;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/DefaultAllocator;");
        _allocateObject = k.getOrCreateStaticMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = k.getOrCreateStaticMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = k.getOrCreateStaticMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = k.getOrCreateStaticMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

/*
 * Allocate.java
 *
 * Created on January 16, 2001, 9:45 PM
 *
 * @author  jwhaley
 * @version 
 */

package Allocator;

import Clazz.jq_Type;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_Array;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import jq;

import java.lang.reflect.Array;
import java.util.Set;
import java.util.HashSet;

public abstract class HeapAllocator implements jq_ClassFileConstants, ObjectLayout {
    
    //// ABSTRACT METHODS THAT ALLOCATORS NEED TO IMPLEMENT.
    
    /** Perform initialization for this allocator.  This will be called before any other methods.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    public abstract void init()
    throws OutOfMemoryError;
    
    /** Allocate an object with the default alignment.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public abstract Object allocateObject(int size, Object vtable)
    throws OutOfMemoryError;
    
    /** Allocate an object such that the first field is 8-byte aligned.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public abstract Object allocateObjectAlign8(int size, Object vtable)
    throws OutOfMemoryError;
    
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
    public abstract Object allocateArray(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException;
    
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
    public abstract Object allocateArrayAlign8(int length, int size, Object vtable)
    throws OutOfMemoryError, NegativeArraySizeException;
    
    /** Returns an estimate of the amount of free memory available.
     *
     * @return bytes of free memory
     */
    public abstract int freeMemory();
    
    /** Returns an estimate of the total memory allocated (both used and unused).
     *
     * @return bytes of memory allocated
     */
    public abstract int totalMemory();
    
    //// STATIC, ALLOCATION-RELATED HELPER METHODS.
    
    /** Initialize class t and return a new uninitialized object of that type.
     * If t is not a class type, throw a VerifyError.
     *
     * @param t type to initialize and create object of
     * @return new uninitialized object of type t
     * @throws VerifyError if t is not a class type
     */
    public static Object clsinitAndAllocateObject(jq_Type t)
    throws VerifyError {
        if (!t.isClassType())
            throw new VerifyError();
        jq_Class k = (jq_Class)t;
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        return k.newInstance();
    }
    
    /** Allocate a multidimensional array with dim dimensions and array type f.
     * dim dimensions are read from the stack frame.  (NOTE: this method does NOT
     * reset the stack pointer for the dimensions arguments!  The caller must handle it!)
     * If f is not an array type, throws VerifyError.
     *
     * @return allocated array object
     * @param dim number of dimensions to allocate.  f must be an array type of at least this dimensionality.
     * @param f type of array
     * @throws VerifyError if t is not a array type of dimensionality at least dim
     * @throws OutOfMemoryError if there is not enough memory to perform operation
     * @throws NegativeArraySizeException if a dimension is negative
     */    
    public static Object multinewarray(char dim, jq_Type f/*, ... */) 
    throws OutOfMemoryError, NegativeArraySizeException, VerifyError {
        if (!f.isArrayType())
            throw new VerifyError();
        jq_Array a = (jq_Array)f;
        a.load(); a.verify(); a.prepare(); a.sf_initialize(); a.cls_initialize();
        if (a.getDimensionality() < dim)
            throw new VerifyError();
        int[] n_elem = new int[dim];
        int/*StackAddress*/ p = Unsafe.EBP() + 16;
        for (int i=dim-1; i>=0; --i) {
            n_elem[i] = Unsafe.peek(p);
	    // check for dim < 0 here, because if a dim is zero, later dim's
	    // are not checked by multinewarray_helper.
	    if (n_elem[i] < 0)
		throw new NegativeArraySizeException("dim "+i+": "+n_elem[i]+" < 0");
            p += 4;
        }
        return multinewarray_helper(n_elem, 0, a);
    }
    
    /** Allocates a multidimensional array of type a, with dimensions given in
     * dims[ind] to dims[dims.length-1].  a must be of dimensionality at least
     * dims.length-ind.
     *
     * @return allocated array object
     * @param dims array of dimensions
     * @param ind start index in array dims
     * @param a array type
     * @throws NegativeArraySizeException if one of the array sizes in dims is negative
     * @throws OutOfMemoryError if there is not enough memory to perform operation
     */    
    public static Object multinewarray_helper(int[] dims, int ind, jq_Array a)
    throws OutOfMemoryError, NegativeArraySizeException {
        a.chkState(STATE_CLSINITIALIZED);
        int length = dims[ind];
        Object o = a.newInstance(length);
	jq.assert(length >= 0);
        if (ind == dims.length-1)
            return o;
        Object[] o2 = (Object[])o;
        jq_Array a2 = (jq_Array)a.getElementType();
        a2.load(); a2.verify(); a2.prepare(); a2.sf_initialize(); a2.cls_initialize();
        for (int i=0; i<length; ++i) {
            o2[i] = multinewarray_helper(dims, ind+1, a2);
        }
        return o2;
    }

    /** Clone the given object.  NOTE: Does not check if the object implements Cloneable.
     *
     * @return new clone
     * @param o object to clone
     * @throws OutOfMemoryError if there is not enough memory to perform operation
     */
    public static Object clone(Object o)
    throws OutOfMemoryError {
        jq_Reference t = Unsafe.getTypeOf(o);
        if (t.isClassType()) {
            jq_Class k = (jq_Class)t;
            Object p = k.newInstance();
            if (k.getInstanceSize()-OBJ_HEADER_SIZE > 0)
                SystemInterface.mem_cpy(Unsafe.addressOf(p), Unsafe.addressOf(o), k.getInstanceSize()-OBJ_HEADER_SIZE);
            return p;
        } else {
            jq.assert(t.isArrayType());
            jq_Array k = (jq_Array)t;
            int length = Array.getLength(o);
            Object p = k.newInstance(length);
            if (length > 0)
                SystemInterface.mem_cpy(Unsafe.addressOf(p), Unsafe.addressOf(o), k.getInstanceSize(length)-ARRAY_HEADER_SIZE);
            return p;
        }
    }
    
    private static boolean isOutOfMemory = false;
    private static final OutOfMemoryError outofmemoryerror = new OutOfMemoryError();

    /** Called in an out of memory situation.
     *
     * @throws OutOfMemoryError always thrown
     */    
    public static void outOfMemory()
    throws OutOfMemoryError {
        if (isOutOfMemory) {
            SystemInterface.die(-1);
        }
        isOutOfMemory = true;
        SystemInterface.debugmsg("Out of memory!");
        throw outofmemoryerror;
    }
    
    public static final Set heapAddressFields;
    public static final jq_StaticMethod _clsinitAndAllocateObject;
    public static final jq_StaticMethod _multinewarray;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/HeapAllocator;");
        _clsinitAndAllocateObject = k.getOrCreateStaticMethod("clsinitAndAllocateObject", "(LClazz/jq_Type;)Ljava/lang/Object;");
        _multinewarray = k.getOrCreateStaticMethod("multinewarray", "(CLClazz/jq_Type;)Ljava/lang/Object;");
        heapAddressFields = new HashSet();
        heapAddressFields.add(Assembler.x86.Code2HeapReference._to_heaploc);
        heapAddressFields.add(Assembler.x86.Heap2CodeReference._from_heaploc);
        heapAddressFields.add(Assembler.x86.Heap2HeapReference._from_heaploc);
        heapAddressFields.add(Assembler.x86.Heap2HeapReference._to_heaploc);
        heapAddressFields.add(Clazz.jq_StaticField._address);
    }
    
}

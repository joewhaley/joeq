// HeapAllocator.java, created Tue Feb 27  2:52:57 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import java.lang.reflect.Array;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_ClassFileConstants;
import joeq.Class.jq_Primitive;
import joeq.Class.jq_Reference;
import joeq.Class.jq_StaticField;
import joeq.Class.jq_StaticMethod;
import joeq.Class.jq_Type;
import joeq.Memory.Address;
import joeq.Memory.HeapAddress;
import joeq.Memory.Heap.Heap;
import joeq.Runtime.Debug;
import joeq.Runtime.SystemInterface;
import joeq.Util.Assert;

/**
 * HeapAllocator
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class HeapAllocator implements jq_ClassFileConstants {
    
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
    
    /**
     * Returns whether the given address falls within the boundaries of this heap.
     */
    public abstract boolean isInHeap(Address a);
    
    /**
     * Initiate a garbage collection.
     */
    public abstract void collect();
    
    /**
     * Process a reference to a heap object during garbage collection.
     */
    public abstract void processPtrField(Address a, boolean b);
    
    //// STATIC, ALLOCATION-RELATED HELPER METHODS.
    
    /**
     * Initialize class t and return a new uninitialized object of that type.
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
        k.cls_initialize();
        return k.newInstance();
    }

    /**
     * Clone the given object.  NOTE: Does not check if the object implements Cloneable.
     *
     * @return new clone
     * @param o object to clone
     * @throws OutOfMemoryError if there is not enough memory to perform operation
     */
    public static Object clone(Object o)
    throws OutOfMemoryError {
        jq_Reference t = jq_Reference.getTypeOf(o);
        if (t.isClassType()) {
            jq_Class k = (jq_Class)t;
            Object p = k.newInstance();
            if (k.getInstanceSize()-ObjectLayout.OBJ_HEADER_SIZE > 0)
                SystemInterface.mem_cpy(HeapAddress.addressOf(p), HeapAddress.addressOf(o), k.getInstanceSize()-ObjectLayout.OBJ_HEADER_SIZE);
            return p;
        } else {
            Assert._assert(t.isArrayType());
            jq_Array k = (jq_Array)t;
            int length = Array.getLength(o);
            Object p = k.newInstance(length);
            if (length > 0)
                SystemInterface.mem_cpy(HeapAddress.addressOf(p), HeapAddress.addressOf(o), k.getInstanceSize(length)-ObjectLayout.ARRAY_HEADER_SIZE);
            return p;
        }
    }
    
    /**
     * Handle heap exhaustion.
     * 
     * @param heap the exhausted heap
     * @param size number of bytes requested in the failing allocation
     * @param count the retry count for the failing allocation.
     */
    public static void heapExhausted(Heap heap, int size, int count)
    throws OutOfMemoryError {
        if (count > 3) outOfMemory();
        // TODO: trigger joeq.GC.
    }
    
    private static boolean isOutOfMemory = false;
    private static final OutOfMemoryError outofmemoryerror = new OutOfMemoryError();

    /**
     * Called in an out of memory situation.
     *
     * @throws OutOfMemoryError always thrown
     */    
    public static void outOfMemory()
    throws OutOfMemoryError {
        if (isOutOfMemory) {
            SystemInterface.die(-1);
        }
        isOutOfMemory = true;
        SystemInterface.debugwriteln("Out of memory!");
        throw outofmemoryerror;
    }
    
    public static HeapAddress data_segment_start;
    public static HeapAddress data_segment_end;
    
    public static void initializeDataSegment() {
        // need to initialize these to be non-null so that they will
        // have relocations.
        data_segment_start = (HeapAddress) HeapAddress.getNull().offset(-1);
        data_segment_end = (HeapAddress) HeapAddress.getNull().offset(-1);
    }
    
    public static final boolean isInDataSegment(Address a) {
        boolean b = a.difference(data_segment_start) > 0 &&
                    a.difference(data_segment_end) < 0;
        return b;
    }
    
    public static boolean isValidAddress(Address a) {
        return DefaultHeapAllocator.isValidAddress(a);
    }
    
    public static final boolean TRACE = false;
    
    public static boolean isValidObjectRef(Address a) {
        if (TRACE) Debug.writeln("Checking if valid object ref: ", a);
        if (!isValidAddress(a)) {
            if (TRACE) Debug.writeln("Cannot be object, invalid address");
            return false;
        }
        Address vt = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        boolean b = isValidVTable(vt);
        if (TRACE) {
            if (b) Debug.writeln("Valid object: ", a);
            else Debug.writeln("Cannot be object, invalid vtable: ", vt);
        }
        return b;
    }
    
    public static boolean isValidArrayRef(Address a) {
        if (TRACE) Debug.writeln("Checking if valid array ref: ", a);
        if (!isValidAddress(a)) {
            if (TRACE) Debug.writeln("Cannot be array, invalid address");
            return false;
        }
        Address vt = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        boolean b = isValidArrayVTable(vt);
        if (TRACE) {
            if (b) Debug.writeln("Valid array: ", a);
            else Debug.writeln("Cannot be object, invalid array vtable: ", vt);
        }
        return b;
    }
    
    public static boolean isValidVTable(Address a) {
        if (TRACE) Debug.writeln("Checking if vtable: ", a);
        if (!isValidAddress(a)) {
            if (TRACE) Debug.writeln("Cannot be vtable, invalid address");
            return false;
        }
        Address vtableTypeAddr = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        jq_Reference r = PrimordialClassLoader.getAddressArray();
        if (!isType(vtableTypeAddr, r)) {
            if (TRACE) Debug.writeln("Cannot be vtable, has wrong type: ", vtableTypeAddr);
            return false;
        }
        boolean b = isValidType((HeapAddress) a.peek());
        if (TRACE) {
            if (b) Debug.writeln("Valid vtable: ", a);
            else Debug.writeln("Cannot be vtable, invalid type in vtable[0]: ", a.peek());
        }
        return b;
    }
    
    public static boolean isValidArrayVTable(Address a) {
        if (TRACE) Debug.writeln("Checking if array vtable: ", a);
        if (!isValidAddress(a)) {
            if (TRACE) Debug.writeln("Cannot be array vtable, invalid address");
            return false;
        }
        Address vtableTypeAddr = a.offset(ObjectLayout.VTABLE_OFFSET);
        jq_Reference r = PrimordialClassLoader.getAddressArray();
        if (!isType(vtableTypeAddr, r)) {
            if (TRACE) Debug.writeln("Cannot be array vtable, has wrong type: ", vtableTypeAddr);
            return false;
        }
        boolean b = isValidArrayType((HeapAddress) a.peek());
        if (TRACE) {
            if (b) Debug.writeln("Valid array vtable: ", a);
            else Debug.writeln("Cannot be array vtable, invalid type in vtable[0]: ", a.peek());
        }
        return b;
    }
    
    public static boolean isType(Address a, jq_Reference t) {
        if (TRACE) {
            Debug.writeln("Checking if matching type: ", a);
            Debug.writeln(t.getDesc());
        }
        if (!isValidAddress(a)) {
            if (TRACE) Debug.writeln("Cannot be type, invalid address");
            return false;
        }

        Address vtable = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        if (!isValidAddress(vtable)) {
            if (TRACE) Debug.writeln("Cannot be type, invalid vtable address: ", vtable);
            return false;
        }
        Address type = vtable.peek();
        Address expected = HeapAddress.addressOf(t);
        boolean b = expected.difference(type) == 0;
        if (TRACE) {
            if (b) Debug.writeln("Matching type: ", type);
            else {
                Debug.writeln("Not matching type: ", type);
                Debug.writeln(jq_Reference.getTypeOf(((HeapAddress) a).asObject()).getDesc());
            }
        }
        return b;
    }
    
    public static boolean isValidType(Address typeAddress) {
        if (TRACE) Debug.writeln("Checking if valid type: ", typeAddress);
        if (!isValidAddress(typeAddress)) {
            if (TRACE) Debug.writeln("Cannot be type, invalid address");
            return false;
        }

        // check if vtable is one of three possible values
        Object vtable = ObjectLayoutMethods.getVTable(((HeapAddress) typeAddress).asObject());
        boolean valid = vtable == jq_Class._class.getVTable() ||
                        vtable == jq_Array._class.getVTable() ||
                        vtable == jq_Primitive._class.getVTable();
        if (TRACE) {
            if (valid) Debug.writeln("Matching vtable: ", HeapAddress.addressOf(vtable));
            else Debug.writeln("Not matching vtable: ", HeapAddress.addressOf(vtable));
        }
        return valid;
    }
    
    public static boolean isValidArrayType(Address typeAddress) {
        if (TRACE) Debug.writeln("Checking if valid array type: ", typeAddress);
        if (!isValidAddress(typeAddress)) {
            if (TRACE) Debug.writeln("Cannot be array type, invalid address");
            return false;
        }

        Object vtable = ObjectLayoutMethods.getVTable(((HeapAddress) typeAddress).asObject());
        boolean valid = vtable == jq_Array._class.getVTable();
        if (TRACE) {
            if (valid) Debug.writeln("Matching array vtable: ", HeapAddress.addressOf(vtable));
            else Debug.writeln("Not matching array vtable: ", HeapAddress.addressOf(vtable));
        }
        return valid;
    }
    
    /**
     * An object of this class represents a pointer to a heap address.
     * It is a wrapped version of HeapAddress, so it can be used like
     * an object.
     */
    public static class HeapPointer implements Comparable {
        
        /** The (actual) address. */
        private final HeapAddress ip;
        
        /** Create a new heap pointer.
         * @param ip  heap pointer value
         */
        public HeapPointer(HeapAddress ip) { this.ip = ip; }
        
        /** Extract the address of this heap pointer.
         * @return  address of this heap pointer
         */
        public HeapAddress get() { return ip; }
        
        /** Compare this heap pointer to another heap pointer.
         * @param that  heap pointer to compare against
         * @return  -1 if this ip is before the given ip, 0 if it is equal
         *           to the given ip, 1 if it is after the given ip
         */
        public int compareTo(HeapPointer that) {
            if (this.ip.difference(that.ip) < 0) return -1;
            if (this.ip.difference(that.ip) > 0) return 1;
            return 0;
        }
        
        /** Compares this heap pointer to the given object.
         * @param that  object to compare to
         * @return  -1 if this is less than, 0 if this is equal, 1 if this
         *           is greater than
         */
        public int compareTo(java.lang.Object that) {
            return compareTo((HeapPointer) that);
        }
        
        /** Returns true if this heap pointer refers to the same location
         * as the given heap pointer, false otherwise.
         * @param that  heap pointer to compare to
         * @return  true if the heap pointers are equal, false otherwise
         */
        public boolean equals(HeapPointer that) {
            return this.ip.difference(that.ip) == 0;
        }
        
        /** Compares this heap pointer with the given object.
         * @param that  object to compare with
         * @return  true if these objects are equal, false otherwise
         */
        public boolean equals(Object that) {
            return equals((HeapPointer) that);
        }
        
        /**  Returns the hash code of this heap pointer.
         * @return  hash code
         */
        public int hashCode() { return this.ip.to32BitValue(); }
        
    }
    
    public static final jq_StaticMethod _clsinitAndAllocateObject;
    public static final jq_StaticField _data_segment_start;
    public static final jq_StaticField _data_segment_end;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Allocator/HeapAllocator;");
        _clsinitAndAllocateObject = k.getOrCreateStaticMethod("clsinitAndAllocateObject", "(Ljoeq/Class/jq_Type;)Ljava/lang/Object;");
        _data_segment_start = k.getOrCreateStaticField("data_segment_start", "Ljoeq/Memory/HeapAddress;");
        _data_segment_end = k.getOrCreateStaticField("data_segment_end", "Ljoeq/Memory/HeapAddress;");
    }
}

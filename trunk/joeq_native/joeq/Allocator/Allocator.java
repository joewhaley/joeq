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
import Clazz.jq_StaticMethod;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import jq;

public abstract class Allocator implements jq_ClassFileConstants {
    
    public static Object clsinitAndAllocateObject(jq_Type t) {
        if (!t.isClassType())
            throw new VerifyError();
        jq_Class k = (jq_Class)t;
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        return k.newInstance();
    }
    
    // NOTE: does not pop its own arguments!
    public static Object multinewarray(char dim, jq_Type f) {
        if (!f.isArrayType())
            throw new VerifyError();
        jq_Array a = (jq_Array)f;
        a.load(); a.verify(); a.prepare(); a.sf_initialize(); a.cls_initialize();
        if (a.getDimensionality() < dim)
            throw new VerifyError();
        int[] n_elem = new int[dim];
        int/*Address*/ p = Unsafe.EBP() + 16;
        for (int i=dim-1; i>=0; --i) {
            n_elem[i] = Unsafe.peek(p);
            // NegativeArraySizeException check is in allocateArray
            p += 4;
        }
        return multinewarray_helper(n_elem, 0, a);
    }
    public static Object multinewarray_helper(int[] dims, int ind, jq_Array a) {
        a.chkState(STATE_CLSINITIALIZED);
        int length = dims[ind];
        int size = a.getInstanceSize(length);
        if (ind == dims.length-1) {
            Object o = SimpleAllocator.allocateArray(length, size, a.getVTable());
            return o;
        }
        Object[] o = (Object[])SimpleAllocator.allocateArray(length, size, a.getVTable());
        jq_Array a2 = (jq_Array)a.getElementType();
        a2.load(); a2.verify(); a2.prepare(); a2.sf_initialize(); a2.cls_initialize();
        for (int i=0; i<length; ++i) {
            o[i] = multinewarray_helper(dims, ind+1, a2);
        }
        return o;
    }

    public static final jq_StaticMethod _clsinitAndAllocateObject;
    public static final jq_StaticMethod _multinewarray;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/Allocator;");
        _clsinitAndAllocateObject = k.getOrCreateStaticMethod("clsinitAndAllocateObject", "(LClazz/jq_Type;)Ljava/lang/Object;");
        _multinewarray = k.getOrCreateStaticMethod("multinewarray", "(CLClazz/jq_Type;)Ljava/lang/Object;");
    }
    
}

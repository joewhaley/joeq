package Run_Time;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Util.Assert;

public class Arrays implements jq_ClassFileConstants {
    /**
     * Allocate a multidimensional array with dim dimensions and array type f.
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
        a.cls_initialize();
        if (a.getDimensionality() < dim)
            throw new VerifyError();
        int[] n_elem = new int[dim];
        int offset = StackAddress.size() + CodeAddress.size() + HeapAddress.size() + HeapAddress.size();
        StackAddress p = (StackAddress) StackAddress.getBasePointer().offset(offset);
        for (int i=dim-1; i>=0; --i) {
            n_elem[i] = p.peek4();
            // check for dim < 0 here, because if a dim is zero, later dim's
            // are not checked by multinewarray_helper.
            if (n_elem[i] < 0)
                throw new NegativeArraySizeException("dim "+i+": "+n_elem[i]+" < 0");
            p = (StackAddress) p.offset(HeapAddress.size());
        }
        return multinewarray_helper(n_elem, 0, a);
    }
    
    /**
     * Allocates a multidimensional array of type a, with dimensions given in
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
        Assert._assert(length >= 0);
        if (ind == dims.length-1)
            return o;
        Object[] o2 = (Object[])o;
        jq_Array a2 = (jq_Array)a.getElementType();
        a2.cls_initialize();
        for (int i=0; i<length; ++i) {
            o2[i] = multinewarray_helper(dims, ind+1, a2);
        }
        return o2;
    }

    public static final jq_StaticMethod _multinewarray;

    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Arrays;");
        _multinewarray = k.getOrCreateStaticMethod("multinewarray", "(CLClazz/jq_Type;)Ljava/lang/Object;");
    }
}

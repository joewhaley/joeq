/*
 * Method.java
 *
 * Created on April 14, 2001, 3:11 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.lang.reflect;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.*;
import jq;

public abstract class Method {

    // additional instance field.
    public final jq_Method jq_method = null;
    
    public static java.lang.Object invoke(java.lang.reflect.Method dis,
                                          java.lang.Object obj,
                                          java.lang.Object[] initargs)
	throws java.lang.InstantiationException, java.lang.IllegalAccessException,
               java.lang.IllegalArgumentException, java.lang.reflect.InvocationTargetException
    {
        jq_Method jq_m = (jq_Initializer)Reflection.getfield_A(dis, _jq_method);
        jq_Class k = jq_m.getDeclaringClass();
        jq_Reference t = Unsafe.getTypeOf(obj);
        if (!TypeCheck.isAssignable(t, k))
            throw new java.lang.IllegalArgumentException(t+" is not assignable to "+k);
        if (!dis.isAccessible()) jq_m.checkCallerAccess(3);
        int offset;
        if (jq_m.isStatic()) {
            obj = null; offset = 0;
        } else {
            offset = 1;
        }
        jq_Type[] argtypes = jq_m.getParamTypes();
        if (initargs.length != argtypes.length-offset) throw new java.lang.IllegalArgumentException();
        if (jq_m.isStatic()) {
            k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        } else {
            jq_m = t.getVirtualMethod(jq_m.getNameAndDesc());
            if (jq_m == null || jq_m.isAbstract())
                throw new java.lang.AbstractMethodError();
        }
        long result = Reflection.invoke(jq_m, obj, initargs);
        jq_Type retType = jq_m.getReturnType();
        if (retType.isReferenceType()) return Unsafe.asObject((int)result);
        if (retType == jq_Primitive.VOID) return null;
        if (retType == jq_Primitive.INT) return new Integer((int)result);
        if (retType == jq_Primitive.LONG) return new Long(result);
        if (retType == jq_Primitive.FLOAT) return new Float(Float.intBitsToFloat((int)result));
        if (retType == jq_Primitive.DOUBLE) return new Double(Double.longBitsToDouble(result));
        if (retType == jq_Primitive.BOOLEAN) return new Boolean((int)result!=0);
        if (retType == jq_Primitive.BYTE) return new Byte((byte)result);
        if (retType == jq_Primitive.SHORT) return new Short((short)result);
        if (retType == jq_Primitive.CHAR) return new Character((char)result);
        jq.UNREACHABLE(); return null;
    }
    // additional methods.
    // ONLY TO BE CALLED BY jq_Member CONSTRUCTOR!!!
    public static java.lang.reflect.Method createNewMethod(jq_Class clazz, jq_Method jq_method) {
        java.lang.reflect.Method o = (java.lang.reflect.Method)_class.newInstance();
        Reflection.putfield_A(o, _jq_method, jq_method);
        return o;
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _jq_method;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Method;");
        _jq_method = _class.getOrCreateInstanceField("jq_method", "LClazz/jq_Method;");
    }
    
}

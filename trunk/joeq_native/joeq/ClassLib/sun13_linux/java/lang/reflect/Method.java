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
        jq_Method jq_m = (jq_Method)Reflection.getfield_A(dis, _jq_method);
        jq_Class k = jq_m.getDeclaringClass();
	if (!jq_m.isStatic()) {
	    jq_Reference t = Unsafe.getTypeOf(obj);
	    if (!TypeCheck.isAssignable(t, k))
		throw new java.lang.IllegalArgumentException(t+" is not assignable to "+k);
	}
        if (!dis.isAccessible()) jq_m.checkCallerAccess(3);
        int offset;
        if (jq_m.isStatic()) {
            obj = null; offset = 0;
        } else {
            offset = 1;
        }
        jq_Type[] argtypes = jq_m.getParamTypes();
	int nargs = initargs==null ? 0 : initargs.length;
        if (nargs != argtypes.length-offset)
	    throw new java.lang.IllegalArgumentException();
        if (jq_m.isStatic()) {
            k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        } else {
	    jq_Reference t = Unsafe.getTypeOf(obj);
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
    public static java.lang.reflect.Method createNewMethod(jq_Class claz, jq_Method jq_method) {
        java.lang.reflect.Method o = (java.lang.reflect.Method)_class.newInstance();
        Reflection.putfield_A(o, _jq_method, jq_method);
        return o;
    }
    public static void initNewMethod(java.lang.reflect.Method o, jq_Method jq_method) {
	if (jq.Bootstrapping) return;
	java.lang.String name = jq_method.getName().toString();
        Reflection.putfield_A(o, _name, name);
	java.lang.Class clazz = jq_method.getDeclaringClass().getJavaLangClassObject();
	jq.assert(clazz != null);
	Reflection.putfield_A(o, _clazz, clazz);
	java.lang.Class returnType = jq_method.getReturnType().getJavaLangClassObject();
	jq.assert(returnType != null);
	Reflection.putfield_A(o, _returnType, returnType);
	jq_Type[] paramTypes = jq_method.getParamTypes();
	int offset;
	if (jq_method instanceof jq_InstanceMethod)
	    offset = 1;
	else
	    offset = 0;
	java.lang.Class[] parameterTypes = new java.lang.Class[paramTypes.length-offset];
	for (int i=offset; i<paramTypes.length; ++i) {
	    parameterTypes[i-offset] = Reflection.getJDKType(paramTypes[i]);
	    jq.assert(parameterTypes[i-offset] != null);
	}
	Reflection.putfield_A(o, _parameterTypes, parameterTypes);
	// TODO: exception types
	java.lang.Class[] exceptionTypes = new java.lang.Class[0];
	Reflection.putfield_A(o, _exceptionTypes, exceptionTypes);
	int modifiers = jq_method.getAccessFlags();
	Reflection.putfield_I(o, _modifiers, modifiers);
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _jq_method;
    public static final jq_InstanceField _clazz;
    public static final jq_InstanceField _name;
    public static final jq_InstanceField _returnType;
    public static final jq_InstanceField _parameterTypes;
    public static final jq_InstanceField _exceptionTypes;
    public static final jq_InstanceField _modifiers;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Method;");
        _jq_method = _class.getOrCreateInstanceField("jq_method", "LClazz/jq_Method;");
        _clazz = _class.getOrCreateInstanceField("clazz", "Ljava/lang/Class;");
        _name = _class.getOrCreateInstanceField("name", "Ljava/lang/String;");
        _returnType = _class.getOrCreateInstanceField("returnType", "Ljava/lang/Class;");
        _parameterTypes = _class.getOrCreateInstanceField("parameterTypes", "[Ljava/lang/Class;");
        _exceptionTypes = _class.getOrCreateInstanceField("exceptionTypes", "[Ljava/lang/Class;");
	_modifiers = _class.getOrCreateInstanceField("modifiers", "I");
    }
    
}

/*
 * Constructor.java
 *
 * Created on April 14, 2001, 3:16 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.lang.reflect;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.*;
import jq;

public abstract class Constructor {

    // additional instance field.
    public final jq_Initializer jq_init = null;
    
    public static java.lang.Object newInstance(java.lang.reflect.Constructor dis,
                                               java.lang.Object[] initargs)
	throws java.lang.InstantiationException, java.lang.IllegalAccessException,
               java.lang.IllegalArgumentException, java.lang.reflect.InvocationTargetException
    {
        jq_Initializer jq_i = (jq_Initializer)Reflection.getfield_A(dis, _jq_init);
        jq_Class k = jq_i.getDeclaringClass();
        if (k.isAbstract()) throw new InstantiationException();
        if (!dis.isAccessible()) jq_i.checkCallerAccess(3);
        jq_Type[] argtypes = jq_i.getParamTypes();
	int nargs = initargs == null ? 0 : initargs.length;
	if (nargs != argtypes.length-1)
	    throw new java.lang.IllegalArgumentException("Constructor takes "+(argtypes.length-1)+" arguments, but "+nargs+" arguments passed in");
        Object o = k.newInstance();
        Reflection.invoke(jq_i, o, initargs);
        return o;
    }
    
    // additional methods.
    // ONLY TO BE CALLED BY jq_Member CONSTRUCTOR!!!
    public static java.lang.reflect.Constructor createNewConstructor(jq_Class claz, jq_Initializer jq_init) {
        java.lang.reflect.Constructor o = (java.lang.reflect.Constructor)_class.newInstance();
        Reflection.putfield_A(o, _jq_init, jq_init);
        return o;
    }
    public static void initNewConstructor(java.lang.reflect.Constructor o, jq_Initializer jq_init) {
	if (jq.Bootstrapping) return;
	java.lang.Class clazz = jq_init.getDeclaringClass().getJavaLangClassObject();
	Reflection.putfield_A(o, _clazz, clazz);
	jq_Type[] paramTypes = jq_init.getParamTypes();
	java.lang.Class[] parameterTypes = new java.lang.Class[paramTypes.length-1];
	for (int i=1; i<paramTypes.length; ++i) {
	    parameterTypes[i-1] = Reflection.getJDKType(paramTypes[i]);
	}
	Reflection.putfield_A(o, _parameterTypes, parameterTypes);
	// TODO: exception types
	java.lang.Class[] exceptionTypes = new java.lang.Class[0];
	Reflection.putfield_A(o, _exceptionTypes, exceptionTypes);
	int modifiers = jq_init.getAccessFlags();
	Reflection.putfield_I(o, _modifiers, modifiers);
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _jq_init;
    public static final jq_InstanceField _clazz;
    public static final jq_InstanceField _parameterTypes;
    public static final jq_InstanceField _exceptionTypes;
    public static final jq_InstanceField _modifiers;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Constructor;");
        _jq_init = _class.getOrCreateInstanceField("jq_init", "LClazz/jq_Initializer;");
        _clazz = _class.getOrCreateInstanceField("clazz", "Ljava/lang/Class;");
        _parameterTypes = _class.getOrCreateInstanceField("parameterTypes", "[Ljava/lang/Class;");
        _exceptionTypes = _class.getOrCreateInstanceField("exceptionTypes", "[Ljava/lang/Class;");
	_modifiers = _class.getOrCreateInstanceField("modifiers", "I");
    }
    
}

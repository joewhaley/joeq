/*
 * Constructor.java
 *
 * Created on April 14, 2001, 3:16 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_win32.java.lang.reflect;

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
        if (initargs.length != argtypes.length-1) throw new java.lang.IllegalArgumentException();
        Object o = k.newInstance();
        Reflection.invoke(jq_i, o, initargs);
        return o;
    }
    
    // additional methods.
    // ONLY TO BE CALLED BY jq_Member CONSTRUCTOR!!!
    public static java.lang.reflect.Constructor createNewConstructor(jq_Class clazz, jq_Initializer jq_init) {
        java.lang.reflect.Constructor o = (java.lang.reflect.Constructor)_class.newInstance();
        Reflection.putfield_A(o, _jq_init, jq_init);
        return o;
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _jq_init;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Constructor;");
        _jq_init = _class.getOrCreateInstanceField("jq_init", "LClazz/jq_Initializer;");
    }
    
}

/*
 * Constructor.java
 *
 * Created on April 14, 2001, 3:16 PM
 *
 */

package ClassLib.Common.java.lang.reflect;

import Clazz.jq_Class;
import Clazz.jq_Initializer;
import Clazz.jq_Type;
import Main.jq;
import Run_Time.Reflection;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class Constructor extends AccessibleObject {

    // additional instance field.
    public final jq_Initializer jq_init;
    
    private java.lang.Class clazz;
    private java.lang.Class[] parameterTypes;
    private java.lang.Class[] exceptionTypes;
    private int modifiers;
    
    private Constructor(jq_Initializer i) {
        this.jq_init = i;
    }
    
    public java.lang.Object newInstance(java.lang.Object[] initargs)
        throws java.lang.InstantiationException, java.lang.IllegalAccessException,
               java.lang.IllegalArgumentException, java.lang.reflect.InvocationTargetException
    {
        jq_Initializer jq_i = this.jq_init;
        jq_Class k = jq_i.getDeclaringClass();
        if (k.isAbstract()) throw new InstantiationException();
        if (!this.isAccessible()) jq_i.checkCallerAccess(2);
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
    public static java.lang.reflect.Constructor createNewConstructor(jq_Initializer jq_init) {
        Object o = new Constructor(jq_init);
        return (java.lang.reflect.Constructor)o;
    }
    
    public static void initNewConstructor(Constructor o, jq_Initializer jq_init) {
        if (jq.Bootstrapping) return;
        java.lang.Class clazz = jq_init.getDeclaringClass().getJavaLangClassObject();
        o.clazz = clazz;
        jq_Type[] paramTypes = jq_init.getParamTypes();
        java.lang.Class[] parameterTypes = new java.lang.Class[paramTypes.length-1];
        for (int i=1; i<paramTypes.length; ++i) {
            parameterTypes[i-1] = Reflection.getJDKType(paramTypes[i]);
        }
        o.parameterTypes = parameterTypes;
        // TODO: exception types
        java.lang.Class[] exceptionTypes = new java.lang.Class[0];
        o.exceptionTypes = exceptionTypes;
        int modifiers = jq_init.getAccessFlags();
        o.modifiers = modifiers;
    }
}

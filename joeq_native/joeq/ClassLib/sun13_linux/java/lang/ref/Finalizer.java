/*
 * Finalizer.java
 *
 * Created on February 26, 2001, 9:01 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.lang.ref;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticMethod;
import Clazz.jq_NameAndDesc;
import Run_Time.Reflection;
import Run_Time.Unsafe;
import UTF.Utf8;
import jq;

public abstract class Finalizer {

    static void invokeFinalizeMethod(jq_Class clazz, Object o) throws Throwable {
        jq_Reference c = Unsafe.getTypeOf(o);
        jq_InstanceMethod m = c.getVirtualMethod(new jq_NameAndDesc(Utf8.get("finalize"), Utf8.get("()V")));
        jq.assert(m != null);
        Reflection.invokeinstance_V(m, o);
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/ref/Finalizer;");
    public static final jq_StaticMethod _runAllFinalizers = _class.getOrCreateStaticMethod("runAllFinalizers", "()V");
    public static final jq_StaticMethod _runFinalization = _class.getOrCreateStaticMethod("runFinalization", "()V");
}

/*
 * Throwable.java
 *
 * Created on January 29, 2000, 10:16 AM
 *
 * @author  jwhaley
 * @version 
 */

package ClassLib.sun13_win32.java.lang;

import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;
import Run_Time.Reflection;
import jq;

public abstract class Throwable {
    
    // native method implementations
    private static void printStackTrace0(java.lang.Throwable dis, java.lang.Object s) {
        java.lang.Object backtrace = Reflection.getfield_A(dis, _backtrace);
        if (s instanceof java.io.PrintWriter)
            ExceptionDeliverer.printStackTrace(backtrace, (java.io.PrintWriter)s);
        else if (s instanceof java.io.PrintStream)
            ExceptionDeliverer.printStackTrace(backtrace, (java.io.PrintStream)s);
        else
            jq.UNREACHABLE();
    }
    
    public static java.lang.Throwable fillInStackTrace(java.lang.Throwable dis) {
        Reflection.putfield_A(dis, _backtrace, ExceptionDeliverer.getStackTrace());
        return dis;
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Throwable;");
    public static final jq_InstanceField _backtrace = (jq_InstanceField)_class.getOrCreateInstanceField("backtrace", "Ljava/lang/Object;");
}

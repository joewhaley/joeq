/*
 * Shutdown.java
 *
 * Created on February 26, 2001, 8:56 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.lang;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import jq;

abstract class Shutdown {
    
    static void halt(jq_Class clazz, int status) {
        SystemInterface.die(status);
        jq.UNREACHABLE();
    }
    private static void runAllFinalizers(jq_Class clazz) {
        try {
            Reflection.invokestatic_V(ClassLib.sun13_linux.java.lang.ref.Finalizer._runAllFinalizers);
        } catch (java.lang.Throwable x) {
        }
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Shutdown;");
}

/*
 * Runtime.java
 *
 * Created on April 16, 2001, 1:25 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun14_win32.java.lang;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.*;
import jq;

public abstract class Runtime {

    // native method implementations.
    private static Process execInternal(java.lang.Runtime dis,
                                        java.lang.String cmdarray[],
                                        java.lang.String envp[],
                                        java.lang.String path) 
        throws java.io.IOException {
        jq.TODO();
        return null;
    }

    public static long freeMemory(java.lang.Runtime dis) {
        // TODO
        return 0L;
    }
    public static long totalMemory(java.lang.Runtime dis) {
        // TODO
        return 0L;
    }
    public static void gc(java.lang.Runtime dis) {
        // TODO
    }
    private static void runFinalization0(jq_Class clazz) {
        try {
            jq_Class k = ClassLib.sun14_win32.java.lang.ref.Finalizer._class;
            k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
            Reflection.invokestatic_V(ClassLib.sun14_win32.java.lang.ref.Finalizer._runFinalization);
            /*
        } catch (java.lang.Error t) {
            throw t;
        } catch (java.lang.RuntimeException t) {
            throw t;
             */
        } catch (java.lang.Throwable t) {}
    }
    public static void traceInstructions(java.lang.Runtime dis, boolean on) {
        // TODO
    }
    public static void traceMethodCalls(java.lang.Runtime dis, boolean on) {
        // TODO
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runtime;");
}

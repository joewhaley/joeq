/*
 * Runtime.java
 *
 * Created on April 16, 2001, 1:25 AM
 *
 */

package ClassLib.Common.java.lang;

import Main.jq;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Runtime {

    // native method implementations.
    private Process execInternal(java.lang.String cmdarray[],
                                 java.lang.String envp[],
                                 java.lang.String path) 
        throws java.io.IOException {
        jq.TODO();
        return null;
    }

    public long freeMemory() {
        // TODO
        return 0L;
    }
    public long totalMemory() {
        // TODO
        return 0L;
    }
    public void gc() {
        // TODO
    }
    private static void runFinalization0() {
        try {
            ClassLib.Common.java.lang.ref.Finalizer.runFinalization();
        } catch (java.lang.Throwable t) {
        }
    }
    public void traceInstructions(boolean on) {
        // TODO
    }
    public void traceMethodCalls(boolean on) {
        // TODO
    }
    
}

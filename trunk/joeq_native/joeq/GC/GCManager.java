/**
 * GCManager
 *
 * Created on Sep 25, 2002, 8:28:16 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package GC;

import java.util.Hashtable;

public class GCManager {
    // Tracing GC
    public static final int TR_GC = 0x10000000;
    // Reference-counting GC
    public static final int RC_GC = 0x20000000;
    // Mark & Sweep Tracing GC
    public static final int MS_TR_GC = 0x10000001;
    // Copying Tracing GC
    public static final int CP_TR_GC = 0x10000002;
    // Incremental Tracing GC
    public static final int IC_TR_GC = 0x10000004;
    // Conservative Tracing GC
    public static final int CS_TR_GC = 0x10000008;
    //Generational Tracing GC
    public static final int GN_TR_GC = 0x10000010;
    // Deferred Reference Counting GC
    public static final int DF_RC_GC = 0x20000001;
    // One-bit Reference Counting GC
    public static final int OB_RC_GC = 0x20000002;
    // Weighted Refernece Counting GC
    public static final int WT_RC_GC = 0x20000004;

    private static boolean initialized = false;
    private static Hashtable candidates = new Hashtable();
    private static Runnable defaultTracingGC = new TraceMSGC();
    private static Runnable defaultRCGC = new SimpleRCGC();

    public static void initialize() {
        candidates.put(new Integer(MS_TR_GC), new TraceMSGC());
        candidates.put(new Integer(RC_GC), new SimpleRCGC());
        initialized = true;
    }

    public static void reset() {
        initialized = false;
    }

    public static Object getGC(int key) {
        if (!initialized) {
            initialize();
        }
        Object result = candidates.get(new Integer(key));
        if (result != null) {
            return result;
        } else if ((key & TR_GC) != 0) {
            return defaultTracingGC;
        } else if ((key & RC_GC) != 0) {
            return defaultRCGC;
        } else {
            return null;
        }
    }
}

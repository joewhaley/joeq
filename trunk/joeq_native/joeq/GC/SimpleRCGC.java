/**
 * Simple Reference Counting GC
 *
 * Created on Sep 25, 2002, 9:21:02 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package GC;

import Scheduler.jq_NativeThread;
import Scheduler.jq_RegisterState;

public class SimpleRCGC implements Runnable, GCVisitor {

    public static /*final*/ boolean TRACE = false;

    public void run() {
    }

    public void visit(jq_RegisterState state) {
    }

    public void farewell(jq_NativeThread[] nt) {
    }
}

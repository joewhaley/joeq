/**
 * TraceMSGC
 *
 * Created on Sep 23, 2002, 11:04:51 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version $Id$
 */
package GC;

import Scheduler.jq_RegisterState;

public class TraceMSGC implements Runnable, GCVisitor {

    public static /*final*/ boolean TRACE = false;

    private TraceRootSet roots = new TraceRootSet();

    public void run() {
    }

    public void mark() {
    }

    public void sweep() {
    }

    public void compact() {
    }

    public void visit(jq_RegisterState state) {
    }
}

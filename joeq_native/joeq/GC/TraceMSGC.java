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

import java.util.ArrayList;

import Run_Time.StackHeapWalker;
import Memory.HeapAddress;

public class TraceMSGC implements Runnable, GCVisitor {

    public static /*final*/ boolean TRACE = false;

    private TraceRootSet roots = new TraceRootSet();

    public void run() {
    }

    public void mark() {
        GCBitsManager.diff();
    }

    public void sweep() {
    }

    public void compact() {
    }

    public void visit(jq_RegisterState state) {
        ArrayList validAddrs = new StackHeapWalker(state.getEsp(), state.getEbp()).getValidHeapAddrs();
        for (int i = 0, size = validAddrs.size(); i < size; ++i) {
            GCBitsManager.mark((HeapAddress)validAddrs.get(i));
        }
    }
}

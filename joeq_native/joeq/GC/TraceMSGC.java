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
import Scheduler.jq_NativeThread;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import Run_Time.StackHeapWalker;
import Memory.HeapAddress;

public class TraceMSGC implements Runnable, GCVisitor {

    public static /*final*/ boolean TRACE = false;

    private TraceRootSet roots = new TraceRootSet();
    private HashSet dumpool = new HashSet();

    public void run() {
    }

    public void visit(jq_RegisterState state) {
        dumpool.add(state);
        ArrayList validAddrs = new StackHeapWalker(state.getEsp(), state.getEbp()).getValidHeapAddrs();
        for (int i = 0, size = validAddrs.size(); i < size; ++i) {
            GCBitsManager.mark((HeapAddress) validAddrs.get(i));
        }
    }

    public void mark() {
        GCBitsManager.diff();
    }

    public void sweep() {
    }

    public void compact() {
    }

    public void farewell(jq_NativeThread[] nt) {
        Iterator iter = dumpool.iterator();
        for (int i = 0; i < nt.length; ++i) {
            nt[i].setContext((jq_RegisterState)iter.next());
        }
    }
}

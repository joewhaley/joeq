// TraceMSjoeq.GC.java, created Wed Sep 25  7:09:24 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.

package joeq.GC;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import joeq.Memory.HeapAddress;
import joeq.Run_Time.StackHeapWalker;
import joeq.Scheduler.jq_NativeThread;
import joeq.Scheduler.jq_RegisterState;

/**
 * Trace Mark & Sweep GC
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
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

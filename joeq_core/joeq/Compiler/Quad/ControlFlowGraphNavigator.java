// ControlFlowGraphNavigator.java, created Sat Mar 29  0:56:01 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Main.HostedVM;
import Util.Collections.AppendList;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ControlFlowGraphNavigator implements Navigator {

    protected ControlFlowGraph cfg;

    /** Construct a new ControlFlowGraphNavigator for the given
     * control flow graph.
     * @param cfg control flow graph
     */
    public ControlFlowGraphNavigator(ControlFlowGraph cfg) {
        this.cfg = cfg;
    }
    private ControlFlowGraphNavigator() {}
    
    /** Singleton object for a control flow graph navigator that
     * does not take into account exception edges.
     */
    public static final ControlFlowGraphNavigator INSTANCE = new ControlFlowGraphNavigator();

    /* (non-Javadoc)
     * @see Util.Graphs.Navigator#next(java.lang.Object)
     */
    public Collection next(Object node) {
        BasicBlock bb = (BasicBlock) node;
        List result = bb.getSuccessors();
        List eh = bb.getExceptionHandlerEntries();
        if (cfg == null || eh.isEmpty()) return result;
        else return new AppendList(result, eh);
    }

    /* (non-Javadoc)
     * @see Util.Graphs.Navigator#prev(java.lang.Object)
     */
    public Collection prev(Object node) {
        BasicBlock bb = (BasicBlock) node;
        List result = bb.getPredecessors();
        if (cfg == null || !bb.isExceptionHandlerEntry()) return result;
        Iterator ex_handlers = cfg.getExceptionHandlersMatchingEntry(bb);
        while (ex_handlers.hasNext()) {
            ExceptionHandler eh = (ExceptionHandler)ex_handlers.next();
            result = new AppendList(result, eh.getHandledBasicBlocks());
        }
        return result;
    }

    // test function
    public static void main(String[] args) {
        HostedVM.initialize();
        HashSet set = new HashSet();
        for (int i=0; i<args.length; ++i) {
            String s = args[i];
            jq_Class c = (jq_Class) jq_Type.parseType(s);
            c.load();
            set.addAll(Arrays.asList(c.getDeclaredStaticMethods()));
        }
        for (Iterator i=set.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            System.out.println("Method "+m);
            ControlFlowGraph cfg = CodeCache.getCode(m);
            ControlFlowGraphNavigator v = new ControlFlowGraphNavigator(cfg);
            SCComponent c = SCComponent.buildSCC(cfg.entry(), v);
            SCCTopSortedGraph g = SCCTopSortedGraph.topSort(c);
            for (Iterator j=g.getFirst().listTopSort().iterator(); j.hasNext(); ) {
                SCComponent d = (SCComponent) j.next();
                System.out.println(d);
            }
        }
    }

}

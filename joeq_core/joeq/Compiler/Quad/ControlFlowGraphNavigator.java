/*
 * Created on Mar 27, 2003
 */
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Main.HostedVM;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley
 * @version $Id$
 */
public class ControlFlowGraphNavigator implements Navigator {

    /* (non-Javadoc)
     * @see Util.Graphs.Navigator#next(java.lang.Object)
     */
    public Collection next(Object node) {
        BasicBlock bb = (BasicBlock) node;
        return bb.getSuccessors();
    }

    /* (non-Javadoc)
     * @see Util.Graphs.Navigator#prev(java.lang.Object)
     */
    public Collection prev(Object node) {
        BasicBlock bb = (BasicBlock) node;
        return bb.getPredecessors();
    }

    private ControlFlowGraphNavigator() {}

    public static final ControlFlowGraphNavigator INSTANCE = new ControlFlowGraphNavigator();

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
            SCComponent c = SCComponent.buildSCC(cfg.entry(), INSTANCE);
            SCCTopSortedGraph g = SCCTopSortedGraph.topSort(c);
            for (Iterator j=g.getFirst().listTopSort().iterator(); j.hasNext(); ) {
                SCComponent d = (SCComponent) j.next();
                System.out.println(d);
            }
        }
    }

}

/*
 * Created on Mar 28, 2003
 */
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_Type;
import Main.HostedVM;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley
 * @version $Id$
 */
public class RootedCHACallGraph extends CHACallGraph {
    
    Collection roots;
    
    public RootedCHACallGraph() { }
    public RootedCHACallGraph(Set classes) {
        super(classes);
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getRoots()
     */
    protected Collection getRoots() {
        return roots;
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#setRoots(java.util.Collection)
     */
    public void setRoots(Collection roots) {
        this.roots = roots;
    }

    public static void main(String[] args) {
        long time;
        
        HostedVM.initialize();
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        
        System.out.print("Building call graph...");
        time = System.currentTimeMillis();
        CallGraph cg = new RootedCHACallGraph();
        cg = new CachedCallGraph(cg);
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        cg.setRoots(roots);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+(time/1000.)+" seconds)");
        
        test(cg);
    }
    
    public static void test(CallGraph cg) {
        long time;
        if (true) {
            System.out.print("Building navigator...");
            time = System.currentTimeMillis();
            Navigator n = cg.getNavigator();
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+(time/1000.)+" seconds)");
            
            System.out.print("Building strongly-connected components...");
            time = System.currentTimeMillis();
            Set s = SCComponent.buildSCC(cg.getRoots(), n);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+(time/1000.)+" seconds)");
            
            System.out.print("Topologically sorting strongly-connected components...");
            time = System.currentTimeMillis();
            SCCTopSortedGraph g = SCCTopSortedGraph.topSort(s);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+(time/1000.)+" seconds)");
            
            for (Iterator j=g.getFirst().listTopSort().iterator(); j.hasNext(); ) {
                SCComponent d = (SCComponent) j.next();
                System.out.println(d);
            }
        }
        
        if (false) {
            Collection[] depths = INSTANCE.findDepths();
            for (int i=0; i<depths.length; ++i) {
                System.out.println(">>>>> Depth "+i);
                System.out.println(depths[i]);
            }
        }
    }

}

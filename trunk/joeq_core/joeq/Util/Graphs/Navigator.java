package Util.Graphs;

import java.util.Collection;

/** The <code>Navigator</code> interface allows graph algorithms to
    detect (and use) the arcs from and to a certain node. This allows
    the use of many graph algorithms (eg construction of strongly
    connected components) even for very general graphs where the arcs
    model only a subtle semantic relation (eg caller-callee) that is
    not directly stored in the structure of the nodes.

   @author  Alexandru SALCIANU <salcianu@MIT.EDU>
   @version $Id$ */
public interface Navigator {
    
    /** Returns the predecessors of <code>node</code>. */
    public Collection next(Object node);

    /** Returns the successors of <code>node</code>. */
    public Collection prev(Object node);
    
}

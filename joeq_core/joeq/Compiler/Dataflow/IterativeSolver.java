// IterativeSolver.java, created Thu Apr 25 16:32:26 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Dataflow;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import joeq.Util.Collections.MapFactory;
import joeq.Util.Graphs.Graph;
import joeq.Util.Graphs.Navigator;
import joeq.Util.Graphs.Traversals;

/**
 * Solves a dataflow problem using a iterative technique.  Successively
 * iterates over the locations in the graph in a given order until there
 * are no more changes.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class IterativeSolver
    extends Solver
{
    /** The order in which the locations are to be traversed. */
    protected List traversalOrder;
    /** The boundary locations. */
    protected Collection boundaries;
    /** Navigator to navigate the graph of locations. */
    protected Navigator graphNavigator;
    /** Change flag, set to true if we need to iterate more times. */
    protected boolean change;

    public IterativeSolver(MapFactory f) {
        super(f);
    }
    public IterativeSolver() {
        super();
    }

    /** Returns an iteration of the order in which the locations are to be traversed. */
    protected Iterator getTraversalOrder() { return traversalOrder.iterator(); }
    
    /** Get the predecessor locations of the given location. */
    protected Collection getPredecessors(Object c) { return graphNavigator.prev(c); }
    /** Get the successor locations of the given location. */
    protected Collection getSuccessors(Object c) { return graphNavigator.next(c); }
    
    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Solver#initialize(Compil3r.Dataflow.Problem, Util.Graphs.Graph)
     */
    public void initialize(Problem p, Graph graph) {
        this.initialize(p, graph, Traversals.reversePostOrder(graph.getNavigator(), graph.getRoots()));
    }
    
    /** Initializes this solver with the given dataflow problem, graph, and
     * traversal order.
     * 
     * @see Compil3r.Dataflow.Solver#initialize(Compil3r.Dataflow.Problem, Util.Graphs.Graph)
     */
    public void initialize(Problem p, Graph graph, List order) {
        super.initialize(p, graph);
        graphNavigator = graph.getNavigator();
        boundaries = graph.getRoots();
        traversalOrder = order;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Solver#allLocations()
     */
    public Iterator allLocations() { return traversalOrder.iterator(); }

    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Solver#boundaryLocations()
     */
    public Iterator boundaryLocations() { return boundaries.iterator(); }

    public static final boolean TRACE = false;

    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Solver#solve()
     */
    public void solve() {
        initializeDataflowValueMap();
        int iterationCount = 0;
        do {
            change = false; if (TRACE) ++iterationCount;
            Iterator i = getTraversalOrder();
            i.next(); // skip boundary node.
            while (i.hasNext()) {
                Object c = i.next();
                Iterator j = direction()?
                             getPredecessors(c).iterator():
                             getSuccessors(c).iterator();
                Object p = j.next();
                Fact in = (Fact) dataflowValues.get(p);
                while (j.hasNext()) {
                    p = j.next();
                    Fact in2 = (Fact) dataflowValues.get(p);
                    in = problem.merge(in, in2);
                }
                TransferFunction tf = problem.getTransferFunction(c);
                Fact out = problem.apply(tf, in);
                Fact old = (Fact) dataflowValues.put(c, out);
                if (!change && !problem.compare(old, out)) {
                    change = true;
                }
            }
        } while (change);
        if (TRACE) System.out.println("Number of iterations: "+iterationCount);
    }

}

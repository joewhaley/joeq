// WorklistSolver.java, created Thu Apr 25 16:32:26 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Dataflow;

import java.util.Collection;
import java.util.Iterator;

import joeq.Util.Collections.MapFactory;
import joeq.Util.Graphs.Graph;
import joeq.Util.Graphs.Navigator;

/**
 * WorklistSolver
 * 
 * @author John Whaley
 * @version $Id$
 */
public abstract class WorklistSolver extends Solver {

    /** Navigator to navigate the graph of locations. */
    protected Navigator graphNavigator;
    /** The boundary locations. */
    protected Collection boundaries;

    protected WorklistSolver(MapFactory f) {
        super(f);
    }
    protected WorklistSolver() {
        super();
    }

    /** Get the predecessor locations of the given location. */
    protected Collection getPredecessors(Object c) { return graphNavigator.prev(c); }
    /** Get the successor locations of the given location. */
    protected Collection getSuccessors(Object c) { return graphNavigator.next(c); }
    
    /** (Re-)initialize the worklist. */
    protected abstract void initializeWorklist();
    /** Returns true if the worklist is not empty, false otherwise. */
    protected abstract boolean hasNext();
    /** Pull the next location off of the worklist. */
    protected abstract Object pull();
    /** Push all of the given locations onto the worklist. */
    protected abstract void pushAll(Collection c);

    /* (non-Javadoc)
     * @see joeq.Compiler.Dataflow.Solver#boundaryLocations()
     */
    public Iterator boundaryLocations() {
        return boundaries.iterator();
    }
    
    /* (non-Javadoc)
     * @see joeq.Compiler.Dataflow.Solver#initialize(joeq.Compiler.Dataflow.Problem, Util.Graphs.Graph)
     */
    public void initialize(Problem p, Graph graph) {
        super.initialize(p, graph);
        graphNavigator = graph.getNavigator();
        boundaries = graph.getRoots();
    }

    /* (non-Javadoc)
     * @see joeq.Compiler.Dataflow.Solver#solve()
     */
    public void solve() {
        initializeDataflowValueMap();
        initializeWorklist();
        while (hasNext()) {
            Object c = pull();
            if (TRACE) System.out.println("Node "+c);
            Iterator j = getPredecessors(c).iterator();
            Object p = j.next();
            if (TRACE) System.out.println("   Predecessor "+p);
            Fact in = (Fact) dataflowValues.get(p);
            while (j.hasNext()) {
                p = j.next();
                if (TRACE) System.out.println("   Predecessor "+p);
                Fact in2 = (Fact) dataflowValues.get(p);
                in = problem.merge(in, in2);
            }
            TransferFunction tf = problem.getTransferFunction(c);
            Fact out = problem.apply(tf, in);
            Fact old = (Fact) dataflowValues.put(c, out);
            if (!problem.compare(old, out)) {
                if (TRACE) System.out.println("Changed!");
                Collection next = getSuccessors(c);
                pushAll(next);
            }
        }
    }

}

//Stratify.java, created May 3, 2004 7:07:16 PM 2004 by jwhaley
//Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
//Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import java.io.PrintStream;

import joeq.Util.Assert;
import joeq.Util.Collections.GenericMultiMap;
import joeq.Util.Collections.HashWorklist;
import joeq.Util.Collections.MultiMap;
import joeq.Util.Graphs.SCCTopSortedGraph;
import joeq.Util.Graphs.SCComponent;

/**
 * Stratify
 * 
 * @author jwhaley
 * @version $Id$
 */
public class Stratify {
    
    boolean USE_NESTED_SCCS = true;
    boolean TRACE;
    PrintStream out;
    
    Solver solver;
    List/*<SCComponent>*/ firstSCCs;
    MultiMap innerSCCs;
    
    Stratify(Solver solver) {
        this.solver = solver;
        this.TRACE = solver.TRACE;
        this.out = solver.out;
    }
    
    public void stratify(List rules, Set inputs, Set outputs) {
        
        firstSCCs = new LinkedList();
        innerSCCs = new GenericMultiMap();
        
        // Build dependence graph.
        InferenceRule.DependenceNavigator depNav = new InferenceRule.DependenceNavigator(rules);
        
        // Do a backward pass to figure out what relations/rules are necessary.
        Set necessary = findNecessary(depNav, outputs);
        
        if (TRACE) out.println("Necessary: "+necessary);
        
        // Print out a warning message if something is unused.
        Set unnecessary = new HashSet(solver.nameToRelation.values());
        unnecessary.addAll(solver.rules);
        unnecessary.removeAll(necessary);
        if (!unnecessary.isEmpty()) {
            System.out.println("Note: the following rules/relations are unused:");
            for (Iterator i = unnecessary.iterator(); i.hasNext(); ) {
                System.out.println("    "+i.next());
            }
        }
        
        // Ignore all edges to/from unnecessary stuff.
        depNav.retainAll(necessary);
        
        // Calculate the set of necessary relations.
        Set allRelations = new HashSet();
        for (Iterator i = necessary.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof Relation) allRelations.add(o);
        }
        
        for (int i = 1; ; ++i) {
            // Discover current stratum.
            if (TRACE) out.println("Discovering Stratum #"+i+"...");
            Set stratumSccs = discoverStratum(depNav, allRelations, inputs);
            Set stratumNodes = getStratumNodes(stratumSccs);
            if (TRACE) out.println("Stratum #"+i+": "+stratumNodes);
            
            // Make a navigator for this stratum.
            InferenceRule.DependenceNavigator depNav2 = new InferenceRule.DependenceNavigator(depNav);
            depNav2.retainAll(stratumNodes);
            
            // Break current stratum into SCCs and sort them.
            // We can't use the SCCs in stratumSccs because they have links to later strata.
            SCComponent first = breakIntoSCCs(stratumNodes, depNav2);
            firstSCCs.add(first);
            
            // Any relations that have been totally computed are inputs to the
            // next stratum.
            boolean again = inputs.addAll(findNewInputs(depNav2, stratumNodes));
            if (!again) break;
            
            // Remove edges for this stratum from navigator.
            depNav.removeAll(stratumNodes);
        }
        
    }
    
    static Set getStratumNodes(Set stratumSccs) {
        Set s = new HashSet();
        for (Iterator i = stratumSccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            s.addAll(scc.nodeSet());
        }
        return s;
    }
    
    static Set findNewInputs(InferenceRule.DependenceNavigator depNav, Set stratumNodes) {
        Set inputs = new HashSet();
        for (Iterator i = stratumNodes.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof Relation) {
                Relation p = ((Relation) o).getNegated();
                if (p != null) {
                    inputs.add(p);
                }
            }
        }
        return inputs;
    }
    
    static Set findNecessary(InferenceRule.DependenceNavigator depNav, Collection outputs) {
        HashWorklist w = new HashWorklist(true);
        w.addAll(outputs);
        while (!w.isEmpty()) {
            Object o = w.pull();
            if (o instanceof Relation) {
                Relation p = ((Relation) o).getNegated();
                if (p != null) w.add(p);
            }
            Collection c = depNav.prev(o);
            w.addAll(c);
        }
        return w.getVisitedSet();
    }
    
    Set discoverStratum(InferenceRule.DependenceNavigator depNav, Collection allRelations, Collection inputs) {
        // Break into SCCs.
        Collection/*<SCComponent>*/ sccs = SCComponent.buildSCC(allRelations, depNav);
        
        LinkedList w = new LinkedList();
        Set stratum = new HashSet();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent o = (SCComponent) i.next();
            for (Iterator j = inputs.iterator(); j.hasNext(); ) {
                if (o.contains(j.next())) {
                    w.add(o);
                    stratum.add(o);
                    break;
                }
            }
        }
        while (!w.isEmpty()) {
            SCComponent o = (SCComponent) w.removeFirst();
            if (TRACE) out.println("Pulling from worklist: "+o);
            Collection c = Arrays.asList(o.next());
            for (Iterator i = c.iterator(); i.hasNext(); ) {
                SCComponent p = (SCComponent) i.next();
                if (TRACE) out.println("  Successor: "+p);
                if (TRACE) out.println("    Predecessors: "+Arrays.asList(p.prev()));
                if (stratum.containsAll(Arrays.asList(p.prev()))) {
                    if (TRACE) out.println("  Adding "+p+" to stratum");
                    if (stratum.add(p)) {
                        if (TRACE) out.println("    New element, adding to worklist.");
                        w.add(p);
                    }
                } else {
                    if (TRACE) out.println("  Not all predecessors of "+p+" (yet)");
                }
            }
        }
        // Remove those nodes that do not have all predecessors in stratum.
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent p = (SCComponent) i.next();
            if (!stratum.containsAll(Arrays.asList(p.prev()))) {
                if (TRACE) out.println("Not all predecessors of relation "+p+", removing.");
                stratum.remove(p);
            }
        }
        return stratum;
    }
    
    SCComponent breakIntoSCCs(Collection stratumNodes, InferenceRule.DependenceNavigator depNav) {
        
        Collection/*<SCComponent>*/ sccs = SCComponent.buildSCC(stratumNodes, depNav);
        
        // Find root SCCs.
        Set roots = new HashSet();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (scc.prevLength() == 0) {
                if (TRACE) out.println("Root SCC: SCC"+scc.getId()+(scc.isLoop()?" (loop)":""));
                roots.add(scc);
            }
        }
        if (roots.isEmpty()) {
            Assert.UNREACHABLE("No roots! "+sccs);
        }
        
        // Topologically-sort SCCs.
        SCCTopSortedGraph sccGraph = SCCTopSortedGraph.topSort(roots);
        SCComponent first = sccGraph.getFirst();
        
        // Find inner SCCs.
        if (USE_NESTED_SCCS) {
            for (SCComponent scc = first; scc != null; scc = scc.nextTopSort()) {
                if (!scc.isLoop()) continue;
                scc.fillEntriesAndExits(depNav);
                InferenceRule.DependenceNavigator depNav2 = new InferenceRule.DependenceNavigator(depNav);
                Set nodeSet = scc.nodeSet();
                depNav2.retainAll(nodeSet);
                // Remove a backedge.
                removeABackedge(scc, depNav2);
                // Break into inner SCCs and sort them.
                SCComponent first2 = breakIntoSCCs(nodeSet, depNav2);
                if (TRACE) {
                    out.print("Order for SCC"+scc.getId()+": ");
                    for (SCComponent scc2 = first2; scc2 != null; scc2 = scc2.nextTopSort()) {
                        out.print(" SCC"+scc2.getId());
                    }
                    out.println();
                }
                innerSCCs.add(scc, first2);
            }
        }
        
        return first;
    }
    
    void removeABackedge(SCComponent scc, InferenceRule.DependenceNavigator depNav) {
        if (TRACE) out.println("SCC"+scc.getId()+" contains: "+scc.nodeSet());
        Object[] entries = scc.entries();
        if (TRACE) out.println("SCC"+scc.getId()+" has "+entries.length+" entries.");
        Object entry;
        if (entries.length > 0) {
            entry = entries[0];
        } else {
            if (TRACE) out.println("No entries, choosing a node.");
            entry = scc.nodes()[0];
        }
        if (TRACE) out.println("Entry into SCC"+scc.getId()+": "+entry);
        
        if (false) {
            Collection preds = depNav.prev(entry);
            if (TRACE) out.println("Predecessors of entry: "+preds);
            Object pred = preds.iterator().next();
            if (TRACE) out.println("Removing backedge "+pred+" -> "+entry);
            depNav.removeEdge(pred, entry);
        } else {
            // find longest path.
            Set visited = new HashSet();
            List queue = new LinkedList();
            queue.add(entry);
            visited.add(entry);
            Object last = null;
            while (!queue.isEmpty()) {
                last = queue.remove(0);
                for (Iterator i = depNav.next(last).iterator(); i.hasNext(); ) {
                    Object q = i.next();
                    if (visited.add(q)) {
                        queue.add(q);
                    }
                }
            }
            if (TRACE) out.println("Last element in SCC: "+last);
            Object last_next;
            List possible = new LinkedList(depNav.next(last));
            if (TRACE) out.println("Successors of last element: "+possible);
            if (possible.size() == 1) last_next = possible.iterator().next();
            else if (possible.contains(entry)) last_next = entry;
            else {
                last_next = possible.iterator().next();
                possible.retainAll(Arrays.asList(entries));
                if (!possible.isEmpty())
                    last_next = possible.iterator().next();
            }
            if (TRACE) out.println("Removing backedge "+last+" -> "+last_next);
            depNav.removeEdge(last, last_next);
        }
    }
    
    boolean again;
    int MAX_ITERATIONS = 128;
    
    boolean iterate(SCComponent first, boolean isLoop) {
        boolean anyChange = false;
        int iterations = 0;
        again = false;
        for (;;) {
            ++iterations;
            boolean outerChange = false;
            SCComponent scc = first;
            while (scc != null) {
                Collection c = innerSCCs.getValues(scc);
                if (!c.isEmpty()) {
                    if (TRACE) out.println("Going inside SCC"+scc.getId());
                    for (Iterator i = c.iterator(); i.hasNext(); ) {
                        SCComponent scc2 = (SCComponent) i.next();
                        boolean b = iterate(scc2, scc.isLoop());
                        if (b) {
                            if (TRACE) out.println("Result changed!");
                            anyChange = true; outerChange = true;
                        }
                    }
                    if (TRACE) out.println("Coming out from SCC"+scc.getId());
                } else for (;;) {
                    boolean innerChange = false;
                    for (Iterator i = scc.nodeSet().iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        if (o instanceof InferenceRule) {
                            InferenceRule ir = (InferenceRule) o;
                            if (TRACE) out.println("Visiting inference rule "+ir);
                            boolean b = ir.update();
                            if (b) {
                                if (TRACE) out.println("Result changed!");
                                anyChange = true; innerChange = true; outerChange = true;
                            }
                        }
                    }
                    if (!scc.isLoop() || !innerChange) break;
                }
                scc = scc.nextTopSort();
            }
            if (!isLoop || !outerChange) break;
            if (iterations == MAX_ITERATIONS) {
                if (TRACE) out.println("Hit max iterations, trying different rules...");
                again = true;
                break;
            }
        }
        return anyChange;
    }
    
    public void solve() {
        
        Set inputs = new HashSet();
        inputs.addAll(solver.relationsToLoad);
        inputs.addAll(solver.relationsToLoadTuples);
        inputs.addAll(solver.equivalenceRelations.values());
        inputs.addAll(solver.notequivalenceRelations.values());
        Set outputs = new HashSet();
        outputs.addAll(solver.relationsToDump);
        outputs.addAll(solver.relationsToDumpTuples);
        outputs.addAll(solver.relationsToDumpNegated);
        outputs.addAll(solver.relationsToDumpNegatedTuples);
        outputs.addAll(solver.relationsToPrintSize);
        
        stratify(solver.rules, inputs, outputs);
        
        Iterator i = firstSCCs.iterator();
        for (int a = 1; i.hasNext(); ++a) {
            SCComponent first = (SCComponent) i.next();
            if (solver.NOISY) out.println("Solving stratum #"+a+"...");
            for (;;) {
                iterate(first, false);
                if (!again) break;
            }
        }
        
    }
    
}

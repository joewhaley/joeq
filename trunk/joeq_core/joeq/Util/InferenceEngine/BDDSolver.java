// BDDSolver.java, created Mar 16, 2004 12:49:19 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import joeq.Util.Collections.MultiMap;
import joeq.Util.Graphs.Navigator;
import joeq.Util.Graphs.SCCTopSortedGraph;
import joeq.Util.Graphs.SCComponent;

/**
 * BDDSolver
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BDDSolver {
    
    public static void main(String[] args) {
        
    }
    
    MultiMap fielddomainsToBDDdomains;
    
    public void solve(RelationSet rs, List/*<InferenceRule>*/ rules) {
        // Build dependence graph.
        Navigator depNav = new InferenceRule.DependenceNavigator(rules);
        
        // Break into SCCs.
        Collection/*<SCComponent>*/ sccs = SCComponent.buildSCC(rules, depNav);
        
        // Find root SCCs.
        Set roots = new HashSet();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (scc.prevLength() == 0) roots.add(scc);
        }
        if (roots.isEmpty()) roots.addAll(sccs);
        
        // Topologically-sort SCCs.
        SCCTopSortedGraph sccGraph = SCCTopSortedGraph.topSort(roots);
        
        
    }
    
}

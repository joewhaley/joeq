// BDDSolver.java, created Mar 16, 2004 12:49:19 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.math.BigInteger;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import joeq.Util.Assert;
import joeq.Util.Collections.GenericMultiMap;
import joeq.Util.Collections.MultiMap;
import joeq.Util.Graphs.Navigator;
import joeq.Util.Graphs.SCCTopSortedGraph;
import joeq.Util.Graphs.SCComponent;

import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;

/**
 * BDDSolver
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BDDSolver extends Solver {
    
    BDDFactory bdd;
    MultiMap fielddomainsToBDDdomains;
    
    public BDDSolver() {
        bdd = BDDFactory.init(1000000, 10000);
        fielddomainsToBDDdomains = new GenericMultiMap();
    }
    
    BDDDomain makeDomain(String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    
    BDDDomain allocateBDDDomain(FieldDomain dom) {
        int version = getBDDDomains(dom).size();
        int bits = BigInteger.valueOf(dom.size).bitCount();
        BDDDomain d = makeDomain(dom.name, bits);
        if (TRACE) out.println("Allocated BDD domain "+d+", "+bits+" bits.");
        
        return d;
    }
    
    Collection getBDDDomains(FieldDomain dom) {
        return fielddomainsToBDDdomains.getValues(dom);
    }
    
    public void solve() {
        // Build dependence graph.
        Navigator depNav = new InferenceRule.DependenceNavigator(rules);
        
        // Break into SCCs.
        Collection/*<SCComponent>*/ sccs = SCComponent.buildSCC(rules, depNav);
        
        if (TRACE) out.println("SCCs: "+sccs);
        
        // Find root SCCs.
        Set roots = new HashSet();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (scc.prevLength() == 0) roots.add(scc);
        }
        if (roots.isEmpty()) roots.addAll(sccs);
        
        if (TRACE) out.println("Roots: "+roots);
        
        // Topologically-sort SCCs.
        SCCTopSortedGraph sccGraph = SCCTopSortedGraph.topSort(roots);
        SCComponent scc = sccGraph.getFirst();
        while (scc != null) {
            if (TRACE) out.println("Visiting SCC "+scc);
            for (;;) {
                boolean change = false;
                for (Iterator i = scc.nodeSet().iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    if (o instanceof InferenceRule) {
                        InferenceRule ir = (InferenceRule) o;
                        if (TRACE) out.println("Visiting inference rule "+ir);
                        boolean b = ir.update();
                        if (b) {
                            if (TRACE) out.println("Result changed!");
                            change = true;
                        }
                    }
                }
                if (!scc.isLoop() || !change) break;
            }
            scc = scc.nextTopSort();
        }
    }

    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.Solver#createInferenceRule(java.util.List, joeq.Util.InferenceEngine.RuleTerm)
     */
    public InferenceRule createInferenceRule(List top, RuleTerm bottom) {
        return new BDDInferenceRule(this, top, bottom);
    }

    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.Solver#createRelation(java.lang.String, java.util.List, java.util.List)
     */
    Relation createRelation(String name, List names, List fieldDomains) {
        return new BDDRelation(this, name, names, fieldDomains);
    }

}

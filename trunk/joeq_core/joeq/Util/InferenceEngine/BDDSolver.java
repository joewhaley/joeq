// BDDSolver.java, created Mar 16, 2004 12:49:19 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;

import joeq.Util.Assert;
import joeq.Util.Collections.GenericMultiMap;
import joeq.Util.Collections.MultiMap;
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
    
    public static String bddDomainInfoFileName = System.getProperty("bddinfo", "bddinfo");

    static boolean USE_NESTED_SCCS = true;
    
    BDDFactory bdd;
    MultiMap fielddomainsToBDDdomains;
    
    public BDDSolver() {
        bdd = BDDFactory.init(1000000, 100000);
        fielddomainsToBDDdomains = new GenericMultiMap();
        bdd.setMaxIncrease(500000);
    }
    
    public void initialize() {
        loadBDDDomainInfo();
    }
    
    void loadBDDDomainInfo() {
        try {
            BufferedReader in = new BufferedReader(new FileReader(bddDomainInfoFileName));
            for (;;) {
                String s = in.readLine();
                if (s == null) break;
                if (s.length() == 0) continue;
                if (s.startsWith("#")) continue;
                StringTokenizer st = new StringTokenizer(s);
                String fieldDomain = st.nextToken();
                FieldDomain fd = (FieldDomain) nameToFieldDomain.get(fieldDomain);
                BDDDomain d = allocateBDDDomain(fd);
            }
        } catch (IOException x) {
        }
    }
    
    public void finish() {
        try {
            saveBDDDomainInfo();
        } catch (IOException x) {
        }
    }
    
    void saveBDDDomainInfo() throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream("r"+bddDomainInfoFileName));
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            BDDDomain d = bdd.getDomain(i);
            for (Iterator j = fielddomainsToBDDdomains.keySet().iterator(); j.hasNext(); ) {
                FieldDomain fd = (FieldDomain) j.next();
                if (fielddomainsToBDDdomains.getValues(fd).contains(d)) {
                    dos.writeBytes(fd.toString()+"\n");
                    break;
                }
            }
        }
    }
    
    BDDDomain makeDomain(String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    
    BDDDomain allocateBDDDomain(FieldDomain dom) {
        int version = getBDDDomains(dom).size();
        int bits = BigInteger.valueOf(dom.size-1).bitLength();
        BDDDomain d = makeDomain(dom.name+version, bits);
        if (TRACE) out.println("Allocated BDD domain "+d+", size "+dom.size+", "+bits+" bits.");
        fielddomainsToBDDdomains.add(dom, d);
        return d;
    }
    
    Collection getBDDDomains(FieldDomain dom) {
        return fielddomainsToBDDdomains.getValues(dom);
    }
    
    MultiMap innerSCCs;
    
    boolean iterate(SCComponent first, boolean isLoop) {
        boolean anyChange = false;
        for (;;) {
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
        }
        return anyChange;
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
        Collection preds = depNav.prev(entry);
        if (TRACE) out.println("Predecessors of entry: "+preds);
        Object pred = preds.iterator().next();
        if (TRACE) out.println("Removing backedge "+pred+" -> "+entry);
        depNav.removeEdge(pred, entry);
    }
    
    public SCComponent buildSCCs(InferenceRule.DependenceNavigator depNav, Collection rules) {
        // Break into SCCs.
        Collection/*<SCComponent>*/ sccs = SCComponent.buildSCC(rules, depNav);
        
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
            if (TRACE) out.println("No roots! Everything is a root.");
            roots.addAll(sccs);
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
                SCComponent first2 = buildSCCs(depNav2, nodeSet);
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
    
    public void solve() {
        innerSCCs = new GenericMultiMap();
        
        // Build dependence graph.
        InferenceRule.DependenceNavigator depNav = new InferenceRule.DependenceNavigator(rules);
        
        SCComponent first = buildSCCs(depNav, rules);
        
        iterate(first, false);
    }

    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.Solver#createInferenceRule(java.util.List, joeq.Util.InferenceEngine.RuleTerm)
     */
    public InferenceRule createInferenceRule(List top, RuleTerm bottom) {
        return new BDDInferenceRule(this, top, bottom);
    }

    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.Solver#createRelation(java.lang.String, java.util.List, java.util.List, java.util.List)
     */
    Relation createRelation(String name, List names, List fieldDomains, List fieldOptions) {
        return new BDDRelation(this, name, names, fieldDomains, fieldOptions);
    }

    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.Solver#readRules(java.io.BufferedReader)
     */
    void readRules(BufferedReader in) throws IOException {
        String varOrderString = System.getProperty("bddvarorder", null);
        
        if (varOrderString != null) {
            int [] varOrder = bdd.makeVarOrdering(true, varOrderString);
            bdd.setVarOrder(varOrder);
        }
        super.readRules(in);
    }
}

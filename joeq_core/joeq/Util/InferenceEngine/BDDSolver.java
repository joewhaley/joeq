// BDDSolver.java, created Mar 16, 2004 12:49:19 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;

import joeq.Util.Assert;
import joeq.Util.PermutationGenerator;
import joeq.Util.Collections.AppendIterator;
import joeq.Util.Collections.GenericMultiMap;
import joeq.Util.Collections.MultiMap;
import joeq.Util.Collections.Pair;

import org.sf.javabdd.BDD;
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

    BDDFactory bdd;
    MultiMap fielddomainsToBDDdomains;
    Map orderingConstraints;
    
    public static int BDDNODES = Integer.parseInt(System.getProperty("bddnodes", "1000000"));
    public static int BDDCACHE = Integer.parseInt(System.getProperty("bddcache", "100000"));
    public static int BDDMINFREE = Integer.parseInt(System.getProperty("bddminfree", "20"));
    
    public BDDSolver() {
        System.out.println("Initializing BDD library ("+BDDNODES+" nodes, cache size "+BDDCACHE+", min free "+BDDMINFREE+"%)");
        bdd = BDDFactory.init(BDDNODES, BDDCACHE);
        fielddomainsToBDDdomains = new GenericMultiMap();
        orderingConstraints = new HashMap();
        bdd.setMaxIncrease(BDDNODES/2);
        bdd.setMinFreeNodes(BDDMINFREE);
    }
    
    public void initialize() {
        loadBDDDomainInfo();
        setVariableOrdering();
        super.initialize();
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
                allocateBDDDomain(fd);
            }
        } catch (IOException x) {
        }
    }
    
    void setVariableOrdering() {
        String varOrderString = System.getProperty("bddvarorder", null);
        if (varOrderString != null) {
            System.out.print("Setting variable ordering to "+varOrderString+", ");
            int [] varOrder = bdd.makeVarOrdering(true, varOrderString);
            bdd.setVarOrder(varOrder);
            System.out.println("done.");
        }
    }
    
    public void solve() {
        Stratify s = new Stratify(this);
        s.solve();
    }
    
    public void finish() {
        try {
            saveBDDDomainInfo();
        } catch (IOException x) {
        }
        calcOrderConstraints();
    }
    
    void calcOrderConstraints() {
        if (orderingConstraints.isEmpty()) return;
        System.out.println("Ordering constraints: "+orderingConstraints);
        Set allDomains = new HashSet();
        for (Iterator i = orderingConstraints.keySet().iterator(); i.hasNext(); ) {
            List list = (List) i.next();
            allDomains.addAll(list);
        }
        List domains = new ArrayList(allDomains);
        PermutationGenerator g = new PermutationGenerator(domains.size());
        List best = null;
        long bestTime = Long.MAX_VALUE;
        while (g.hasMore()) {
            int[] p = g.getNext();
            List domains2 = new ArrayList(p.length);
            for (int k = 0; k < p.length; ++k) {
                domains2.add(domains.get(p[k]));
            }
            long totalTime = 0L;
            for (Iterator i = orderingConstraints.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry) i.next();
                // Check if this ordering constraint matches p.
                List key = (List) e.getKey();
                if (doesOrderMatch(domains2, key)) {
                    //System.out.println(e);
                    totalTime += ((Long) e.getValue()).longValue();
                    if (totalTime < 0) totalTime = Long.MAX_VALUE;
                }
            }
            if (false || totalTime < bestTime) {
                System.out.print("Order: "+domains2);
                System.out.println(" Time: "+totalTime+" ms");
            }
            if (totalTime < bestTime) {
                best = domains2;
                bestTime = totalTime;
            }
        }
        System.out.print("Best order: "+best);
        System.out.println(" Time: "+bestTime+" ms");
    }
    
    static final Long MAX = new Long(Long.MAX_VALUE);
    
    void registerOrderConstraint(List doms, long time) {
        if (time == Long.MAX_VALUE) {
            System.out.println("Invalidating "+doms);
            // special case, obliterate all matching orders.
            for (Iterator i = orderingConstraints.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry) i.next();
                List list = (List) e.getKey();
                if (doesOrderMatch(list, doms)) {
                    if (!e.getValue().equals(MAX)) {
                        System.out.println("Invalidating "+doms+" also invalidates "+list);
                    }
                    e.setValue(MAX);
                } else {
                    //System.out.println("orders don't match. "+list+" and "+doms);
                }
            }
            orderingConstraints.put(doms, MAX);
        } else {
            Long t = (Long) orderingConstraints.get(doms);
            if (t == null) {
                orderingConstraints.put(doms, new Long(time));
            } else {
                time = t.longValue()+time;
                if (time < 0L) orderingConstraints.put(doms, MAX);
                else orderingConstraints.put(doms, new Long(time));
            }
        }
    }
    
    // returns true if a implies b
    static boolean doesOrderMatch(List a, List b) {
        Iterator i = a.iterator();
        Iterator j = b.iterator();
        for (;;) {
            if (!i.hasNext()) return !j.hasNext();
            if (!j.hasNext()) return true;
            Object c = i.next();
            Object d = j.next();
            for (;;) {
                if (c == d) break;
                if (!i.hasNext()) return false;
                c = i.next();
            }
        }
    }
    
    long getOrderConstraint(List doms) {
        Long t = (Long) orderingConstraints.get(doms);
        if (t == null) {
            // check if it matches an invalidated one.
            for (Iterator i = orderingConstraints.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry) i.next();
                if (!e.getValue().equals(MAX)) continue;
                // Check if this ordering constraint matches p.
                List key = (List) e.getKey();
                if (doesOrderMatch(doms, key)) {
                    System.out.println("Order "+doms+" invalidated by "+key);
                    orderingConstraints.put(doms, MAX);
                    return Long.MAX_VALUE;
                }
            }
            return 0L;
        } else {
            return t.longValue();
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

    Relation createEquivalenceRelation(FieldDomain fd) {
        String name = fd+"_eq";
        List names = new Pair(fd+"1", fd+"2");
        List fieldDomains = new Pair(fd, fd);
        List fieldOptions = new Pair("", "");
        BDDRelation r = new BDDRelation(this, name, names, fieldDomains, fieldOptions);
        return r;
    }
    
    Relation createNotEquivalenceRelation(FieldDomain fd) {
        String name = fd+"_neq";
        List names = new Pair(fd+"1", fd+"2");
        List fieldDomains = new Pair(fd, fd);
        List fieldOptions = new Pair("", "");
        BDDRelation r = new BDDRelation(this, name, names, fieldDomains, fieldOptions);
        return r;
    }
    
    void saveResults() throws IOException {
        super.saveResults();
        bdd.done();
    }
    
    void findPhysicalDomainMapping() {
        BDDFactory my_bdd = BDDFactory.init(100000, 10000);
        
        int BITS = BigInteger.valueOf(my_bdd.numberOfDomains()).bitLength();
        
        // one BDDDomain for each relation field.
        Set activeRelations = new HashSet();
        Map fieldOrVarToFieldDomain = new HashMap();
        Map fieldOrVarToBDDDomain = new HashMap();
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule ir = (InferenceRule) i.next();
            for (Iterator j = new AppendIterator(ir.top.iterator(), Collections.singleton(ir.bottom).iterator()); j.hasNext(); ) {
                RuleTerm rt = (RuleTerm) j.next();
                if (activeRelations.add(rt.relation)) {
                    int x = 0;
                    for (Iterator k = rt.relation.fieldDomains.iterator(); k.hasNext(); ++x) {
                        Object field = new Pair(rt.relation, new Integer(x));
                        Assert._assert(!fieldOrVarToFieldDomain.containsKey(field));
                        Assert._assert(!fieldOrVarToBDDDomain.containsKey(field));
                        FieldDomain fd = (FieldDomain)k.next();
                        fieldOrVarToFieldDomain.put(field, fd);
                        BDDDomain dom = makeDomain(my_bdd, field.toString(), BITS);
                        fieldOrVarToBDDDomain.put(field, dom);
                    }
                }
            }
        }
        
        // one BDDDomain for each necessary variable.
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule ir = (InferenceRule) i.next();
            for (Iterator j = new AppendIterator(ir.top.iterator(), Collections.singleton(ir.bottom).iterator()); j.hasNext(); ) {
                RuleTerm rt = (RuleTerm) j.next();
                for (int k = 0; k < rt.variables.size(); ++k) {
                    Variable v = (Variable) rt.variables.get(k);
                    if (!ir.necessaryVariables.contains(v)) continue;
                    FieldDomain fd = (FieldDomain) rt.relation.fieldDomains.get(k);
                    FieldDomain fd2 = (FieldDomain) fieldOrVarToFieldDomain.get(v);
                    Assert._assert(fd2 == null || fd == fd2);
                    fieldOrVarToFieldDomain.put(v, fd);
                    BDDDomain dom = (BDDDomain) fieldOrVarToBDDDomain.get(v);
                    if (dom == null) {
                        dom = makeDomain(my_bdd, v.toString(), BITS);
                        fieldOrVarToBDDDomain.put(v, dom);
                    }
                }
            }
        }
        
        BDD sol = my_bdd.one();
        
        // Every field and variable must be assigned to a physical domain
        // of the appropriate size.
        for (Iterator i = fieldOrVarToFieldDomain.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            BDDDomain my_d = (BDDDomain) fieldOrVarToBDDDomain.get(e.getKey());
            FieldDomain fd = (FieldDomain) e.getValue();
            Collection s = fielddomainsToBDDdomains.getValues(fd);
            BDD t = bdd.zero();
            for (Iterator j = s.iterator(); j.hasNext(); ) {
                BDDDomain d = (BDDDomain) j.next();
                int index = d.getIndex();
                t.orWith(my_d.ithVar(index));
            }
            sol.andWith(t);
        }
        
        // Every field of a particular relation must be assigned to different
        // physical domains.
        for (Iterator i = activeRelations.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            int x = 0;
            for (Iterator j = r.fieldDomains.iterator(); j.hasNext(); ++x) {
                FieldDomain fd1 = (FieldDomain)j.next();
                Object f1 = new Pair(r, new Integer(x));
                BDDDomain dom1 = (BDDDomain) fieldOrVarToBDDDomain.get(f1);
                int y = 0;
                for (Iterator k = r.fieldDomains.iterator(); k.hasNext(); ++y) {
                    FieldDomain fd2 = (FieldDomain)j.next();
                    Object f2 = new Pair(r, new Integer(y));
                    BDDDomain dom2 = (BDDDomain) fieldOrVarToBDDDomain.get(f2);
                    BDD not_eq = dom1.buildEquals(dom2).not();
                    sol.andWith(not_eq);
                }
            }
        }
        
        // Every variable of a single rule must be assigned to a different
        // physical domain.
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule ir = (InferenceRule) i.next();
            for (Iterator j = ir.necessaryVariables.iterator(); j.hasNext(); ) {
                Variable v1 = (Variable) j.next();
                BDDDomain dom1 = (BDDDomain) fieldOrVarToBDDDomain.get(v1);
                for (Iterator k = ir.necessaryVariables.iterator(); k.hasNext(); ) {
                    Variable v2 = (Variable) k.next();
                    BDDDomain dom2 = (BDDDomain) fieldOrVarToBDDDomain.get(v2);
                    BDD not_eq = dom1.buildEquals(dom2).not();
                    sol.andWith(not_eq);
                }
            }
        }
        
        // Set user-specified domains.
        for (Iterator i = activeRelations.iterator(); i.hasNext(); ) {
            BDDRelation r = (BDDRelation) i.next();
            for (int k = 0; k < r.fieldDomains.size(); ++k) {
                String name = (String) r.fieldNames.get(k);
                String option = (String) r.fieldOptions.get(k);
                FieldDomain fd = (FieldDomain) r.fieldDomains.get(k);
                if (!option.startsWith(fd.name))
                    throw new IllegalArgumentException("Field "+name+" has domain "+fd+", but tried to assign "+option);
                Collection doms = getBDDDomains(fd);
                BDDDomain d = null;
                for (Iterator j = doms.iterator(); j.hasNext(); ) {
                    BDDDomain dom = (BDDDomain) j.next();
                    if (dom.getName().equals(option)) {
                        d = dom;
                        break;
                    }
                }
                if (d == null)
                    throw new IllegalArgumentException("Unknown BDD domain "+option);
                int index = d.getIndex();
                Object field = new Pair(r, new Integer(k));
                BDDDomain my_dom = (BDDDomain) fieldOrVarToBDDDomain.get(field);
                sol.andWith(my_dom.ithVar(index));
            }
        }
        
        System.out.println("Solutions to physical domain assignment constraint problem:\n   "+sol.toStringWithDomains());
        
        sol.free();
        my_bdd.done();
    }
    
    static BDDDomain makeDomain(BDDFactory bdd, String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    
    public void reportStats() {
        int final_node_size = bdd.getNodeNum();
        int final_table_size = bdd.getAllocNum();
        System.out.println("MAX_NODES="+final_table_size);
        System.out.println("FINAL_NODES="+final_node_size);
    }
    
}

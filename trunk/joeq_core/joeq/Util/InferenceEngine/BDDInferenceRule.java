// BDDInferenceRule.java, created Mar 16, 2004 3:08:59 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import joeq.Util.Assert;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

/**
 * BDDInferenceRule
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BDDInferenceRule extends InferenceRule {
    
    BDDSolver solver;
    BDD[] oldRelationValues;
    Map variableToBDDDomain;
    boolean incrementalize = true;
    int updateCount;
    long totalTime;
    
    public BDDInferenceRule(BDDSolver solver, List/* <RuleTerm> */ top, RuleTerm bottom) {
        super(top, bottom);
        updateCount = 0;
        this.solver = solver;
        initialize();
    }
    
    void initialize() {
        super.initialize();
        updateCount = 0;
        this.oldRelationValues = null;
        this.variableToBDDDomain = new HashMap();
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                // In the relation, this variable uses domain d
                BDDDomain d = (BDDDomain) r.domains.get(j);
                Assert._assert(d != null);
                // In the rule, we use domain d2
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                if (d2 == null) {
                    if (!variableToBDDDomain.containsValue(d)) {
                        d2 = d;
                    } else {
                        // need to use a new BDDDomain
                        FieldDomain fd = (FieldDomain) r.fieldDomains.get(j);
                        Collection existingBDDDomains = solver.getBDDDomains(fd);
                        for (Iterator k = existingBDDDomains.iterator(); k.hasNext(); ) {
                            BDDDomain d3 = (BDDDomain) k.next();
                            if (!variableToBDDDomain.containsValue(d3)) {
                                d2 = d3;
                                break;
                            }
                        }
                        if (d2 == null) {
                            d2 = solver.allocateBDDDomain(fd);
                        }
                    }
                    if (solver.TRACE) solver.out.println("Variable "+v+" allocated to BDD domain "+d2);
                    variableToBDDDomain.put(v, d2);
                } else {
                    if (solver.TRACE) solver.out.println("Variable "+v+" already allocated to BDD domain "+d2);
                }
            }
        }
    }
    
    void initializeOldRelationValues() {
        this.oldRelationValues = new BDD[top.size()];
        for (int i = 0; i < oldRelationValues.length; ++i) {
            oldRelationValues[i] = solver.bdd.zero();
        }
    }
    
    // non-incremental version.
    public boolean update() {
        ++updateCount;
        if (incrementalize) {
            if (oldRelationValues != null) return updateIncremental();
        }
        
        //if (solver.TRACE)
            solver.out.println("Applying inference rule:\n   "+this+" ("+updateCount+")");
        
        long time = System.currentTimeMillis();
        
        BDD[] relationValues = new BDD[top.size()];
        
        // Quantify out unnecessary fields in input relations.
        if (solver.TRACE) solver.out.println("Necessary variables: "+necessaryVariables);
        if (solver.TRACE) solver.out.println("Unnecessary variables: "+unnecessaryVariables);
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            relationValues[i] = r.relation.id();
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                if (v instanceof Constant) {
                    if (solver.TRACE) solver.out.println("Constant: restricting "+d+" = "+v);
                    relationValues[i].restrictWith(d.ithVar(((Constant)v).value));
                    continue;
                }
                if (unnecessaryVariables.contains(v)) {
                    if (solver.TRACE) solver.out.println(v+" is unnecessary, quantifying out "+d);
                    BDD q = relationValues[i].exist(d.set());
                    relationValues[i].free();
                    relationValues[i] = q;
                }
            }
            if (relationValues[i].isZero()) {
                if (solver.TRACE) solver.out.println("Relation "+r+" is now empty!  Stopping early.");
                for (int j = 0; j <= i; ++j) {
                    relationValues[j].free();
                }
                totalTime += System.currentTimeMillis() - time;
                return false;
            }
        }
        
        // Replace BDDDomain's in the BDD relations to match variable
        // assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            if (solver.TRACE) solver.out.println("Relation "+r+" "+relationValues[i].nodeCount()+" nodes");
            if (solver.TRACE_FULL) solver.out.println("   current value: "+relationValues[i].toStringWithDomains());
            BDDPairing pairing = null;
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                if (unnecessaryVariables.contains(v)) continue;
                BDDDomain d = (BDDDomain) r.domains.get(j);
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                Assert._assert(d2 != null);
                if (d != d2) {
                    if (solver.TRACE) solver.out.println("Variable "+v+": replacing "+d+" -> "+d2);
                    if (pairing == null) pairing = solver.bdd.makePair();
                    pairing.set(d, d2);
                } else {
                    if (solver.TRACE) solver.out.println("Variable "+v+": already matches domain "+d);
                }
            }
            if (pairing != null) {
                if (solver.TRACE)
                    System.out.println("Relation "+r+" domains "+domainsOf(relationValues[i]));
                relationValues[i].replaceWith(pairing);
                if (solver.TRACE)
                    System.out.println("Relation "+r+" domains now "+domainsOf(relationValues[i]));
                pairing.reset();
            }
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE_FULL) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        Set variablesToQuantify = new HashSet(necessaryVariables);
        variablesToQuantify.removeAll(bottom.variables);
        if (solver.TRACE) solver.out.println("Variables to quantify: "+variablesToQuantify);
        List domainsToQuantify = new LinkedList();
        for (Iterator i = variablesToQuantify.iterator(); i.hasNext(); ) {
            Variable v = (Variable) i.next();
            domainsToQuantify.add(variableToBDDDomain.get(v));
        }
        if (solver.TRACE) solver.out.println("Domains to quantify: "+domainsToQuantify);
        
        BDD quantify = solver.bdd.one();
        for (Iterator i = domainsToQuantify.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            quantify.andWith(d.set());
        }
        if (incrementalize) {
            if (oldRelationValues == null) initializeOldRelationValues();
            for (int i = 0; i < relationValues.length; ++i) {
                oldRelationValues[i].orWith(relationValues[i].id());
            }
        }
        BDD topBdd = solver.bdd.one();
        for (int i = 1; i < relationValues.length; ++i) {
            if (solver.TRACE) solver.out.print(" (and nodes: "+topBdd.nodeCount()+"x"+relationValues[i].nodeCount());
            topBdd.andWith(relationValues[i]);
            if (solver.TRACE) solver.out.print("="+topBdd.nodeCount()+")");
            if (topBdd.isZero()) {
                if (solver.TRACE) solver.out.println("Result became empty!  Stopping early.");
                for ( ; i < relationValues.length; ++i) {
                    relationValues[i].free();
                }
                relationValues[0].free();
                totalTime += System.currentTimeMillis() - time;
                return false;
            }
        }
        
        BDD result;
        if (relationValues.length > 0) {
            if (solver.TRACE) solver.out.print(" (relprod nodes: "+topBdd.nodeCount()+"x"+relationValues[0].nodeCount()+"x"+quantify.nodeCount());
            result = topBdd.relprod(relationValues[0], quantify);
            if (solver.TRACE) solver.out.print("="+result.nodeCount()+")");
            relationValues[0].free();
        } else {
            // special case of rule with nothing on the top
            result = solver.bdd.one();
        }
        topBdd.free();
        quantify.free();
        
        BDDPairing pairing = null;
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            if (unnecessaryVariables.contains(v)) continue;            
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                if (solver.TRACE) solver.out.println("Result "+bottom+" variable "+v+": replacing "+d2+" -> "+d);
                if (pairing == null) pairing = solver.bdd.makePair();
                pairing.set(d2, d);
            }
        }
        if (pairing != null) {
            if (solver.TRACE)
                System.out.println("Result domains "+domainsOf(result));
            result.replaceWith(pairing);
            pairing.reset();
        }
        for (int i = 0; i < bottom.variables.size(); ++i) {
            Variable v = (Variable) bottom.variables.get(i);
            if (v instanceof Constant) {
                Constant c = (Constant) v;
                BDDDomain d = (BDDDomain) r.domains.get(i);
                result.andWith(d.ithVar(c.value));
            }
        }
        if (solver.TRACE_FULL) solver.out.println("Adding to "+bottom+": "+result.toStringWithDomains());
        BDD oldRelation = r.relation.id();
        r.relation.orWith(result);
        if (solver.TRACE) solver.out.println("Relation "+r+" is now "+r.relation.nodeCount()+" nodes");
        if (solver.TRACE_FULL) solver.out.println("Relation "+r+" is now: "+r.relation.toStringWithDomains());
        boolean changed = !oldRelation.equals(r.relation);
        oldRelation.free();
        if (solver.TRACE) solver.out.println("Relation "+r+" changed: "+changed);
        totalTime += System.currentTimeMillis() - time;
        return changed;
    }
    
    // Incremental version.
    public boolean updateIncremental() {
        //if (solver.TRACE)
        solver.out.println("Applying inference rule:\n   "+this+" (inc) ("+updateCount+")");
        
        long time = System.currentTimeMillis();
        
        BDD[] allRelationValues = new BDD[top.size()];
        BDD[] newRelationValues = new BDD[top.size()];
        
        // Quantify out unnecessary fields in input relations.
        if (solver.TRACE) solver.out.println("Necessary variables: "+necessaryVariables);
        if (solver.TRACE) solver.out.println("Unnecessary variables: "+unnecessaryVariables);
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            allRelationValues[i] = r.relation.id();
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                if (v instanceof Constant) {
                    if (solver.TRACE) solver.out.println("Constant: restricting "+d+" = "+v);
                    allRelationValues[i].restrictWith(d.ithVar(((Constant)v).value));
                    continue;
                }
                if (unnecessaryVariables.contains(v)) {
                    if (solver.TRACE) solver.out.println(v+" is unnecessary, quantifying out "+d);
                    BDD q = allRelationValues[i].exist(d.set());
                    allRelationValues[i].free();
                    allRelationValues[i] = q;
                }
            }
            if (allRelationValues[i].isZero()) {
                if (solver.TRACE) solver.out.println("Relation "+r+" is now empty!  Stopping early.");
                for (int j = 0; j <= i; ++j)
                    allRelationValues[i].free();
                totalTime += System.currentTimeMillis() - time;
                return false;
            }
        }
        
        // Replace BDDDomain's in the BDD relations to match variable
        // assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            if (solver.TRACE) solver.out.println("Relation "+r+" "+allRelationValues[i].nodeCount()+" nodes");
            if (solver.TRACE_FULL) solver.out.println("   current value: "+allRelationValues[i].toStringWithDomains());
            BDDPairing pairing = null;
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                if (unnecessaryVariables.contains(v)) continue;
                BDDDomain d = (BDDDomain) r.domains.get(j);
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                Assert._assert(d2 != null);
                if (d != d2) {
                    if (solver.TRACE) solver.out.println("Variable "+v+": replacing "+d+" -> "+d2);
                    if (pairing == null) pairing = solver.bdd.makePair();
                    pairing.set(d, d2);
                } else {
                    if (solver.TRACE) solver.out.println("Variable "+v+": already matches domain "+d);
                }
            }
            if (pairing != null) {
                if (solver.TRACE)
                    System.out.println("Relation "+r+" domains "+domainsOf(allRelationValues[i]));
                allRelationValues[i].replaceWith(pairing);
                if (solver.TRACE)
                    System.out.println("Relation "+r+" domains now "+domainsOf(allRelationValues[i]));
                pairing.reset();
            }
            newRelationValues[i] = allRelationValues[i].apply(oldRelationValues[i], BDDFactory.diff);
            oldRelationValues[i].free();
            if (solver.TRACE_FULL) solver.out.println("New for relation "+r+": "+newRelationValues[i].toStringWithDomains());
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE_FULL) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        Set variablesToQuantify = new HashSet(necessaryVariables);
        variablesToQuantify.removeAll(bottom.variables);
        if (solver.TRACE) solver.out.println("Variables to quantify: "+variablesToQuantify);
        List domainsToQuantify = new LinkedList();
        for (Iterator i = variablesToQuantify.iterator(); i.hasNext(); ) {
            Variable v = (Variable) i.next();
            domainsToQuantify.add(variableToBDDDomain.get(v));
        }
        if (solver.TRACE) solver.out.println("Domains to quantify: "+domainsToQuantify);
        
        BDD quantify = solver.bdd.one();
        for (Iterator i = domainsToQuantify.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            quantify.andWith(d.set());
        }
        BDD[] results = new BDD[newRelationValues.length];
    outer:
        for (int i = 0; i < newRelationValues.length; ++i) {
            if (newRelationValues[i].isZero()) {
                if (solver.TRACE) solver.out.println("Nothing new for "+(RuleTerm)top.get(i)+", skipping.");
                newRelationValues[i].free();
                continue;
            }
            BDD topBdd = solver.bdd.one();
            for (int j = 0; j < allRelationValues.length; ++j) {
                if (i == j) continue;
                if (solver.TRACE) solver.out.print(" & " + ((RuleTerm)top.get(j)).relation);
                topBdd.andWith(allRelationValues[j].id());
                if (topBdd.isZero()) {
                    if (solver.TRACE) solver.out.println("Relation "+r+" became empty, skipping.");
                    topBdd.free();
                    newRelationValues[i].free();
                    continue outer;
                }
            }
            if (solver.TRACE) solver.out.print(" x " + ((RuleTerm)top.get(i)).relation+"'");
            if (solver.TRACE) solver.out.print(" (relprod nodes: "+topBdd.nodeCount()+"x"+newRelationValues[i].nodeCount()+"x"+quantify.nodeCount());
            results[i] = topBdd.relprod(newRelationValues[i], quantify);
            if (solver.TRACE) solver.out.print("="+results[i].nodeCount()+")");
            topBdd.free();
            newRelationValues[i].free();
            if (solver.TRACE_FULL) solver.out.println(" = "+results[i].toStringWithDomains());
            else if (solver.TRACE) solver.out.println(" = ");
        }
        quantify.free();
        BDD result = solver.bdd.zero();
        for (int i = 0; i < results.length; ++i) {
            if (results[i] != null) {
                result.orWith(results[i]);
            }
        }
        
        BDDPairing pairing = null;
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            if (unnecessaryVariables.contains(v)) continue;
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                if (solver.TRACE) solver.out.println("Result "+bottom+" variable "+v+": replacing "+d2+" -> "+d);
                if (pairing == null) pairing = solver.bdd.makePair();
                pairing.set(d2, d);
            }
        }
        if (pairing != null) {
            if (solver.TRACE)
                System.out.println("Result domains "+domainsOf(result));
            result.replaceWith(pairing);
            pairing.reset();
        }
        for (int i = 0; i < bottom.variables.size(); ++i) {
            Variable v = (Variable) bottom.variables.get(i);
            if (v instanceof Constant) {
                Constant c = (Constant) v;
                BDDDomain d = (BDDDomain) r.domains.get(i);
                result.andWith(d.ithVar(c.value));
            }
        }
        if (solver.TRACE_FULL) solver.out.println("Adding to "+bottom+": "+result.toStringWithDomains());
        BDD oldRelation = r.relation.id();
        r.relation.orWith(result);
        if (solver.TRACE_FULL) solver.out.println("Relation "+r+" is now: "+r.relation.toStringWithDomains());
        boolean changed = !oldRelation.equals(r.relation);
        oldRelation.free();
        oldRelationValues = allRelationValues;
        if (solver.TRACE) solver.out.println("Relation "+r+" changed: "+changed);
        totalTime += System.currentTimeMillis() - time;
        return changed;
    }

    public void reportStats() {
        System.out.println("Rule "+this);
        System.out.println("   Updates: "+updateCount);
        System.out.println("   Time: "+totalTime+" ms");
    }
    
    /**
     * @param bdd
     * @return
     */
    private String domainsOf(BDD b) {
        BDD s = b.support();
        int[] a = s.scanSetDomains();
        s.free();
        if (a == null) return "(none)";
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < a.length; ++i) {
            sb.append(solver.bdd.getDomain(a[i]));
            if (i < a.length-1) sb.append(',');
        }
        return sb.toString();
    }
    
}

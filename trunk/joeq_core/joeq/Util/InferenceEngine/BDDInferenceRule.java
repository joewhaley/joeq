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
    
    public BDDInferenceRule(BDDSolver solver, List/*<RuleTerm>*/ top, RuleTerm bottom) {
        super(top, bottom);
        this.solver = solver;
        this.oldRelationValues = new BDD[top.size()];
        for (int i = 0; i < oldRelationValues.length; ++i) {
            oldRelationValues[i] = solver.bdd.zero();
        }
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
    
    // non-incremental version.
    public boolean update() {
        if (incrementalize) return updateIncremental();
        
        if (solver.TRACE) solver.out.println("Applying inference rule "+this);
        
        BDD[] relationValues = new BDD[top.size()];
        
        // Replace BDDDomain's in the BDD relations to match variable assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            relationValues[i] = r.relation.id();
            if (solver.TRACE) solver.out.println("Relation "+r+", current value: "+relationValues[i].toStringWithDomains());
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                if (v instanceof Constant) {
                    if (solver.TRACE) solver.out.println("Constant: restricting "+d+" = "+v);
                    relationValues[i].restrictWith(d.ithVar(((Constant)v).value));
                }
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                Assert._assert(d2 != null);
                if (d != d2) {
                    if (solver.TRACE) solver.out.println("Variable "+v+": replacing "+d+" -> "+d2);
                    BDDPairing pairing = solver.bdd.makePair(d, d2);
                    relationValues[i].replaceWith(pairing);
                    pairing.reset();
                } else {
                    if (solver.TRACE) solver.out.println("Variable "+v+": already matches domain "+d);
                }
            }
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        Set variablesToQuantify = new HashSet(variableToBDDDomain.keySet());
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
        BDD topBdd = solver.bdd.one();
        for (int i = 1; i < relationValues.length; ++i) {
            topBdd.andWith(relationValues[i]);
        }
        BDD result = topBdd.relprod(relationValues[0], quantify);
        topBdd.free();
        quantify.free();
        
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                if (solver.TRACE) solver.out.println("Result "+bottom+" variable "+v+": replacing "+d2+" -> "+d);
                BDDPairing pairing = solver.bdd.makePair(d2, d);
                result.replaceWith(pairing);
                pairing.reset();
            }
        }
        if (solver.TRACE) solver.out.println("Adding to "+bottom+": "+result.toStringWithDomains());
        BDD oldRelation = r.relation.id();
        r.relation.orWith(result);
        if (solver.TRACE) solver.out.println("Relation "+r+" is now: "+r.relation.toStringWithDomains());
        boolean changed = !oldRelation.equals(r.relation);
        oldRelation.free();
        return changed;
    }
    
    // Incremental version.
    public boolean updateIncremental() {
        if (solver.TRACE) solver.out.println("Applying inference rule "+this);
        
        BDD[] allRelationValues = new BDD[top.size()];
        BDD[] newRelationValues = new BDD[top.size()];
        
        // Replace BDDDomain's in the BDD relations to match variable assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            allRelationValues[i] = r.relation.id();
            if (solver.TRACE) solver.out.println("Relation "+r+", current value: "+allRelationValues[i].toStringWithDomains());
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                if (v instanceof Constant) {
                    if (solver.TRACE) solver.out.println("Constant: restricting "+d+" = "+v);
                    allRelationValues[i].restrictWith(d.ithVar(((Constant)v).value));
                }
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                Assert._assert(d2 != null);
                if (d != d2) {
                    if (solver.TRACE) solver.out.println("Variable "+v+": replacing "+d+" -> "+d2);
                    BDDPairing pairing = solver.bdd.makePair(d, d2);
                    allRelationValues[i].replaceWith(pairing);
                    pairing.reset();
                } else {
                    if (solver.TRACE) solver.out.println("Variable "+v+": already matches domain "+d);
                }
            }
            newRelationValues[i] = allRelationValues[i].apply(oldRelationValues[i], BDDFactory.diff);
            oldRelationValues[i].free();
            if (solver.TRACE) solver.out.println("New for relation "+r+": "+newRelationValues[i].toStringWithDomains());
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        Set variablesToQuantify = new HashSet(variableToBDDDomain.keySet());
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
        for (int i = 0; i < newRelationValues.length; ++i) {
            BDD topBdd = solver.bdd.one();
            for (int j = 0; j < allRelationValues.length; ++j) {
                if (i == j) continue;
                if (solver.TRACE) solver.out.print(" & " + ((RuleTerm)top.get(j)).relation);
                topBdd.andWith(allRelationValues[j].id());
            }
            if (solver.TRACE) solver.out.print(" x " + ((RuleTerm)top.get(i)).relation+"'");
            results[i] = topBdd.relprod(newRelationValues[i], quantify);
            topBdd.free();
            newRelationValues[i].free();
            if (solver.TRACE) solver.out.println(" = "+results[i].toStringWithDomains());
        }
        BDD result = solver.bdd.zero();
        for (int i = 0; i < results.length; ++i) {
            result.orWith(results[i]);
        }
        
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                if (solver.TRACE) solver.out.println("Result "+bottom+" variable "+v+": replacing "+d2+" -> "+d);
                BDDPairing pairing = solver.bdd.makePair(d2, d);
                result.replaceWith(pairing);
                pairing.reset();
            }
        }
        if (solver.TRACE) solver.out.println("Adding to "+bottom+": "+result.toStringWithDomains());
        BDD oldRelation = r.relation.id();
        r.relation.orWith(result);
        if (solver.TRACE) solver.out.println("Relation "+r+" is now: "+r.relation.toStringWithDomains());
        boolean changed = !oldRelation.equals(r.relation);
        oldRelation.free();
        oldRelationValues = allRelationValues;
        return changed;
    }
    
}
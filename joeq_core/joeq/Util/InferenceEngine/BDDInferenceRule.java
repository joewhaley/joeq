// BDDInferenceRule.java, created Mar 16, 2004 3:08:59 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import java.io.IOException;

import joeq.Util.Assert;
import joeq.Util.PermutationGenerator;
import joeq.Util.Collections.AppendIterator;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.FindBestOrder;

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
    BDDPairing[] renames;
    BDDPairing bottomRename;
    boolean incrementalize = true;
    boolean cache_before_rename = true;
    BDD[] canQuantifyAfter;
    int updateCount;
    long totalTime;
    boolean find_best_order = !System.getProperty("findbestorder", "no").equals("no");
    
    public BDDInferenceRule(BDDSolver solver, List/* <RuleTerm> */ top, RuleTerm bottom) {
        super(top, bottom);
        updateCount = 0;
        this.solver = solver;
        //initialize();
    }
    
    void initialize() {
        super.initialize();
        if (solver.TRACE) solver.out.println("Initializing BDDInferenceRule "+this);
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
                    //if (solver.TRACE) solver.out.println("Variable "+v+" already allocated to BDD domain "+d2);
                }
            }
        }
        if (this.renames == null) {
            renames = new BDDPairing[top.size()];
        }
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            if (renames[i] != null) renames[i].reset();
            renames[i] = calculateRenames(rt, true);
        }
        if (bottomRename != null) bottomRename.reset();
        bottomRename = calculateRenames(bottom, false);
        if (canQuantifyAfter == null) {
            canQuantifyAfter = new BDD[top.size()];
        }
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            if (canQuantifyAfter[i] != null) canQuantifyAfter[i].free();
            canQuantifyAfter[i] = solver.bdd.one();
        outer:
            for (Iterator k = rt.variables.iterator(); k.hasNext(); ) {
                Variable v = (Variable) k.next();
                if (bottom.variables.contains(v)) continue;
                for (int j = i+1; j < top.size(); ++j) {
                    RuleTerm rt2 = (RuleTerm) top.get(j);
                    if (rt2.variables.contains(v)) continue outer;
                }
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                canQuantifyAfter[i].andWith(d2.set());
            }
        }
    }
    
    void initializeOldRelationValues() {
        this.oldRelationValues = new BDD[top.size()];
        for (int i = 0; i < oldRelationValues.length; ++i) {
            oldRelationValues[i] = solver.bdd.zero();
        }
    }
    
    BDDPairing calculateRenames(RuleTerm rt, boolean direction) {
        BDDRelation r = (BDDRelation) rt.relation;
        if (solver.TRACE) solver.out.println("Calculating renames for "+r);
        BDDPairing pairing = null;
        for (int j = 0; j < rt.variables.size(); ++j) {
            Variable v = (Variable) rt.variables.get(j);
            if (unnecessaryVariables.contains(v)) continue;
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                if (!direction) { BDDDomain d3 = d2; d2 = d; d = d3; }
                if (solver.TRACE) solver.out.println(rt+" variable "+v+": replacing "+d+" -> "+d2);
                if (pairing == null) pairing = solver.bdd.makePair();
                pairing.set(d, d2);
            }
        }
        return pairing;
    }
    
    // non-incremental version.
    public boolean update() {
        ++updateCount;
        if (incrementalize) {
            if (oldRelationValues != null) return updateIncremental();
        }
        
        if (solver.NOISY)
            solver.out.println("Applying inference rule:\n   "+this+" ("+updateCount+")");
        
        long time = 0L;
        if (solver.REPORT_STATS || solver.TRACE) time = System.currentTimeMillis();
        
        BDD[] relationValues = new BDD[top.size()];
        
        // Quantify out unnecessary fields in input relations.
        if (solver.TRACE) solver.out.println("Quantifying out unnecessary domains and restricting constants...");
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
                if (solver.REPORT_STATS)
                    totalTime += System.currentTimeMillis() - time;
                if (solver.TRACE)
                    solver.out.println("Time spent: "+(System.currentTimeMillis()-time));
                return false;
            }
        }
        
        // If we are incrementalizing, cache copies of the input relations.
        // This happens after we have quantified away and restricted constants,
        // but before we do renaming.
        if (incrementalize && cache_before_rename) {
            if (solver.TRACE) solver.out.println("Caching values of input relations");
            if (oldRelationValues == null) initializeOldRelationValues();
            for (int i = 0; i < relationValues.length; ++i) {
                oldRelationValues[i].orWith(relationValues[i].id());
            }
        }
        
        // Replace BDDDomain's in the BDD relations to match variable
        // assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            if (solver.TRACE) solver.out.println("Relation "+r+" "+relationValues[i].nodeCount()+" nodes, domains "+domainsOf(relationValues[i]));
            if (solver.TRACE_FULL) solver.out.println("   current value: "+relationValues[i].toStringWithDomains());
            BDDPairing pairing = renames[i];
            if (pairing != null) {
                if (solver.TRACE)
                    System.out.print("Relation "+r+" domains "+domainsOf(relationValues[i])+" -> ");
                relationValues[i].replaceWith(pairing);
                if (solver.TRACE)
                    System.out.println(domainsOf(relationValues[i]));
            }
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE_FULL) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        // If we are incrementalizing, cache copies of the input relations.
        // If the option is set, we do this after the rename.
        if (incrementalize && !cache_before_rename) {
            if (oldRelationValues == null) initializeOldRelationValues();
            for (int i = 0; i < relationValues.length; ++i) {
                oldRelationValues[i].orWith(relationValues[i].id());
            }
        }
        
        BDD result = solver.bdd.one();
        for (int j = 0; j < relationValues.length; ++j) {
            RuleTerm rt = (RuleTerm) top.get(j);
            BDD canNowQuantify = canQuantifyAfter[j];
            if (solver.TRACE) solver.out.print(" x " + rt.relation);
            BDD b = relationValues[j];
            if (find_best_order && !result.isOne()) {
                String varOrder = System.getProperty("bddvarorder");
                findBestDomainOrder(solver.bdd, null, varOrder, result, b, canNowQuantify);
            }
            if (!canNowQuantify.isOne()) {
                if (solver.TRACE) solver.out.print(" (relprod "+b.nodeCount()+"x"+canNowQuantify.nodeCount());
                BDD topBdd = result.relprod(b, canNowQuantify);
                b.free();
                if (solver.TRACE) solver.out.print("="+topBdd.nodeCount()+")");
                result.free(); result = topBdd;
            } else {
                if (solver.TRACE) solver.out.print(" (and "+b.nodeCount());
                result.andWith(b);
                if (solver.TRACE) solver.out.print("="+result.nodeCount()+")");
            }
            if (result.isZero()) {
                if (solver.TRACE) solver.out.println(" Became empty, stopping.");
                for (++j; j < relationValues.length; ++j) {
                    relationValues[j].free();
                }
                if (solver.REPORT_STATS)
                    totalTime += System.currentTimeMillis() - time;
                if (solver.TRACE)
                    solver.out.println("Time spent: "+(System.currentTimeMillis()-time));
                return false;
            }
        }
        if (solver.TRACE_FULL) solver.out.println(" = "+result.toStringWithDomains());
        else if (solver.TRACE) solver.out.println(" = "+result.nodeCount());
        
        if (bottomRename != null) {
            if (solver.TRACE)
                System.out.print("Result domains "+domainsOf(result)+" -> ");
            result.replaceWith(bottomRename);
            if (solver.TRACE)
                System.out.println(domainsOf(result));
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
        if (solver.TRACE) solver.out.println("Relation "+r+" is now "+r.relation.nodeCount()+" nodes, "+r.size()+" elements.");
        if (solver.TRACE_FULL) solver.out.println("Relation "+r+" is now: "+r.relation.toStringWithDomains());
        boolean changed = !oldRelation.equals(r.relation);
        oldRelation.free();
        if (solver.TRACE) solver.out.println("Relation "+r+" changed: "+changed);
        if (solver.REPORT_STATS)
            totalTime += System.currentTimeMillis() - time;
        if (solver.TRACE)
            solver.out.println("Time spent: "+(System.currentTimeMillis()-time));
        return changed;
    }
    
    // Incremental version.
    public boolean updateIncremental() {
        if (solver.NOISY)
            solver.out.println("Applying inference rule:\n   "+this+" (inc) ("+updateCount+")");
        
        long time = 0L;
        if (solver.REPORT_STATS || solver.TRACE) time = System.currentTimeMillis();
        
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
                if (solver.REPORT_STATS)
                    totalTime += System.currentTimeMillis() - time;
                if (solver.TRACE)
                    solver.out.println("Time spent: "+(System.currentTimeMillis()-time));
                return false;
            }
        }
        
        // If we cached before renaming, diff with cache now.
        boolean[] needWholeRelation = null;
        if (cache_before_rename) {
            needWholeRelation = new boolean[allRelationValues.length];
            for (int i = 0; i < allRelationValues.length; ++i) {
                if (!allRelationValues[i].equals(oldRelationValues[i])) {
                    if (solver.TRACE) solver.out.print("Diff relation #"+i+": ("+allRelationValues[i].nodeCount()+"x"+oldRelationValues[i].nodeCount()+"=");
                    newRelationValues[i] = allRelationValues[i].apply(oldRelationValues[i], BDDFactory.diff);
                    if (solver.TRACE) solver.out.println(newRelationValues[i].nodeCount()+")");
                    if (solver.TRACE_FULL) {
                        solver.out.println("New for relation #"+i+": "+newRelationValues[i].toStringWithDomains());
                    }
                    Assert._assert(!newRelationValues[i].isZero());
                    for (int j = 0; j < needWholeRelation.length; ++j) {
                        if (i == j) continue;
                        needWholeRelation[j] = true;
                    }
                }
                oldRelationValues[i].free();
            }
        }
        
        BDD[] rallRelationValues;
        if (cache_before_rename) rallRelationValues = new BDD[top.size()];
        else rallRelationValues = allRelationValues;
        // Replace BDDDomain's in the BDD relations to match variable
        // assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            if (solver.TRACE) solver.out.println("Relation "+r+" "+allRelationValues[i].nodeCount()+" nodes, domains "+domainsOf(allRelationValues[i]));
            if (solver.TRACE_FULL) solver.out.println("   current value: "+allRelationValues[i].toStringWithDomains());
            BDDPairing pairing = renames[i];
            if (cache_before_rename) {
                if (pairing != null) {
                    if (newRelationValues[i] != null) {
                        if (solver.TRACE)
                            System.out.println("Diff for Relation "+r+" domains "+domainsOf(newRelationValues[i]));
                        newRelationValues[i].replaceWith(pairing);
                        if (solver.TRACE)
                            System.out.println("Diff for Relation "+r+" domains now "+domainsOf(newRelationValues[i]));
                    }
                    if (needWholeRelation[i]) {
                        if (solver.TRACE)
                            System.out.println("Whole Relation "+r+" is necessary, renaming it...");
                        rallRelationValues[i] = allRelationValues[i].replace(pairing);
                        if (solver.TRACE)
                            System.out.println("Whole Relation "+r+" domains now "+domainsOf(rallRelationValues[i]));
                    }
                }
                if (rallRelationValues[i] == null) {
                    rallRelationValues[i] = allRelationValues[i].id();
                }
            } else {
                if (pairing != null) {
                    if (solver.TRACE)
                        System.out.println("Relation "+r+" domains "+domainsOf(allRelationValues[i]));
                    allRelationValues[i].replaceWith(pairing);
                    if (solver.TRACE)
                        System.out.println("Relation "+r+" domains now "+domainsOf(allRelationValues[i]));
                }
                if (!allRelationValues[i].equals(oldRelationValues[i])) {
                    if (solver.TRACE) solver.out.print("Diff relation #"+i+": ("+allRelationValues[i].nodeCount()+"x"+oldRelationValues[i].nodeCount()+"=");
                    newRelationValues[i] = allRelationValues[i].apply(oldRelationValues[i], BDDFactory.diff);
                    if (solver.TRACE) solver.out.println(newRelationValues[i].nodeCount()+")");
                    if (solver.TRACE_FULL) solver.out.println("New for relation "+r+": "+newRelationValues[i].toStringWithDomains());
                }
                oldRelationValues[i].free();
            }
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        if (solver.TRACE_FULL) solver.out.println("Current value of relation "+bottom+": "+r.relation.toStringWithDomains());
        
        BDD[] results = new BDD[newRelationValues.length];
    outer:
        for (int i = 0; i < newRelationValues.length; ++i) {
            if (newRelationValues[i] == null) {
                if (solver.TRACE) solver.out.println("Nothing new for "+(RuleTerm)top.get(i)+", skipping.");
                continue;
            }
            Assert._assert(!newRelationValues[i].isZero());
            RuleTerm rt_new = (RuleTerm) top.get(i);
            for (int j = 0; j < i; ++j) {
                RuleTerm rt2 = (RuleTerm) top.get(j);
                if (rt2.relation == rt_new.relation) {
                    boolean skippable = true;
                    for (Iterator k = new AppendIterator(rt2.variables.iterator(), rt_new.variables.iterator()); k.hasNext(); ) {
                        Variable var = (Variable) k.next();
                        if (var instanceof Constant || !necessaryVariables.contains(var)) {
                            skippable = false;
                            break;
                        }
                    }
                    if (skippable) {
                        if (solver.TRACE) solver.out.println("Already did "+(RuleTerm)top.get(i)+", skipping.");
                        newRelationValues[i].free();
                        continue outer;
                    }
                }
            }
            results[i] = solver.bdd.one();
            for (int j = 0; j < rallRelationValues.length; ++j) {
                RuleTerm rt = (RuleTerm) top.get(j);
                BDD canNowQuantify = canQuantifyAfter[j];
                if (solver.TRACE) solver.out.print(" x " + rt.relation);
                BDD b;
                if (i != j) {
                    b = rallRelationValues[j].id();
                } else {
                    b = newRelationValues[i];
                    if (solver.TRACE) solver.out.print("'");
                }
                if (find_best_order && !results[i].isOne()) {
                    String varOrder = System.getProperty("bddvarorder");
                    findBestDomainOrder(solver.bdd, null, varOrder, results[i], b, canNowQuantify);
                }
                if (!canNowQuantify.isOne()) {
                    if (solver.TRACE) solver.out.print(" (relprod "+b.nodeCount()+"x"+canNowQuantify.nodeCount());
                    BDD topBdd = results[i].relprod(b, canNowQuantify);
                    if (solver.TRACE) solver.out.print("="+topBdd.nodeCount()+")");
                    b.free();
                    results[i].free(); results[i] = topBdd;
                } else {
                    if (solver.TRACE) solver.out.print(" (and "+b.nodeCount());
                    results[i].andWith(b);
                    if (solver.TRACE) solver.out.print("="+results[i].nodeCount()+")");
                }
                if (results[i].isZero()) {
                    if (solver.TRACE) solver.out.println(" Became empty, skipping.");
                    if (j < i) newRelationValues[i].free();
                    continue outer;
                }
            }
            if (solver.TRACE_FULL) solver.out.println(" = "+results[i].toStringWithDomains());
            else if (solver.TRACE) solver.out.println(" = "+results[i].nodeCount());
        }
        if (cache_before_rename) {
            for (int i = 0; i < rallRelationValues.length; ++i) {
                rallRelationValues[i].free();
            }
        }
        BDD result = solver.bdd.zero();
        for (int i = 0; i < results.length; ++i) {
            if (results[i] != null) {
                result.orWith(results[i]);
            }
        }
        if (solver.TRACE) solver.out.println("Result: "+result.nodeCount());
        
        if (bottomRename != null) {
            if (solver.TRACE)
                System.out.println("Result domains: "+domainsOf(result));
            result.replaceWith(bottomRename);
            if (solver.TRACE)
                System.out.println("Result domains now: "+domainsOf(result));
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
        if (solver.REPORT_STATS) totalTime += System.currentTimeMillis() - time;
        if (solver.TRACE)
            solver.out.println("Time spent: "+(System.currentTimeMillis()-time));
        return changed;
    }

    public void reportStats() {
        System.out.println("Rule "+this);
        System.out.println("   Updates: "+updateCount);
        System.out.println("   Time: "+totalTime+" ms");
    }
    
    /**
     * @param b
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
    
    static void addDomainsOf(BDD b, Collection domains) {
        BDD s = b.support();
        int[] a = s.scanSetDomains();
        s.free();
        if (a == null) return;
        for (int i = 0; i < a.length; ++i) {
            domains.add(b.getFactory().getDomain(a[i]));
        }
    }
    
    String findBestDomainOrder(BDDFactory bdd,
                               List domains,
                               String origVarOrder,
                               BDD b1, BDD b2, BDD b3) {
        if (domains == null) {
            List domainSet1 = new LinkedList();
            List domainSet2 = new LinkedList();
            List domainSet3 = new LinkedList();
            addDomainsOf(b1, domainSet1);
            addDomainsOf(b2, domainSet2);
            addDomainsOf(b3, domainSet3);
            Set domainSet = new HashSet();
            for (Iterator i = domainSet1.iterator(); i.hasNext(); ) {
                Object o = i.next();
                if (domainSet2.contains(o) || !domainSet3.contains(o))
                    domainSet.add(o);
            }
            for (Iterator i = domainSet2.iterator(); i.hasNext(); ) {
                Object o = i.next();
                if (domainSet1.contains(o) || !domainSet3.contains(o))
                    domainSet.add(o);
            }
            domains = new ArrayList(domainSet);
        }
        String origVarOrder2 = origVarOrder;
        int[] indices = new int[domains.size()];
        for (int i = 0; i < domains.size(); ++i) {
            BDDDomain d1 = (BDDDomain) domains.get(i);
            int index = origVarOrder2.indexOf(d1.getName());
            String name = "$"+i+"$";
            origVarOrder2 = origVarOrder2.substring(0, index) + name + origVarOrder2.substring(index+d1.getName().length());
        }
        List listOfDoms = new LinkedList();
        List listOfOrders = new LinkedList();
        PermutationGenerator g = new PermutationGenerator(domains.size());
        String varOrder2 = origVarOrder;
        while (g.hasMore()) {
            String varOrder = origVarOrder2;
            int[] p = g.getNext();
            int diff = 0;
            for (int i = 0; i < p.length; ++i) {
                BDDDomain d2 = (BDDDomain) domains.get(p[i]);
                String name = "$"+i+"$";
                int index = varOrder.indexOf(name);
                varOrder = varOrder.substring(0, index) + d2.getName() + varOrder.substring(index+name.length());
            }
            List order = getDomainOrder(varOrder, domains, p);
            long t = solver.getOrderConstraint(order);
            if (t == Long.MAX_VALUE) continue;
            listOfDoms.add(order);
            listOfOrders.add(varOrder);
            varOrder2 = varOrder;
        }
        if (listOfDoms.size() <= 1) {
            if (listOfDoms.size() == 0) System.out.println("Warning: no valid permutations for "+domains);
            return varOrder2;
        }
        FindBestOrder fbo = null;
        System.out.println("Trying "+listOfDoms.size()+" permutations of "+domains);
        if (true) {
            try {
                fbo = new FindBestOrder(b1, b2, b3, BDDFactory.and,
                        BDDSolver.BDDCACHE, BDDSolver.BDDNODES/2,
                        Long.MAX_VALUE, 5000);
            } catch (IOException x) {
            }
        }
        String bestVarOrder = origVarOrder;
        List bestOrder = null;
        long bestTime = Long.MAX_VALUE;
        Iterator i = listOfDoms.iterator();
        Iterator j = listOfOrders.iterator();
        while (i.hasNext()) {
            List order = (List) i.next();
            String varOrder = (String) j.next();
            long time;
            if (fbo != null) {
                System.out.println("Trying "+order);
                time = fbo.tryOrder(true, varOrder);
            } else {
                int[] varOrdering = bdd.makeVarOrdering(true, varOrder);
                System.out.print("Setting variable order to "+varOrder+", ");
                bdd.setVarOrder(varOrdering);
                System.out.println("done.");
                System.out.print(b1.nodeCount()+"x"+b2.nodeCount()+" = ");
                time = System.currentTimeMillis();
                BDD result = b1.relprod(b2, b3);
                time = System.currentTimeMillis() - time;
                System.out.println(result.nodeCount()+" ("+time+" ms)");
                result.free();
            }
            solver.registerOrderConstraint(order, time);
            if (time < bestTime) {
                bestTime = time;
                bestOrder = order;
                bestVarOrder = varOrder;
                System.out.println("New best order: "+bestOrder+" time: "+bestTime+" ms");
            }
        }
        System.out.println("Best relative order: "+bestOrder);
        System.out.println("Best variable ordering: "+bestVarOrder);
        return bestVarOrder;
    }
    
    static List getDomainOrder(String varorder, List domains, int[] p) {
        List order = new LinkedList();
        int[] indices = new int[p.length];
        for (int i = 0; i < p.length; ++i) {
            BDDDomain d = (BDDDomain) domains.get(p[i]);
            for (ListIterator j = order.listIterator(); ; ) {
                if (!j.hasNext()) {
                    j.add(d);
                    break;
                }
                BDDDomain d2 = (BDDDomain) j.next();
                if (varorder.indexOf(d.getName()) < varorder.indexOf(d2.getName())) {
                    j.previous();
                    j.add(d);
                    break;
                }
            }
        }
        return order;
    }
    
    public void free() {
        for (int i = 0; i < oldRelationValues.length; ++i) {
            if (oldRelationValues[i] != null) {
                oldRelationValues[i].free();
                oldRelationValues[i] = null;
            }
        }
        for (int i = 0; i < canQuantifyAfter.length; ++i) {
            if (canQuantifyAfter[i] != null) {
                canQuantifyAfter[i].free();
                canQuantifyAfter[i] = null;
            }
        }
        for (int i = 0; i < renames.length; ++i) {
            if (renames[i] != null) {
                renames[i].reset();
                renames[i] = null;
            }
        }
        if (bottomRename != null) {
            bottomRename.reset();
            bottomRename = null;
        }
    }
    
}

// NumberingRule.java, created May 4, 2004 8:57:36 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import java.io.PrintStream;
import java.math.BigInteger;

import joeq.Util.Assert;
import joeq.Util.Graphs.PathNumbering;
import joeq.Util.Graphs.SCCPathNumbering;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;

/**
 * NumberingRule
 * 
 * @author jwhaley
 * @version $Id$
 */
public class NumberingRule extends InferenceRule {
    
    boolean TRACE = true;
    PrintStream out = System.out;
    
    RelationGraph rg;
    PathNumbering pn;
    long totalTime;
    
    NumberingRule(InferenceRule ir) {
        super(ir.top, ir.bottom);
        Assert._assert(ir.top.size() > 1);
    }
    
    void initialize() {
        if (TRACE) out.println("Initializing numbering rule: "+this);
        RuleTerm root = (RuleTerm) top.get(0);
        Variable rootVar;
        if (root.variables.size() == 1) {
            rootVar = (Variable) root.variables.get(0);
        } else {
            List rootVars = new LinkedList(root.variables);
            calculateNecessaryVariables();
            rootVars.retainAll(necessaryVariables);
            Assert._assert(rootVars.size() == 1);
            rootVar = (Variable) rootVars.get(0);
        }
        if (TRACE) out.println("Root variable: "+rootVar);
        List edges = top.subList(1, top.size());
        rg = new RelationGraph(root, rootVar, edges);
    }
    
    public Collection/*<InferenceRule>*/ split(int myIndex, Solver s) {
        throw new InternalError("Cannot split a numbering rule!");
    }
    
    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.InferenceRule#update()
     */
    public boolean update() {
        if (pn != null) {
            if (TRACE) out.println("Numbering already calculated, skipping.");
            return false;
        }
        
        long time = System.currentTimeMillis();
        
        pn = new SCCPathNumbering();
        BigInteger num = pn.countPaths(rg);
        Iterator i = bottom.variables.iterator();
        Variable v1, v2;
        v1 = (Variable) i.next();
        v2 = (Variable) i.next();
        if (TRACE) out.println("Finding relations with ("+v1+","+v2+")");
        
        // Which relation(s) are we talking about here?
        for (i = rg.edges.iterator(); i.hasNext(); ) {
            RuleTerm rt = (RuleTerm) i.next();
            if (rt.variables.get(0) == v1 &&
                rt.variables.get(1) == v2) {
                if (TRACE) out.println("Match: "+rt);
                
                // TODO: generalize this to be not BDD-specific
                BDDRelation bddr = (BDDRelation) bottom.relation;
                Iterator k = bddr.domains.iterator();
                BDDDomain d0, d1, d2, d3;
                d0 = (BDDDomain) k.next();
                d1 = (BDDDomain) k.next();
                if (TRACE) out.println("Domains for edge: "+d0+" -> "+d1);
                d2 = (BDDDomain) k.next();
                d3 = (BDDDomain) k.next();
                if (TRACE) out.println("Domains for numbering: "+d2+" -> "+d3);
                Assert._assert(d0 != d1);
                Assert._assert(d2 != d3);
                
                for (TupleIterator j = rt.relation.iterator(); j.hasNext(); ) {
                    long[] t = j.nextTuple();
                    Object source = RelationGraph.makeGraphNode(v1, t[0]);
                    Object target = RelationGraph.makeGraphNode(v2, t[1]);
                    PathNumbering.Range r0 = pn.getRange(source);
                    PathNumbering.Range r1 = pn.getEdge(source, target);
                    if (TRACE) out.println("Edge: "+source+" -> "+target+"\t"+r0+" -> "+r1);
                    
                    // TODO: generalize this to be not BDD-specific
                    BDD result = buildMap(d2, PathNumbering.toBigInt(r0.low), PathNumbering.toBigInt(r0.high),
                                          d3, PathNumbering.toBigInt(r1.low), PathNumbering.toBigInt(r1.high));
                    result.andWith(d0.ithVar(t[0]));
                    result.andWith(d1.ithVar(t[1]));
                    bddr.relation.orWith(result);
                }
            }
        }
        
        time = System.currentTimeMillis() - time;
        if (TRACE) out.println("Time spent: "+time+" ms");
        
        totalTime += time;
        
        return true;
    }

    public static BDD buildMap(BDDDomain d1, BigInteger startD1, BigInteger endD1,
                               BDDDomain d2, BigInteger startD2, BigInteger endD2)
    {
        BDD r;
        BigInteger sizeD1 = endD1.subtract(startD1);
        BigInteger sizeD2 = endD2.subtract(startD2);
        if (sizeD1.signum() == -1) {
            r = d2.varRange(startD2.longValue(), endD2.longValue());
            r.andWith(d1.ithVar(0));
        } else if (sizeD2.signum() == -1) {
            r = d1.varRange(startD1.longValue(), endD1.longValue());
            r.andWith(d2.ithVar(0));
        } else {
            int bits;
            if (endD1.compareTo(endD2) != -1) { // >=
                bits = endD1.bitLength();
            } else {
                bits = endD2.bitLength();
            }
            long val = startD2.subtract(startD1).longValue();
            r = d1.buildAdd(d2, bits, val);
            if (sizeD2.compareTo(sizeD1) != -1) { // >=
                // D2 is bigger, or they are equal.
                r.andWith(d1.varRange(startD1.longValue(), endD1.longValue()));
            } else {
                // D1 is bigger.
                r.andWith(d2.varRange(startD2.longValue(), endD2.longValue()));
            }
        }
        return r;
    }
    
    /* (non-Javadoc)
     * @see joeq.Util.InferenceEngine.InferenceRule#reportStats()
     */
    public void reportStats() {
        System.out.println("Rule "+this);
        System.out.println("   Time: "+totalTime+" ms");
    }
}

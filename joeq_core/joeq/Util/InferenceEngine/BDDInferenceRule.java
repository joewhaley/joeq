// BDDInferenceRule.java, created Mar 16, 2004 3:08:59 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import joeq.Util.Assert;
import joeq.Util.Collections.SortedArraySet;

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
    
    BDDFactory bdd;
    BDD[] oldRelationValues;
    Map variableToBDDDomain;
    
    public BDDInferenceRule(BDDFactory bdd, List top, RuleTerm bottom) {
        super(top, bottom);
        this.bdd = bdd;
        this.oldRelationValues = new BDD[top.size()];
        for (int i = 0; i < oldRelationValues.length; ++i) {
            oldRelationValues[i] = bdd.zero();
        }
        this.variableToBDDDomain = new HashMap();
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                if (d2 == null) variableToBDDDomain.put(v, d2 = d);
            }
        }
    }
    
    // Incremental version.
    public void update() {
        BDD[] allRelationValues = new BDD[top.size()];
        BDD[] newRelationValues = new BDD[top.size()];
        
        // Replace BDDDomain's in the BDD relations to match variable assignments.
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            BDDRelation r = (BDDRelation) rt.relation;
            allRelationValues[i] = r.relation.id();
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                BDDDomain d = (BDDDomain) r.domains.get(j);
                BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
                Assert._assert(d2 != null);
                if (d != d2) {
                    BDDPairing pairing = bdd.makePair(d, d2);
                    allRelationValues[i].replaceWith(pairing);
                    pairing.reset();
                }
            }
            newRelationValues[i] = allRelationValues[i].apply(oldRelationValues[i], BDDFactory.diff);
            oldRelationValues[i].free();
        }
        BDDRelation r = (BDDRelation) bottom.relation;
        Set domainsToQuantify = SortedArraySet.FACTORY.makeSet(variableToBDDDomain.values());
        domainsToQuantify.removeAll(r.domains);
        
        BDD quantify = bdd.one();
        for (Iterator i = domainsToQuantify.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            quantify.andWith(d.set());
        }
        BDD[] results = new BDD[newRelationValues.length];
        for (int i = 0; i < newRelationValues.length; ++i) {
            BDD topBdd = bdd.one();
            for (int j = 0; j < allRelationValues.length; ++j) {
                if (i == j) continue;
                topBdd.andWith(allRelationValues[j].id());
            }
            results[i] = topBdd.relprod(newRelationValues[i], quantify);
            topBdd.free();
            newRelationValues[i].free();
        }
        BDD result = bdd.zero();
        for (int i = 0; i < results.length; ++i) {
            result.orWith(results[i]);
        }
        
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            BDDDomain d = (BDDDomain) r.domains.get(j);
            BDDDomain d2 = (BDDDomain) variableToBDDDomain.get(v);
            Assert._assert(d2 != null);
            if (d != d2) {
                BDDPairing pairing = bdd.makePair(d, d2);
                result.replaceWith(pairing);
                pairing.reset();
            }
        }
        r.relation.orWith(result);
        oldRelationValues = allRelationValues;
    }
    
}

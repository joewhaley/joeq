// InferenceRule.java, created Mar 16, 2004 12:41:14 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import joeq.Util.Assert;
import joeq.Util.Collections.GenericMultiMap;
import joeq.Util.Collections.MultiMap;
import joeq.Util.Graphs.Navigator;

/**
 * InferenceRule
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class InferenceRule {
    
    List/*<RuleTerm>*/ top;
    RuleTerm bottom;
    Set necessaryVariables;
    Set unnecessaryVariables;
    
    /**
     * @param top
     * @param bottom
     */
    protected InferenceRule(List/*<RuleTerm>*/ top, RuleTerm bottom) {
        super();
        this.top = top;
        this.bottom = bottom;
    }
    
    void initialize() {
        calculateNecessaryVariables();
    }
    
    static Set calculateNecessaryVariables(Collection s, List terms) {
        Set necessaryVariables = new HashSet();
        Set unnecessaryVariables = new HashSet(s);
        for (int i = 0; i < terms.size(); ++i) {
            RuleTerm rt = (RuleTerm) terms.get(i);
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                if (necessaryVariables.contains(v)) continue;
                if (unnecessaryVariables.contains(v)) {
                    necessaryVariables.add(v);
                    unnecessaryVariables.remove(v);
                } else {
                    unnecessaryVariables.add(v);
                }
            }
        }
        return necessaryVariables;
    }
    
    Set calculateNecessaryVariables() {
        necessaryVariables = new HashSet();
        unnecessaryVariables = new HashSet();
        for (int i = 0; i < top.size(); ++i) {
            RuleTerm rt = (RuleTerm) top.get(i);
            for (int j = 0; j < rt.variables.size(); ++j) {
                Variable v = (Variable) rt.variables.get(j);
                if (necessaryVariables.contains(v)) continue;
                if (unnecessaryVariables.contains(v)) {
                    necessaryVariables.add(v);
                    unnecessaryVariables.remove(v);
                } else {
                    unnecessaryVariables.add(v);
                }
            }
        }
        
        for (int j = 0; j < bottom.variables.size(); ++j) {
            Variable v = (Variable) bottom.variables.get(j);
            if (necessaryVariables.contains(v)) continue;
            if (unnecessaryVariables.contains(v)) {
                necessaryVariables.add(v);
                unnecessaryVariables.remove(v);
            } else {
                unnecessaryVariables.add(v);
            }
        }
        return necessaryVariables;
    }
    
    public abstract boolean update();
    
    public static MultiMap getRelationToUsingRule(List/*<InferenceRule>*/ rules) {
        MultiMap mm = new GenericMultiMap();
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule ir = (InferenceRule) i.next();
            for (Iterator j = ir.top.iterator(); j.hasNext(); ) {
                RuleTerm rt = (RuleTerm) j.next();
                mm.add(rt.relation, ir);
            }
        }
        return mm;
    }
    
    public Collection/*<InferenceRule>*/ split(Solver s) {
        List newRules = new LinkedList();
        int myIndex = s.rules.indexOf(this);
        int count = 0;
        while (top.size() > 2) {
            RuleTerm rt1 = (RuleTerm) top.remove(0);
            RuleTerm rt2 = (RuleTerm) top.remove(0);
            if (s.NOISY) s.out.println("Combining "+rt1+" and "+rt2+" into a new rule.");
            
            // Calculate our new necessary variables.
            LinkedList ll = new LinkedList();
            ll.addAll(rt1.variables); ll.addAll(rt2.variables);
            LinkedList terms = new LinkedList(top); terms.add(bottom);
            Set myNewNecessaryVariables = calculateNecessaryVariables(ll, terms);
            
            List newTop = new LinkedList();
            newTop.add(rt1);
            newTop.add(rt2);
            // Make a new relation for the bottom.
            Map neededVariables = new HashMap();
            Iterator i = rt1.variables.iterator();
            Iterator j = rt1.relation.fieldDomains.iterator();
            while (i.hasNext()) {
                Variable v = (Variable) i.next();
                FieldDomain d = (FieldDomain) j.next();
                if (!myNewNecessaryVariables.contains(v)) continue;
                FieldDomain d2 = (FieldDomain) neededVariables.get(v);
                if (d2 != null && d != d2) {
                    throw new IllegalArgumentException(v+": "+d+" != "+d2);
                }
                neededVariables.put(v, d);
            }
            i = rt2.variables.iterator();
            j = rt2.relation.fieldDomains.iterator();
            while (i.hasNext()) {
                Variable v = (Variable) i.next();
                FieldDomain d = (FieldDomain) j.next(); 
                if (!myNewNecessaryVariables.contains(v)) continue;
                FieldDomain d2 = (FieldDomain) neededVariables.get(v);
                if (d2 != null && d != d2) {
                    throw new IllegalArgumentException(v+": "+d+" != "+d2);
                }
                neededVariables.put(v, d);
            }
            // Make a new relation for the bottom.
            List fieldNames = new LinkedList();
            List fieldDomains = new LinkedList();
            List fieldOptions = new LinkedList();
            List newVariables = new LinkedList();
            for (i = neededVariables.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry) i.next();
                Variable v = (Variable) e.getKey();
                FieldDomain d = (FieldDomain) e.getValue();
                fieldNames.add("_"+v);
                fieldDomains.add(d);
                fieldOptions.add("");
                newVariables.add(v);
            }
            String relationName = "syn_"+myIndex+"_"+count;
            Relation newRelation = s.createRelation(relationName, fieldNames, fieldDomains, fieldOptions);
            if (s.NOISY) s.out.println("New relation: "+newRelation);
            RuleTerm newBottom = new RuleTerm(newVariables, newRelation);
            InferenceRule newRule = s.createInferenceRule(newTop, newBottom);
            if (s.NOISY) s.out.println("New rule: "+newRule);
            if (s.NOISY) s.out.println("Necessary variables: "+newRule.necessaryVariables);
            //s.rules.add(newRule);
            newRules.add(newRule);
            // Now include the bottom of the new rule on the top of our rule.
            top.add(0, newBottom);
            // Reinitialize this rule because the terms have changed.
            this.initialize();
            if (s.NOISY) s.out.println("Current rule is now: "+this);
            if (s.NOISY) s.out.println("My new necessary variables: "+necessaryVariables);
            Assert._assert(necessaryVariables.equals(myNewNecessaryVariables));
            ++count;
        }
        return newRules;
    }
    
    public static MultiMap getRelationToDefiningRule(List/*<InferenceRule>*/ rules) {
        MultiMap mm = new GenericMultiMap();
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule ir = (InferenceRule) i.next();
            mm.add(ir.bottom.relation, ir);
        }
        return mm;
    }
    
    public static class DependenceNavigator implements Navigator {

        MultiMap relationToUsingRule;
        MultiMap relationToDefiningRule;
        
        public DependenceNavigator(List/*<InferenceRule>*/ rules) {
            this(getRelationToUsingRule(rules), getRelationToDefiningRule(rules));
        }
        /**
         * @param relationToUsingRule
         * @param relationToDefiningRule
         */
        public DependenceNavigator(MultiMap relationToUsingRule,
                                   MultiMap relationToDefiningRule) {
            super();
            this.relationToUsingRule = relationToUsingRule;
            this.relationToDefiningRule = relationToDefiningRule;
        }
        
        /* (non-Javadoc)
         * @see joeq.Util.Graphs.Navigator#next(java.lang.Object)
         */
        public Collection next(Object node) {
            if (node instanceof InferenceRule) {
                InferenceRule ir = (InferenceRule) node;
                return Collections.singleton(ir.bottom.relation);
            } else {
                Relation r = (Relation) node;
                Collection c = relationToUsingRule.getValues(r);
                return c;
            }
        }

        /* (non-Javadoc)
         * @see joeq.Util.Graphs.Navigator#prev(java.lang.Object)
         */
        public Collection prev(Object node) {
            if (node instanceof InferenceRule) {
                InferenceRule ir = (InferenceRule) node;
                List list = new LinkedList();
                for (Iterator i = ir.top.iterator(); i.hasNext(); ) {
                    RuleTerm rt = (RuleTerm) i.next();
                    list.add(rt.relation);
                }
                return list;
            } else {
                Relation r = (Relation) node;
                Collection c = relationToDefiningRule.getValues(r);
                return c;
            }
        }
        
    }
    
    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (Iterator i = top.iterator(); i.hasNext(); ) {
            sb.append(i.next());
            if (i.hasNext()) sb.append(" , ");
        }
        sb.append(" / ");
        sb.append(bottom);
        return sb.toString();
    }
}

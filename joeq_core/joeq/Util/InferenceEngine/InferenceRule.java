// InferenceRule.java, created Mar 16, 2004 12:41:14 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

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
    
    public abstract static class Factory {
        public abstract InferenceRule create(List top, RuleTerm bottom);
    }
    
    /**
     * @param top
     * @param bottom
     */
    protected InferenceRule(List top, RuleTerm bottom) {
        super();
        this.top = top;
        this.bottom = bottom;
    }
    
    static Factory factory;
    
    public static InferenceRule parseRule(RelationSet rs, StringTokenizer st) {
        Map/*<String,Variable>*/ nameToVar = new HashMap();
        List/*<RuleTerm>*/ terms = new LinkedList();
        for (;;) {
            RuleTerm rt = parseRuleTerm(rs, nameToVar, st);
            terms.add(rt);
            String sep = st.nextToken();
            if (sep.equals("/")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException();
        }
        RuleTerm bottom = parseRuleTerm(rs, nameToVar, st);
        InferenceRule ir = factory.create(terms, bottom);
        return null;
    }
    
    static RuleTerm parseRuleTerm(RelationSet rs, Map/*<String,Variable>*/ nameToVar, StringTokenizer st) {
        String openParen = st.nextToken();
        if (!openParen.equals("(")) throw new IllegalArgumentException();
        List/*<Variable>*/ vars = new LinkedList();
        for (;;) {
            String varName = st.nextToken();
            Variable var = null;
            if (!varName.equals("_")) {
                var = (Variable) nameToVar.get(varName);
                if (var == null) nameToVar.put(varName, var = new Variable(varName));
            } else {
                var = new Variable("_");
            }
            vars.add(var);
            String sep = st.nextToken();
            if (sep.equals(")")) break;
            if (!sep.equals(",")) break;
        }
        String in = st.nextToken();
        if (!in.equals("in")) throw new IllegalArgumentException();
        String relationName = st.nextToken();
        Relation r = rs.getOrCreate(relationName, vars);
        RuleTerm rt = new RuleTerm(vars, r);
        return rt;
    }
    
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
    
}

// CachedCallGraph.java, created Sat Mar 29  0:56:01 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.*;
import Util.Collections.GenericInvertibleMultiMap;
import Util.Collections.GenericMultiMap;
import Util.Collections.InvertibleMultiMap;
import Util.Collections.MultiMap;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class CachedCallGraph extends CallGraph {

    private final CallGraph delegate;

    private Set/*<jq_Method>*/ methods;
    private MultiMap/*<jq_Method,ProgramLocation>*/ callSites;
    private InvertibleMultiMap/*<ProgramLocation,jq_Method>*/ edges;

    public CachedCallGraph(CallGraph cg) {
        this.delegate = cg;
    }

    public void invalidateCache() {
        this.edges = new GenericInvertibleMultiMap();
        for (Iterator i = delegate.getAllCallSites().iterator(); i.hasNext(); ) {
            ProgramLocation p = (ProgramLocation) i.next();
            Collection callees = this.edges.getValues(p);
            Collection callees2 = delegate.getTargetMethods(p);
            callees.addAll(callees2);
        }
        this.methods = new HashSet();
        this.callSites = new GenericMultiMap();
        for (Iterator i = this.edges.keySet().iterator(); i.hasNext(); ) {
            ProgramLocation p = (ProgramLocation) i.next();
            jq_Method m = (jq_Method) p.getMethod();
            methods.add(m);
            methods.addAll(this.edges.getValues(p));
            Collection c = callSites.getValues(m);
            c.add(p);
        }
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#setRoots(java.util.Collection)
     */
    public void setRoots(Collection roots) {
        delegate.setRoots(roots);
        invalidateCache();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#entrySet()
     */
    public Set entrySet() {
        if (edges == null) invalidateCache();
        return edges.entrySet();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getAllCallSites()
     */
    public Collection getAllCallSites() {
        if (edges == null) invalidateCache();
        if (true) {
            return edges.keySet();
        } else {
            return callSites.values();
        }
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getAllMethods()
     */
    public Collection getAllMethods() {
        if (edges == null) invalidateCache();
        if (true) {
            return methods;
        } else {
            LinkedHashSet allMethods = new LinkedHashSet(edges.values());
            allMethods.addAll(delegate.getRoots());
            return allMethods;
        }
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallees(Compil3r.Quad.ControlFlowGraph)
     */
    public Collection getCallees(ControlFlowGraph cfg) {
        return getCallees(cfg.getMethod());
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallees(Clazz.jq_Method)
     */
    public Collection getCallees(jq_Method caller) {
        if (edges == null) invalidateCache();
        return getFromMultiMap(callSites, edges, caller);
    }

    public static Collection getFromMultiMap(MultiMap m1, MultiMap m2, jq_Method method) {
        Collection c1 = m1.getValues(method);
        Iterator i = c1.iterator();
        if (!i.hasNext()) return Collections.EMPTY_SET;
        Object o = i.next();
        if (!i.hasNext()) return m2.getValues(o);
        Set result = new LinkedHashSet();
        for (;;) {
            result.addAll(m2.getValues(o));
            if (!i.hasNext()) break;
            o = i.next();
        }
        return result;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallers(Clazz.jq_Method)
     */
    public Collection getCallers(jq_Method callee) {
        if (edges == null) invalidateCache();
        MultiMap m1 = edges.invert();
        Collection c1 = m1.getValues(callee);
        Iterator i = c1.iterator();
        if (!i.hasNext()) return Collections.EMPTY_SET;
        ProgramLocation o = (ProgramLocation) i.next();
        if (!i.hasNext()) return Collections.singleton(o.getMethod());
        Set result = new LinkedHashSet();
        for (;;) {
            result.add(o.getMethod());
            if (!i.hasNext()) break;
            o = (ProgramLocation) i.next();
        }
        return result;
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallSites(Compil3r.Quad.ControlFlowGraph)
     */
    public Collection getCallSites(ControlFlowGraph cfg) {
        return getCallSites(cfg.getMethod());
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallSites(Clazz.jq_Method)
     */
    public Collection getCallSites(jq_Method caller) {
        if (callSites == null) invalidateCache();
        return callSites.getValues(caller);
    }

    /**
     * @param context
     * @param callSite
     * @return
     */
    public Collection getTargetMethods(Object context, ProgramLocation callSite) {
        if (edges == null) invalidateCache();
        return edges.getValues(callSite);
    }
    
    public void inlineEdge(jq_Method caller, ProgramLocation callSite, jq_Method callee) {
        if (false) System.out.println("Inlining edge "+callSite+" -> "+callee);
        // remove call site from caller.
        callSites.remove(caller, callSite);
        // add all call sites in callee into caller.
        callSites.addAll(caller, callSites.getValues(callee));
    }

    /* (non-Javadoc)
     * @see java.util.AbstractMap#keySet()
     */
    public Set keySet() {
        if (edges == null) invalidateCache();
        return edges.keySet();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getRoots()
     */
    public Collection getRoots() {
        return delegate.getRoots();
    }

}

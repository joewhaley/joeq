// CSPAResults.java, created Aug 7, 2003 12:34:24 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.IPA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_Field;
import joeq.Class.jq_Method;
import joeq.Class.jq_NameAndDesc;
import joeq.Class.jq_Reference;
import joeq.Class.jq_Type;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.FieldNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.HeapObject;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.Node;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.PassedParameter;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ReturnValueNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ThrownExceptionNode;
import joeq.Compiler.Analysis.IPA.PA.ThreadRootMap;
import joeq.Compiler.Analysis.IPA.PA.VarPathSelector;
import joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import joeq.Compiler.Quad.CallGraph;
import joeq.Compiler.Quad.CodeCache;
import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.LoadedCallGraph;
import joeq.Compiler.Quad.Operator;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.QuadIterator;
import joeq.Main.HostedVM;
import jwutil.collections.GenericInvertibleMultiMap;
import jwutil.collections.GenericMultiMap;
import jwutil.collections.IndexMap;
import jwutil.collections.InvertibleMultiMap;
import jwutil.collections.MultiMap;
import jwutil.collections.Pair;
import jwutil.collections.SortedArraySet;
import jwutil.collections.UnmodifiableIterator;
import jwutil.graphs.Navigator;
import jwutil.graphs.PathNumbering;
import jwutil.graphs.SCCPathNumbering;
import jwutil.graphs.SCCTopSortedGraph;
import jwutil.graphs.SCComponent;
import jwutil.graphs.Traversals;
import jwutil.graphs.PathNumbering.Range;
import jwutil.graphs.SCCPathNumbering.Path;
import jwutil.strings.Strings;
import jwutil.util.Assert;
import net.sf.javabdd.BDD;
import net.sf.javabdd.BDDDomain;
import net.sf.javabdd.BDDFactory;
import net.sf.javabdd.BDDPairing;

/**
 * Records results for context-sensitive pointer analysis.  The results can
 * be saved and reloaded.  This class also provides methods to query the results.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class CSPAResults {

    /** Call graph. */
    CallGraph cg;

    /** Path numbering for call graph. */
    SCCPathNumbering pn;

    /** BDD factory object, to perform BDD operations. */
    BDDFactory bdd;

    /** Map between variables and indices in the V1/V2 domain. */
    IndexMap Vmap;
    /** Map between invocations and indices in the I domain. */
    IndexMap Imap;
    /** Map between heap objects and indices in the H1/H2 domain. */
    IndexMap Hmap;
    /** Map between methods and indices in the M domain. */
    IndexMap Mmap;
    /** Map between fields and indices in the F domain. */
    IndexMap Fmap;

    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    BDDDomain V1c, V2c, H1c, H2c;
    BDDDomain V3, V3c, H3, H3c, I2, M2, F2;
    
    BDD A;      // V1xV2, arguments and return values   (+context)
    
    /** Variable points-to BDD: V1c x V1 x H1c x H1.
     * This contains the result of the points-to analysis.
     * A relation (V,H) is in the BDD if variable V can point to heap object H.
     */
    BDD vP;

    /** Stores BDD: V2c x V2 x F x V1c x V1. 
     * Holds the store instructions in the program.
     * A relation (V2c,V2,F,V1c,V1) is in the BDD if the program contains
     * a store instruction of the form v1.F = v2;
     */
    BDD S;

    /** Loads BDD: V2c x V2 x V1c x V1 x F. 
     * Holds the load instructions in the program.
     * A relation (V2c,V2,V1c,V1,F) is in the BDD if the program contains
     * a load instruction of the form v2 = v1.F;
     */
    BDD L;
    
    BDD vT;     // V1xT1, variable type                 (no context)
    BDD hT;     // H1xT2, heap type                     (no context)
    BDD aT;     // T1xT2, assignable types              (no context)
    BDD cha;    // T2xNxM, class hierarchy information  (no context)
    BDD actual; // IxZxV2, actual parameters            (no context)
    BDD formal; // MxZxV1, formal parameters            (no context)
    BDD Iret;   // IxV1, invocation return value        (no context)
    BDD Mret;   // MxV2, method return value            (no context)
    BDD Ithr;   // IxV1, invocation thrown value        (no context)
    BDD Mthr;   // MxV2, method thrown value            (no context)
    BDD mI;     // MxIxN, method invocations            (no context)
    BDD mV;     // MxV, method variables                (no context)
    
    /** Heap points-to BDD: H1c x H1 x F x H2c x H2.
     * This contains the result of the points-to analysis.
     * A relation (H1,F,H2) is in the BDD if the field F of heap object H1 can
     * point to heap object H2.
     */
    BDD hP;
    
    BDD IE;     // IxM, invocation edges                (no context)
    BDD filter; // V1xH1, type filter                   (no context)
    BDD IEc;    // V2cxIxV1cxM, context-sensitive edges
    
    /** Accessible locations BDD: V1c x V1 x V2c x V2.
     * This contains the call graph relation.
     * A relation (V1c,V1,V2c,V2) is in the BDD if the method containing V1
     * under the context V1c calls the method containing V2 under the context
     * V2c.
     */
    BDD callGraphRelation;
    
    /** Accessible locations BDD: V1c x V1 x V2c x V2.
     * This contains the transitive call graph relation.
     * A relation (V1c,V1,V2c,V2) is in the BDD if there is a path in the
     * call graph from the method containing V1 under the context V1c to the
     * method containing V2 under the context V2c.
     */
    BDD accessibleLocations;
    
    /** Context-insensitive variable points-to BDD: V1 x H1.
     * Just cached because it is used often.
     */
    BDD vP_ci;
    
    /** Nodes that are returned from their methods. */
    Collection returned;
    /** Nodes that are thrown from their methods. */
    Collection thrown;
    /** Multi-map between passed parameters and nodes they operate on. */
    InvertibleMultiMap passedParams;

    /** Roots of the SCC graph. */
    Collection sccRoots;

    /** Map from SCC to a BDD of the vars that it accesses
     *  (context-insensitive). */
    Map sccToVars;

    /** Map from SCC to a BDD of the vars that it transitively accesses.
     *  (context-insensitive). */
    Map sccToVarsTransitive;
    
    /** Returns the points-to set for the given variable. */
    public TypedBDD getPointsToSet(int var) {
        BDD result = vP_ci.restrict(V1.ithVar(var));
        return new TypedBDD(result, H1);
    }

    /** Returns the pointed-to-by set for the given heap object. */
    public TypedBDD getPointedToBySet(int heap) {
        BDD result = vP_ci.restrict(H1.ithVar(heap));
        return new TypedBDD(result, V1);
    }
    
    /** Returns the locations that are aliased with the given location. */
    public TypedBDD getAliasedLocations(int var) {
        BDD a = V1.ithVar(var);
        BDD heapObjs = vP_ci.restrict(a);
        a.free();
        TypedBDD result = new TypedBDD(vP_ci.relprod(heapObjs, H1.set()), V1);
        heapObjs.free();
        return result;
    }
    
    /** Returns the set of objects pointed to by BOTH of the given variables. */
    public TypedBDD getAliased(int v1, int v2) {
        BDD a = V1.ithVar(v1);
        BDD heapObjs1 = vP_ci.restrict(a);
        a.free();
        BDD b = V1.ithVar(v2);
        BDD heapObjs2 = vP_ci.restrict(b);
        b.free();
        heapObjs1.andWith(heapObjs2);
        TypedBDD result = new TypedBDD(heapObjs1, H1);
        return result;
    }
    
    /** Returns all heap objects of the given (exact) type. */
    public TypedBDD getAllHeapOfType(jq_Reference type) {
        int j=0;
        BDD result = bdd.zero();
        for (Iterator i=Hmap.iterator(); i.hasNext(); ++j) {
            Node n = (Node) i.next();
            Assert._assert(getHeapIndex(n) == j);
            if (n != null && n.getDeclaredType() == type)
                result.orWith(H1.ithVar(j));
        }
        return new TypedBDD(result, H1);
        /*
        {
            int i = Tmap.get(type);
            BDD a = T2.ithVar(i);
            BDD result = aC.restrict(a);
            a.free();
            return result;
        }
        */
    }
    
    /** Get a context-insensitive version of the points-to information.
     * It achieves this by merging all of the contexts together.  The
     * returned BDD is: V1 x H1.
     */
    public TypedBDD getContextInsensitivePointsTo() {
        return new TypedBDD(vP_ci.id(), V1, H1);
    }

    /** Load call graph from the given file name.
     */
    public void loadCallGraph(String fn) throws IOException {
        cg = new LoadedCallGraph(fn);
    }

    public void numberPaths() {
        // todo: load path numbering instead of renumbering.
        pn = new SCCPathNumbering(new VarPathSelector(VC_BITS));
        Map thread_map = new ThreadRootMap(findThreadRuns(cg));
        Number paths = pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), thread_map);
        System.out.println("Number of paths in call graph="+paths);
    }
    
    public boolean findAliasedParameters2(jq_Method m) {
        Collection s = methodToVariables.getValues(m);
        Collection paramNodes = new LinkedList();
        BDD vars = bdd.zero();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            Object o = j.next();
            if (o instanceof ParamNode || o instanceof FieldNode) {
            //if (!(o instanceof ThrownExceptionNode) && !(o instanceof GlobalNode))
                paramNodes.add(o);
                int v1 = getVariableIndex((Node) o);
                vars.orWith(V1.ithVar(v1));
            }
        }
        BDD mpointsTo = vP.and(vars);
        BDDPairing V1toV2 = bdd.makePair();
        V1toV2.set(new BDDDomain[] {V1c, V1}, new BDDDomain[] {V2c, V2});
        BDD v2_mpointsTo = mpointsTo.replace(V1toV2);
        BDD dom = H1c.set();
        dom.andWith(H1.set());
        BDD result = mpointsTo.relprod(v2_mpointsTo, dom);
        v2_mpointsTo.free();
        dom.free();
        if (false) {
            TypedBDD r = new TypedBDD(result, V1c, V1, V2c, V2);
        }
        return !result.isZero();
    }

    /** Build a map from SCC to the set of local variables. */
    void buildSCCToVarBDD() {
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        sccRoots = new LinkedList();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            SCComponent scc = pn.getSCC(i.next());
            sccRoots.add(scc);
        }
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.postOrder(nav, sccRoots);
        sccToVars = new HashMap();
        sccToVarsTransitive = new HashMap();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (TRACE_ACC_LOC) System.out.println("Visiting SCC"+scc.getId());
            // build the set of local vars in domain V1
            BDD localVars = bdd.zero();
            for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                Object o = j.next();
                if (!(o instanceof jq_Method)) continue;
                jq_Method method = (jq_Method) o;
                Collection method_nodes = methodToVariables.getValues(method);
                if (TRACE_ACC_LOC) System.out.println("Node "+method+" "+method_nodes.size()+" nodes");
                for (Iterator k = method_nodes.iterator(); k.hasNext(); ) {
                    Node n = (Node) k.next();
                    int x = getVariableIndex(n);
                    localVars.orWith(V1.ithVar(x));
                }
            }
            sccToVars.put(scc, localVars);
            BDD transVars = localVars.id();
            for (int j = 0; j < scc.nextLength(); ++j) {
                SCComponent callee = scc.next(j);
                BDD tvars_callee = (BDD) sccToVarsTransitive.get(callee);
                transVars.orWith(tvars_callee.id());
            }
            sccToVarsTransitive.put(scc, transVars);
        }
    }

    /** Build a map from SCC to the set of local variables. */
    void buildCallGraphRelation() {
        if (VC_BITS <= 1) return;
        
        BDDPairing V1ToV2 = bdd.makePair(V1, V2);
        
        System.out.print("Building call graph relation...");
        long time = System.currentTimeMillis();
        
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        List sccroots = new LinkedList();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            SCComponent scc = pn.getSCC(i.next());
            sccroots.add(scc);
        }
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.postOrder(nav, sccroots);
        Map sccToVars = new HashMap();
        callGraphRelation = bdd.zero();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (TRACE_CG_RELATION) System.out.println("Visiting SCC"+scc.getId());
            // build the set of local vars in domain V1
            BDD localVars = bdd.zero();
            boolean anyMethods = false;
            for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                Object o = j.next();
                if (!(o instanceof jq_Method)) {
                    continue;
                }
                anyMethods = true;
                jq_Method method = (jq_Method) o;
                Collection method_nodes = methodToVariables.getValues(method);
                if (TRACE_CG_RELATION) System.out.println("Node "+method+" "+method_nodes.size()+" nodes");
                for (Iterator k = method_nodes.iterator(); k.hasNext(); ) {
                    Node n = (Node) k.next();
                    int x = getVariableIndex(n);
                    localVars.orWith(V1.ithVar(x));
                }
            }
            if (!anyMethods) continue;
            sccToVars.put(scc, localVars.replace(V1ToV2));
            
            Range r1 = pn.getSCCRange(scc);
            if (TRACE_CG_RELATION) System.out.println("Local Vars="+localVars.toStringWithDomains()+" Context Range="+r1);
            BDD b = V1c.varRange(r1.low.longValue(), r1.high.longValue());
            localVars.andWith(b);
            
            BDD finalMap = bdd.zero();
            for (int j = 0; j < scc.nextLength(); ++j) {
                SCComponent call = scc.next(j);
                if (TRACE_CG_RELATION) System.out.println("SCC"+scc.getId()+" -> SCC"+call.getId());
                Collection m;
                if (sccToVars.get(call) == null) {
                    m = Arrays.asList(call.next());
                } else {
                    m = Collections.singleton(call);
                    call = scc;
                }
                for (Iterator n = m.iterator(); n.hasNext(); ) {
                    SCComponent callee = (SCComponent) n.next();
                    Collection edges = pn.getSCCEdges(call, callee);
                    if (TRACE_CG_RELATION) System.out.println("SCC"+call.getId()+" -> SCC"+callee.getId()+": "+edges.size()+" edges");
                    BDD contextVars_callee = (BDD) sccToVars.get(callee);
                    Assert._assert(contextVars_callee != null);
                    // build a map to translate callee path numbers into caller path numbers
                    BDD contextMap = bdd.zero();
                    for (Iterator k = edges.iterator(); k.hasNext(); ) {
                        Pair e = (Pair) k.next();
                        Range r2 = pn.getEdge(e);
                        if (TRACE_CG_RELATION) {
                            System.out.println("Caller vars="+((BDD)sccToVars.get(scc)).toStringWithDomains()+"\n"+
                                               " contexts "+r1+"\n"+
                                               " match callee vars "+contextVars_callee.toStringWithDomains()+"\n"+
                                               " contexts "+r2+" Edge: "+e);
                        }
                        BDD b2 = PA.buildContextMap(V1c, PathNumbering.toBigInt(r1.low), PathNumbering.toBigInt(r1.high),
                                                    V2c, PathNumbering.toBigInt(r2.low), PathNumbering.toBigInt(r2.high));
                        contextMap.orWith(b2);
                    }
                    contextMap.andWith(contextVars_callee.id());
                    //if (TRACE_CG_RELATION) System.out.println("Context map="+contextMap.toStringWithDomains());
                    finalMap.orWith(localVars.and(contextMap));
                    contextMap.free();
                }
            }
            //if (TRACE_CG_RELATION) System.out.println("Final map for SCC"+scc.getId()+": "+finalMap.toStringWithDomains());
            callGraphRelation.orWith(finalMap);
        }
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds, "+callGraphRelation.nodeCount()+" nodes)");
        for (Iterator i = sccToVars.values().iterator(); i.hasNext(); ) {
            BDD b = (BDD) i.next();
            b.free();
        }
    }
    
    public TypedBDD getAccessedLocationsCI(jq_Method m) {
        if (sccToVarsTransitive == null) buildSCCToVarBDD();
        SCComponent scc = pn.getSCC(m);
        BDD b = (BDD) sccToVarsTransitive.get(scc);
        return new TypedBDD(b, V1);
    }
    
    public TypedBDD getAccessedLocationsCS(jq_Method m) {
        if (accessibleLocations != null) {
            BDDPairing V2ToV1 = bdd.makePair();
            V2ToV1.set(new BDDDomain[] {V2c, V2}, new BDDDomain[] {V1c, V1} );
            
            Collection method_nodes = methodToVariables.getValues(m);
            Node n = (Node) method_nodes.iterator().next();
            BDD b = V1.ithVar(getVariableIndex(n));
            b.andWith(accessibleLocations.id());
            BDD c = b.exist(V1c.set());
            b.free();
            c.replaceWith(V2ToV1);
            return new TypedBDD(c, V1c, V1);
        }
        
        // TODO!
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.reversePostOrder(nav, sccRoots);
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (TRACE_ACC_LOC) System.out.println("Visiting SCC"+scc.getId());
        }
        return null;
    }
    
    
    
    void buildAccessibleLocations() {
        System.out.print("Building transitive call graph relation...");
        long time = System.currentTimeMillis();
        BDDPairing V1ToV2 = bdd.makePair();
        V1ToV2.set(new BDDDomain[] {V1c, V1}, new BDDDomain[] {V2c, V2} );
        BDDPairing V3cToV1c = bdd.makePair(V3c, V1c);
        BDDPairing V2ToV1 = bdd.makePair(V2, V1);
        BDD V1cset = V1c.set();
        
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        List sccroots = new LinkedList();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            SCComponent scc = pn.getSCC(i.next());
            sccroots.add(scc);
        }
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.postOrder(nav, sccroots);
        Map sccToVars = new HashMap();
        accessibleLocations = bdd.zero();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (TRACE_ACC_LOC) System.out.println("Visiting SCC"+scc.getId());
            // build the set of local vars in domain V2
            BDD localVars = bdd.zero();
            for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                Object o = j.next();
                if (!(o instanceof jq_Method)) continue;
                jq_Method method = (jq_Method) o;
                if (TRACE_ACC_LOC) System.out.println("Node "+method);
                Collection method_nodes = methodToVariables.getValues(method);
                for (Iterator k = method_nodes.iterator(); k.hasNext(); ) {
                    Node n = (Node) k.next();
                    int x = getVariableIndex(n);
                    localVars.orWith(V2.ithVar(x));
                }
            }
            Range r1 = pn.getSCCRange(scc);
            if (TRACE_ACC_LOC) System.out.println("Local Vars="+localVars.toStringWithDomains()+" Context Range="+r1);
            BDD b = PA.buildContextMap(V1c, PathNumbering.toBigInt(r1.low), PathNumbering.toBigInt(r1.high),
                                       V2c, PathNumbering.toBigInt(r1.low), PathNumbering.toBigInt(r1.high));
            BDD contextVars = b.and(localVars);
            b.free();
            if (TRACE_ACC_LOC) System.out.println("With context="+contextVars.toStringWithDomains());
            for (int j = 0; j < scc.nextLength(); ++j) {
                SCComponent callee = scc.next(j);
                Collection edges = pn.getSCCEdges(scc, callee);
                if (TRACE_ACC_LOC) System.out.println("SCC"+scc.getId()+" -> SCC"+callee.getId()+": "+edges.size()+" edges");
                BDD contextVars_callee = (BDD) sccToVars.get(callee);
                // build a map to translate callee path numbers into caller path numbers
                BDD contextMap = bdd.zero();
                for (Iterator k = edges.iterator(); k.hasNext(); ) {
                    Pair e = (Pair) k.next();
                    Range r2 = pn.getEdge(e);
                    if (TRACE_ACC_LOC) System.out.println("Edge="+e+" Caller range="+r1+" Callee range="+r2);
                    BDD b2 = PA.buildContextMap(V1c, PathNumbering.toBigInt(r2.low), PathNumbering.toBigInt(r2.high),
                                                V3c, PathNumbering.toBigInt(r1.low), PathNumbering.toBigInt(r1.high));
                    contextMap.orWith(b2);
                }
                if (TRACE_ACC_LOC) System.out.println("Context map="+contextMap.toStringWithDomains());
                // relprod to translate callee path numbers into caller path numbers
                BDD r = contextVars_callee.relprod(contextMap, V1cset);
                contextMap.free();
                r.replaceWith(V3cToV1c);
                if (TRACE_ACC_LOC) System.out.println("Translated="+r.toStringWithDomains());
                contextVars.orWith(r);
            }
            if (TRACE_ACC_LOC) System.out.println("Final context vars for SCC"+scc.getId()+": "+contextVars.toStringWithDomains());
            sccToVars.put(scc, contextVars);
            BDD al = contextVars.id();
            al.andWith(localVars.replace(V2ToV1));
            if (TRACE_ACC_LOC) System.out.println("Final location map for SCC"+scc.getId()+": "+al.toStringWithDomains());
            accessibleLocations.orWith(al);
        }
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds, "+accessibleLocations.nodeCount()+" nodes)");
        for (Iterator i = sccToVars.values().iterator(); i.hasNext(); ) {
            BDD b = (BDD) i.next();
            b.free();
        }
    }

    public int countMod() {
        int mod = 0, exception = 0, total = 0;
        for (Iterator i = methodToVariables.keySet().iterator(); i.hasNext(); ) {
            int n_mods = 0;
            jq_Method m = (jq_Method) i.next();
            if (m != null) {
                System.out.print("Method "+m.getName()+"(): ");
            } else {
                System.out.print("Method null: ");
            }
            Collection c = methodToVariables.getValues(m);
            boolean only_exception = true;
            if (c.isEmpty()) {
            } else {
                TypedBDD res = findMod(m);
                n_mods = (int) res.satCount();
                
                BDD f = res.bdd.exist(H1c.set().and(H1.set()));
                TypedBDD f2 = new TypedBDD(f, F);
                for (Iterator j = f2.iterator(); j.hasNext(); ) {
                    int z = ((Integer) j.next()).intValue();
                    jq_Field field = getField(z);
                    if (field == null) {
                        only_exception = false;
                    } else {
                        jq_Class cl = field.getDeclaringClass();
                        cl.prepare(); PrimordialClassLoader.getJavaLangThrowable().prepare();
                        if (!cl.isSubtypeOf(PrimordialClassLoader.getJavaLangThrowable())) {
                            only_exception = false;
                        }
                    }
                }
                if (only_exception && n_mods != 0)
                    System.out.print(" (only exceptions) ");
            }
            System.out.println(n_mods+" mods");
            if (n_mods != 0) {
                ++mod;
                if (only_exception) ++exception;
            }
            ++total;
        }
        System.out.println("Mod: "+mod);
        System.out.println("Exception: "+exception);
        System.out.println("Total: "+total);
        return mod;
    }
    
    public TypedBDD findMod(Object method) {
        if (method == null) {
            return new TypedBDD(bdd.zero(), V1c, V1, H1c, H1, F);
        }
        BDD V1set = V1c.set(); V1set.andWith(V1.set());
        BDD H1Fset = H1c.set(); H1Fset.andWith(H1.set()); H1Fset.andWith(F.set());
        BDDPairing V2ToV1 = bdd.makePair();
        V2ToV1.set(new BDDDomain[] {V2c, V2}, new BDDDomain[] {V1c, V1});
        
        SCComponent sc = pn.getSCC(method);
        
        if (accessibleLocations != null) {
            BDD bdd1 = (BDD) sccToVars.get(sc);
            //BDD a = bdd1.getDomains();
            BDD a = bdd.one();
            a.andWith(V1c.set()); a.andWith(V1.set());
            BDD b = accessibleLocations.relprod(bdd1, a); // V2c x V2
            BDD d = S.exist(V1set); // F x V2c x V2
            d.andWith(b);
            d.replaceWith(V2ToV1); // V1c x V1 x F
            BDD e = vP.and(d); // V1c x V1 x H1c x H1 x F
            return new TypedBDD(e, V1c, V1, H1c, H1, F);
        }
        
        BDD globalStores = S.exist(V1set);
        globalStores.replaceWith(V2ToV1);
        globalStores.andWith(vP.id());
        
        BDD relation = bdd.zero();
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.reversePostOrder(nav, sc);
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            BDD locations = (BDD) this.sccToVars.get(scc);
            BDD localStores = locations.and(globalStores);
            if (TRACE_MOD) {
                System.out.println("Visiting "+scc);
                System.out.println("Mods: "+new TypedBDD(localStores, V1c, V1, H1c, H1, F));
            }
            relation.orWith(localStores);
        }
        
        if (callGraphRelation == null)
            buildCallGraphRelation();
        
        BDD newRelations = relation.id();
        for (int i=0; ; ++i) {
            if (TRACE_MOD) {
                System.out.println("Iteration "+i+": "+newRelations.nodeCount()+" new");
            }
            BDD relation2 = newRelations.relprod(callGraphRelation, V1set);
            newRelations.free();
            relation2.replaceWith(V2ToV1);
            relation2.applyWith(relation.id(), BDDFactory.diff);
            boolean done = relation2.isZero();
            if (done) {
                relation2.free();
                break;
            }
            newRelations = relation2;
            relation.orWith(newRelations.id());
        }
        
        return new TypedBDD(relation, V1c, V1, H1c, H1, F);
    }

    public TypedBDD findRef(TypedBDD bdd1) {
        BDD V1set = V1c.set(); V1set.andWith(V1.set());
        BDDPairing V2ToV1 = bdd.makePair();
        V2ToV1.set(new BDDDomain[] {V2c, V2}, new BDDDomain[] {V1c, V1});
        
        BDD a = bdd1.getDomains();
        a.andWith(V1c.set()); a.andWith(V1.set());
        BDD b = accessibleLocations.relprod(bdd1.bdd, a); // V2c x V2
        b.replaceWith(V2ToV1); // V1c x V1
        BDD e = vP.relprod(b, V1set); // H1c x H1
        return new TypedBDD(e, H1c, H1);
    }
    
    Map buildTransitiveAccessedLocations() {
        SCCTopSortedGraph sccgraph = pn.getSCCGraph();
        List sccroots = new LinkedList();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            SCComponent scc = pn.getSCC(i.next());
            sccroots.add(scc);
        }
        Navigator nav = sccgraph.getNavigator();
        List sccs = Traversals.postOrder(nav, sccroots);
        Map sccToVarBDD = new HashMap();
        for (Iterator i = sccs.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            BDD vars = bdd.zero();
            for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                jq_Method method = (jq_Method) j.next();
                Collection method_nodes = methodToVariables.getValues(method);
                for (Iterator k = method_nodes.iterator(); k.hasNext(); ) {
                    Node n = (Node) k.next();
                    int x = getVariableIndex(n);
                    vars.orWith(V1.ithVar(x));
                }
            }
            for (int j = 0; j < scc.nextLength(); ++j) {
                SCComponent scc2 = scc.next(j);
                BDD vars2 = (BDD) sccToVarBDD.get(scc2);
                vars.orWith(vars2.id());
            }
            sccToVarBDD.put(scc, vars);
        }
        return sccToVarBDD;
    }

    public TypedBDD ci_modRef(jq_Method m) {
        Map sccToVarBDD = buildTransitiveAccessedLocations();
        SCComponent scc = pn.getSCC(m);
        BDD b = (BDD) sccToVarBDD.get(scc);
        BDD dom = V1c.set();
        dom.andWith(V1.set());
        BDD c = b.relprod(vP, dom);
        return new TypedBDD(c, H1c, H1);
    }

    public TypedBDD findEquivalentObjects_fi() {
        BDDPairing H2toH3 = bdd.makePair();
        H2toH3.set(new BDDDomain[] { H2c, H2 }, new BDDDomain[] { H3c, H3 } );
        BDDPairing H3toH1 = bdd.makePair();
        H3toH1.set(new BDDDomain[] { H3c, H3 }, new BDDDomain[] { H1c, H1 } );
        BDD H1set = H1c.set().and(H1.set());
        
        BDD heapPt = hP.exist(F.set());
        BDD heapPt2 = heapPt.replace(H2toH3);
        
        BDD heapPt3 = heapPt.applyAll(heapPt2, BDDFactory.biimp, H1set);
        heapPt.free(); heapPt2.free();
        
        heapPt3.replaceWith(H3toH1);
        
        BDD pointedTo = hP.exist(H1set.and(F.set()));
        heapPt3.andWith(pointedTo);
        return new TypedBDD(heapPt3, H1c, H1, H2c, H2);
    }

    public TypedBDD findEquivalentObjects_fs() {
        BDDPairing H1toH2 = bdd.makePair();
        H1toH2.set(new BDDDomain[] { H1c, H1 }, new BDDDomain[] { H2c, H2 } );
        BDDPairing H2toH1 = bdd.makePair();
        H2toH1.set(new BDDDomain[] { H2c, H2 }, new BDDDomain[] { H1c, H1 } );
        BDDPairing H2toH3 = bdd.makePair();
        H2toH3.set(new BDDDomain[] { H2c, H2 }, new BDDDomain[] { H3c, H3 } );
        BDDPairing H3toH1 = bdd.makePair();
        H3toH1.set(new BDDDomain[] { H3c, H3 }, new BDDDomain[] { H1c, H1 } );
        BDD H1set = H1c.set().and(H1.set());
        BDD H1H2set = H1set.and(H2c.set().and(H2.set()));
        BDD H1andFset = H1set.and(F.set());
        
        BDD fieldPt2 = hP.replace(H2toH3);
        BDD fieldPt3 = hP.applyAll(fieldPt2, BDDFactory.biimp, H1andFset);
        fieldPt2.free();
        
        fieldPt3.replaceWith(H3toH1);
        
        BDD pointedTo = hP.exist(H1andFset);
        BDD filter = H1.buildEquals(H2);
        //filter.andWith(H1c.buildEquals(H2c));
        fieldPt3.andWith(filter.not());
        filter.free();
        fieldPt3.andWith(pointedTo);
        
        BDD iter = fieldPt3.id();
        int num = 0;
        while (!iter.isZero()) {
            ++num;
            BDD sol = iter.satOne(H1H2set, false);
            BDD sol_h2 = sol.exist(H1set);
            sol.free();
            BDD sol3 = iter.restrict(sol_h2);
            sol3.orWith(sol_h2.replace(H2toH1));
            sol_h2.free();
            System.out.println("EC "+num+":\n"+new TypedBDD(sol3, H1c, H1).toString());
            sol3.andWith(sol3.replace(H1toH2));
            iter.applyWith(sol3, BDDFactory.diff);
            //try { System.in.read(); } catch (IOException x) {}
        }
        iter.free();
        System.out.println("There are "+num+" equivalence classes.");
        
        return new TypedBDD(fieldPt3, H1c, H1, H2c, H2);
    }
    
    public boolean findAliasedParameters(jq_Method m) {
        Collection s = methodToVariables.getValues(m);
        Collection paramNodes = new LinkedList();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            Object o = j.next();
            if (o instanceof ParamNode || o instanceof FieldNode)
            //if (!(o instanceof ThrownExceptionNode) && !(o instanceof GlobalNode))
                paramNodes.add(o);
        }
        boolean hasAliased = false;
        int n = 1;
        for (Iterator j = paramNodes.iterator(); j.hasNext(); ) {
            Node p1 = (Node) j.next();
            int v1 = getVariableIndex(p1);
            Iterator k = paramNodes.iterator();
            for (int a = 0; a < n; ++a) k.next();
            while (k.hasNext()) {
                Node p2 = (Node) k.next();
                Assert._assert(p1 != p2);
                int v2 = getVariableIndex(p2);
                TypedBDD result = getAliased(v1, v2);
                for (Iterator l = result.iterator(); l.hasNext(); ) {
                    int h = ((Integer)l.next()).intValue();
                    BDD relation = V1.ithVar(v1);
                    relation.orWith(V1.ithVar(v2));
                    relation.andWith(H1.ithVar(h));
                    BDD c_result = vP.relprod(relation, V1.set().and(H1.set()));
                    relation.free();
                    if (!c_result.isZero()) {
                        //System.out.println("Aliased: "+m+" "+p1+" "+p2+":");
                        //System.out.println(result);
                        //System.out.println("Under contexts: "+c_result.toStringWithDomains());
                        hasAliased = true;
                    }
                    c_result.free();
                }
                result.free();
            }
            ++n;
        }
        return hasAliased;
    }
    
    public void findAliasedParameters() {
        int noAlias = 0, hasAlias = 0;
        for (Iterator i = methodToVariables.keySet().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            boolean hasAliased = findAliasedParameters(m);
            if (hasAliased) hasAlias++;
            else noAlias++;
        }
        System.out.println("No aliased parameters: "+noAlias);
        System.out.println("Has aliased parameters: "+hasAlias);
    }

    public TypedBDD getReachableObjects(int heap) {
        BDDPairing H1H2toH2H3 = bdd.makePair();
        H1H2toH2H3.set(new BDDDomain[] { H1c, H1, H2c, H2 }, new BDDDomain[] { H2c, H2, H3c, H3 } );
        BDDPairing H3toH2 = bdd.makePair();
        H3toH2.set(new BDDDomain[] { H3c, H3 }, new BDDDomain[] { H2c, H2 } );
        BDDPairing H2toH1 = bdd.makePair();
        H2toH1.set(new BDDDomain[] { H2c, H2 }, new BDDDomain[] { H1c, H1 } );
        BDD H1set = H1c.set();
        H1set.andWith(H1.set());
        BDD H2set = H2c.set();
        H2set.andWith(H2.set());
        
        BDD heapPt12 = hP.exist(F.set().and(H1c.set()).and(H2c.set()));
        BDD heapPt23 = heapPt12.replace(H1H2toH2H3);
        BDD oldReachable = heapPt12.and(H1.ithVar(heap));
        BDD reachable = oldReachable.id();
        int count = 0;
        for (;;) {
            BDD newReachable = reachable.relprod(heapPt23, H2set);
            newReachable.replaceWith(H3toH2);
            reachable.orWith(newReachable);
            boolean done = reachable.equals(oldReachable);
            oldReachable.free();
            if (done) break;
            oldReachable = reachable.id();
            ++count;
        }
        System.out.println("Depth: "+count);
        BDD result = reachable.exist(H1set);
        result.replaceWith(H2toH1);
        return new TypedBDD(result, H1c, H1);
    }

    static final boolean FILTER_NULL = true;

    public static final byte DOT = 1;
    public static final byte HYPVIEW = 2;
    public static byte format = DOT;

    public void dumpObjectConnectivityGraph(int heapnum, BufferedWriter out) throws IOException {
        BDD context = H1c.set();
        context.andWith(H2c.set());
        BDD ci_fieldPt = hP.exist(context);

        TypedBDD reach = getReachableObjects(heapnum);
        System.out.println(reach);
        BDD reachable = reach.bdd;
        reachable.orWith(H1.ithVar(heapnum));

        out.write("digraph \"ObjectConnectivity\" {\n");
        BDD iter;
        iter = reachable.id();
        while (!iter.isZero()) {
            BDD s = iter.satOne(H1.set(), false);
            BigInteger[] val = s.scanAllVar();
            int target_i = val[H1.getIndex()].intValue();
            s.andWith(H1.ithVar(target_i));
            HeapObject h = (HeapObject) getHeapNode(target_i);
            ProgramLocation l = (h != null) ? h.getLocation() : null;
            jq_Type t = null;
            if (h != null) {
                t = h.getDeclaredType();
                if (FILTER_NULL && t == null) {
                    iter.applyWith(s, BDDFactory.diff);
                    continue;
                }
            }
            String name = null;
            if (t != null) name = t.shortName();
            int j = Hmap.get(h);
            out.write("n"+j+" [label=\""+name+"\"];\n");
            iter.applyWith(s, BDDFactory.diff);
        }
        iter.free();
        
        iter = reachable.id();
        while (!iter.isZero()) {
            BDD s = iter.satOne(H1.set(), false);
            BigInteger[] val = s.scanAllVar();
            int target_i = val[H1.getIndex()].intValue();
            s.andWith(H1.ithVar(target_i));
            HeapObject h = (HeapObject) getHeapNode(target_i);
            jq_Type t = null;
            if (h != null) {
                t = h.getDeclaredType();
                if (FILTER_NULL && t == null) {
                    iter.applyWith(s, BDDFactory.diff);
                    continue;
                }
            }
            BDD pt = ci_fieldPt.restrict(H1.ithVar(target_i));
            while (!pt.isZero()) {
                BDD s2 = pt.satOne(H2.set().and(F.set()), false);
                BigInteger[] val2 = s2.scanAllVar();
                int target2_i = val2[H2.getIndex()].intValue();
                s2.andWith(H2.ithVar(target2_i));
                HeapObject target = (HeapObject) getHeapNode(target2_i);
                if (FILTER_NULL && target != null && target.getDeclaredType() == null) {
                    pt.applyWith(s2, BDDFactory.diff);
                    continue;
                }
                int fn = val2[F.getIndex()].intValue();
                jq_Field f = getField(fn);
                String fieldName = "[]";
                if (f != null) fieldName = f.getName().toString();
                out.write("n"+target_i+
                               " -> n"+target2_i+
                               " [label=\""+fieldName+"\"];\n");
                pt.applyWith(s2, BDDFactory.diff);
            }
            iter.applyWith(s, BDDFactory.diff);
        }
        iter.free();
        out.write("}\n");
    }
    
    public void dumpObjectConnectivityGraph(BufferedWriter out) throws IOException {
        BDD context = H1c.set();
        context.andWith(H2c.set());
        BDD ci_fieldPt = hP.exist(context);

        out.write("digraph \"ObjectConnectivity\" {\n");
        int j = 0;
        for (Iterator i = Hmap.iterator(); i.hasNext(); ++j) {
            HeapObject h = (HeapObject) i.next();
            Assert._assert(j == Hmap.get(h));
            jq_Type t = null;
            if (h != null) {
                t = h.getDeclaredType();
                if (FILTER_NULL && t == null) continue;
            }
            String name = null;
            if (t != null) name = t.shortName();
            out.write("n"+j+" [label=\""+name+"\"];\n");
        }
        
        j = 0;
        for (Iterator i = Hmap.iterator(); i.hasNext(); ++j) {
            HeapObject h = (HeapObject) i.next();
            Assert._assert(j == Hmap.get(h));
            if (FILTER_NULL && h != null && h.getDeclaredType() == null)
                continue;
            BDD pt = ci_fieldPt.restrict(H1.ithVar(j));
            while (!pt.isZero()) {
                BDD s = pt.satOne(H2.set().and(F.set()), false);
                BigInteger[] val = s.scanAllVar();
                int target_i = val[H2.getIndex()].intValue();
                HeapObject target = (HeapObject) getHeapNode(target_i);
                s.andWith(H2.ithVar(target_i));
                if (FILTER_NULL && h != null && h.getDeclaredType() == null) {
                    continue;
                }
                int fn = val[F.getIndex()].intValue();
                jq_Field f = getField(fn);
                String fieldName = "[]";
                if (f != null) fieldName = f.getName().toString();
                out.write("n"+j+
                               " -> n"+target_i+
                               " [label=\""+fieldName+"\"];\n");
                pt.applyWith(s, BDDFactory.diff);
            }
        }
        out.write("}\n");
    }

    public static boolean TRACE_ESCAPE = false;
    public static boolean TRACE_ACC_LOC = false;
    public static boolean TRACE_CG_RELATION = false;
    public static boolean TRACE_MOD = false;

    public Collection getTargetMethods(ProgramLocation callSite) {
        return cg.getTargetMethods(mapCall(callSite));
    }
    
    public boolean isReturned(Node n) {
        return returned.contains(n);
    }
    
    public boolean isThrown(Node n) {
        return thrown.contains(n);
    }
    
    public TypedBDD escapeAnalysis() {
        
        BDD escapingLocations = bdd.zero();
        
        List order = Traversals.postOrder(cg.getNavigator(), cg.getRoots());
        Map methodToVarBDD = new HashMap();
        for (Iterator i = order.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            BDD m_vars;
            SCComponent scc = (SCComponent) pn.getSCC(m);
            if (scc.isLoop()) {
                m_vars = bdd.zero();
            } else {
                //Collection m_nodes = methodToVariables.getValues(m);
                m_vars = bdd.zero();
                for (Iterator j = cg.getCallees(m).iterator(); j.hasNext(); ) {
                    jq_Method callee = (jq_Method) j.next();
                    BDD m_vars2 = (BDD) methodToVarBDD.get(callee);
                    if (m_vars2 == null) continue;
                    m_vars.orWith(m_vars2.id());
                }
            }
            methodToVarBDD.put(m, m_vars);
            Collection m_nodes = methodToVariables.getValues(m);
            HashMap concreteNodes = new HashMap();
            for (Iterator j = m_nodes.iterator(); j.hasNext(); ) {
                Node o = (Node) j.next();
                if (o instanceof ConcreteTypeNode) {
                    ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                    ProgramLocation pl = ctn.getLocation();
                    pl = mapCall(pl);
                    concreteNodes.put(pl, ctn);
                }
                boolean bad = false;
                if (o.getEscapes()) {
                    if (TRACE_ESCAPE) System.out.println(o+" escapes, bad");
                    bad = true;
                } else if (cg.getRoots().contains(m) && isThrown(o)) {
                    if (TRACE_ESCAPE) System.out.println(o+" is thrown from root set, bad");
                    bad = true;
                } else {
                    Set passedParams = o.getPassedParameters();
                    if (passedParams != null) {
                        outer:
                        for (Iterator k = passedParams.iterator(); k.hasNext(); ) {
                            PassedParameter pp = (PassedParameter) k.next();
                            ProgramLocation mc = pp.getCall();
                            for (Iterator a = getTargetMethods(mc).iterator(); a.hasNext(); ) {
                                jq_Method m2 = (jq_Method) a.next();
                                if (m2.getBytecode() == null) {
                                    if (TRACE_ESCAPE) System.out.println(o+" is passed into a native method, bad");
                                    bad = true;
                                    break outer;
                                }
                            }
                        }
                    }
                }
                if (!bad) {
                    int v_i = getVariableIndex(o);
                    m_vars.orWith(V1.ithVar(v_i));
                    //if (TRACE_ESCAPE) System.out.println("Var "+v_i+" is good: "+m_vars.toStringWithDomains());
                }
            }
            if (TRACE_ESCAPE) System.out.println("Non-escaping locations for "+m+" = "+m_vars.toStringWithDomains());
            ControlFlowGraph cfg = CodeCache.getCode(m);
            boolean trivial = false;
            for (QuadIterator j = new QuadIterator(cfg); j.hasNext(); ) {
                Quad q = j.nextQuad();
                if (q.getOperator() instanceof Operator.New ||
                    q.getOperator() instanceof Operator.NewArray) {
                    ProgramLocation pl = new QuadProgramLocation(m, q);
                    pl = mapCall(pl);
                    ConcreteTypeNode ctn = (ConcreteTypeNode) concreteNodes.get(pl);
                    if (ctn == null) {
                        //trivial = true;
                        trivial = q.getOperator() instanceof Operator.New;
                        System.out.println(cfg.getMethod()+": "+q+" trivially doesn't escape.");
                    } else {
                        int v_i = getVariableIndex(ctn);
                        BDD h = vP_ci.restrict(V1.ithVar(v_i));
                        Assert._assert(h.satCount(H1.set()) == 1.0);
                        if (TRACE_ESCAPE) {
                            System.out.println("Heap location: "+h.toStringWithDomains()+" = "+ctn);
                            System.out.println("Pointed to by: "+vP_ci.restrict(h).toStringWithDomains());
                        }
                        h.andWith(m_vars.not());
                        escapingLocations.orWith(h);
                    }
                }
            }
            if (trivial) {
                System.out.println(cfg.fullDump());
            }
        }
        for (Iterator i = methodToVarBDD.values().iterator(); i.hasNext(); ) {
            BDD b = (BDD) i.next();
            b.free();
        }

        BDD escapingHeap = escapingLocations.relprod(vP_ci, V1.set());
        escapingLocations.free();
        System.out.println("Escaping heap: "+escapingHeap.satCount(H1.set()));
        //System.out.println("Escaping heap: "+escapingHeap.toStringWithDomains());
        BDD capturedHeap = escapingHeap.not();
        capturedHeap.andWith(H1.varRange(0, Hmap.size()-1));
        System.out.println("Captured heap: "+capturedHeap.satCount(H1.set()));
        
        int capturedSites = 0;
        int escapedSites = 0;
        long capturedSize = 0L;
        long escapedSize = 0L;
        
        for (Iterator i = Hmap.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            int ndex = getHeapIndex(n);
            if (n instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) n;
                jq_Reference t = (jq_Reference) ctn.getDeclaredType();
                if (t == null) continue;
                int size = 0;
                t.prepare();
                if (t instanceof jq_Class)
                    size = ((jq_Class) t).getInstanceSize();
                else
                    continue;
                BDD bdd = capturedHeap.and(H1.ithVar(ndex));
                if (bdd.isZero()) {
                    // not captured.
                    if (TRACE_ESCAPE) System.out.println("Escaped: "+n);
                    escapedSites ++;
                    escapedSize += size;
                } else {
                    // captured.
                    if (TRACE_ESCAPE) System.out.println("Captured: "+n);
                    capturedSites ++;
                    capturedSize += size;
                }
            }
        }
        System.out.println("Captured sites = "+capturedSites+", "+capturedSize+" bytes.");
        System.out.println("Escaped sites = "+escapedSites+", "+escapedSize+" bytes.");
        return new TypedBDD(capturedHeap, H1);
    }

    static Map thread_runs = new HashMap();
    public Map findThreadRuns(CallGraph cg) {
        thread_runs.clear();
        
        PrimordialClassLoader.getJavaLangThread().prepare();
        jq_Class jlt = PrimordialClassLoader.getJavaLangThread();
        jq_Class jlr = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;");
        jlr.prepare();
        
        for (Iterator i = Hmap.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            jq_Reference type = n.getDeclaredType();
            if (type != null) {
                type.prepare();
                if (type.isSubtypeOf(jlt) ||
                    type.isSubtypeOf(jlr)) {
                    jq_Method rm = type.getVirtualMethod(PA.run_method);
                    Set s = (Set) thread_runs.get(rm);
                    if (s == null) thread_runs.put(rm, s = new HashSet());
                    s.add(n);
                }
            }
        }
        return thread_runs;
    }

    /** Load points-to results from the given file name prefix.
     */
    public void load(String fn) throws IOException {
        BufferedReader di = null;
        try {
            di = new BufferedReader(new FileReader(fn+".config"));
            readConfig(di);
        } finally {
            if (di != null) di.close();
        }
        
        System.out.print("Initializing...");
        initialize();
        System.out.println("done.");
        
        System.out.print("Loading BDDs...");
        this.vP = bdd.load(fn+".vP");
        System.out.print("vP "+this.vP.nodeCount()+" nodes, ");
        this.hP = bdd.load(fn+".hP");
        System.out.print("hP "+this.hP.nodeCount()+" nodes, ");
        this.S = bdd.load(fn+".S");
        System.out.print("S "+this.S.nodeCount()+" nodes, ");
        this.L = bdd.load(fn+".L");
        System.out.print("L "+this.L.nodeCount()+" nodes, ");
        this.A = bdd.load(fn+".A");
        System.out.print("A "+this.A.nodeCount()+" nodes, ");
        this.mV = bdd.load(fn+".mV");
        System.out.print("mV "+this.mV.nodeCount()+" nodes, ");
        this.mI = bdd.load(fn+".mI");
        System.out.print("mI "+this.mI.nodeCount()+" nodes, ");
        this.actual = bdd.load(fn+".actual");
        System.out.print("actual "+this.actual.nodeCount()+" nodes, ");
        this.IE = bdd.load(fn+".IE");
        System.out.print("IE "+this.IE.nodeCount()+" nodes, ");
        if (new File(fn+".IEc").exists()) {
            this.IEc = bdd.load(fn+".IEc");
            System.out.print("IEc "+this.IEc.nodeCount()+" nodes, ");
        }
        System.out.println("done.");
        
        this.returned = new HashSet();
        this.thrown = new HashSet();
        this.passedParams = new GenericInvertibleMultiMap();
        
        System.out.print("Loading maps...");
        di = new BufferedReader(new FileReader(fn+".Vmap"));
        Vmap = IndexMap.load("Variable", di);
        di.close();
        
        di = new BufferedReader(new FileReader(fn+".Imap"));
        Imap = IndexMap.load("Invoke", di);
        di.close();
        
        di = new BufferedReader(new FileReader(fn+".Hmap"));
        Hmap = IndexMap.load("Heap", di);
        di.close();
        
        di = new BufferedReader(new FileReader(fn+".Mmap"));
        Mmap = IndexMap.load("Method", di);
        di.close();
        
        di = new BufferedReader(new FileReader(fn+".Fmap"));
        Fmap = IndexMap.load("Field", di);
        di.close();
        System.out.println("done.");

    }
    
    private void sanityCheck() {
        boolean bad = false;
        for (int i = 0; i < Vmap.size(); ++i) {
            Node n = (Node) Vmap.get(i);
            if (n instanceof ConcreteTypeNode && n.getDeclaredType() != null) {
                BDD b = vP.restrict(V1.ithVar(i));
                if (b.satCount(H1.set()) != 1.0) {
                    bad = true;
                    break;
                }
                int H_i = b.scanVar(H1).intValue();
                Node n2 = (Node) Hmap.get(H_i);
                if (!(n2 instanceof ConcreteTypeNode)) {
                    bad = true;
                    break;
                }
            }
        }
        if (bad) {
            System.err.println("Something is wrong, vP BDD looks corrupted.");
        }
    }
    
    void reindex(CSPAResults that) {
        if (that.Vmap == null) {
            that.Vmap = this.Vmap;
            that.Imap = this.Imap;
            that.Hmap = this.Hmap;
            that.Mmap = this.Mmap;
            that.Fmap = this.Fmap;
            return;
        }
        
        BDDPairing V3V1 = bdd.makePair(V3, V1);
        BDDPairing V3V2 = bdd.makePair(V3, V2);
        BDDPairing H3H1 = bdd.makePair(H3, H1);
        BDDPairing H3H2 = bdd.makePair(H3, H2);
        BDDPairing F2F = bdd.makePair(F2, F);
        BDD v_mapping13 = reindex(this.Vmap, that.Vmap, V1, V3);
        BDD i_mapping12 = reindex(this.Imap, that.Imap, I, I2);
        BDD h_mapping13 = reindex(this.Hmap, that.Hmap, H1, H3);
        BDD h_mapping23 = h_mapping13.replace(H1toH2);
        BDD m_mapping12 = reindex(this.Mmap, that.Mmap, M, M2);
        BDD f_mapping12 = reindex(this.Fmap, that.Fmap, F, F2);
        
        BDD vP1 = vP.relprod(v_mapping13, V1set);
        vP.free();
        vP1.replaceWith(V3V1);
        BDD vP2 = vP1.relprod(h_mapping13, H1set);
        vP1.free();
        vP2.replaceWith(H3H1);
        vP = vP2;
        
        BDD hP1 = hP.relprod(h_mapping13, H1set);
        hP.free();
        hP1.replaceWith(H3H1);
        BDD hP2 = hP1.relprod(h_mapping23, H2set);
        hP.free();
        hP2.replaceWith(H3H2);
        BDD hP3 = hP2.relprod(f_mapping12, Fset);
        hP2.free();
        hP3.replaceWith(F2F);
        hP = hP3;
        
        this.Vmap = that.Vmap;
        this.Imap = that.Imap;
        this.Hmap = that.Hmap;
        this.Mmap = that.Mmap;
        this.Fmap = that.Fmap;
        
    }
    
    BDD reindex(IndexMap thismap, IndexMap thatmap, BDDDomain d1, BDDDomain d2) {
        BDD mapping = bdd.zero();
        for (int i = 0; i < thismap.size(); ++i) {
            Object o = thismap.get(i);
            int j = thatmap.get(o);
            BDD bdd1 = d1.ithVar(i);
            bdd1.andWith(d2.ithVar(j));
            System.out.println(thismap.toString()+": "+i+" -> "+j);
            mapping.orWith(bdd1);
        }
        return mapping;
    }
    
    private void buildContextInsensitive() {
        BDD context = V1c.set();
        context.andWith(H1c.set());
        this.vP_ci = vP.exist(context);
        context.free();
    }

    String varorder;
    boolean reverseLocal = true;

    int V_BITS=1, I_BITS=1, H_BITS=1, Z_BITS=1, F_BITS=1, T_BITS=1, N_BITS=1, M_BITS=1;
    int VC_BITS=1, HC_BITS=1;

    private void readConfig(BufferedReader in) throws IOException {
        for (;;) {
            String s = in.readLine();
            if (s == null) break;
            int index = s.indexOf('=');
            if (index == -1) index = s.length();
            String s1 = s.substring(0, index);
            String s2 = index < s.length() ? s.substring(index+1) : null;
            if (s1.equals("V")) {
                V_BITS = Integer.parseInt(s2);
            } else if (s1.equals("I")) {
                I_BITS = Integer.parseInt(s2);
            } else if (s1.equals("H")) {
                H_BITS = Integer.parseInt(s2);
            } else if (s1.equals("Z")) {
                Z_BITS = Integer.parseInt(s2);
            } else if (s1.equals("F")) {
                F_BITS = Integer.parseInt(s2);
            } else if (s1.equals("T")) {
                T_BITS = Integer.parseInt(s2);
            } else if (s1.equals("N")) {
                N_BITS = Integer.parseInt(s2);
            } else if (s1.equals("M")) {
                M_BITS = Integer.parseInt(s2);
            } else if (s1.equals("VC")) {
                VC_BITS = Integer.parseInt(s2);
            } else if (s1.equals("HC")) {
                HC_BITS = Integer.parseInt(s2);
            } else if (s1.equals("Order")) {
                varorder = s2;
            } else if (s1.equals("Reverse")) {
                reverseLocal = s2.equals("true");
            } else {
                System.err.println("Unknown config option "+s);
            }
        }
    }
    
    BDDDomain makeDomain(String name, int bits) {
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDDPairing V1cV2ctoV2cV1c, V1ctoV2c;
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1H1set, IMset, H1Fset, H2Fset, H1FH2set, T2Nset, MZset, IZset;
    BDD V1cV2cset;
    
    void initialize() {
        V1 = makeDomain("V1", V_BITS);
        V2 = makeDomain("V2", V_BITS);
        I = makeDomain("I", I_BITS);
        H1 = makeDomain("H1", H_BITS);
        H2 = makeDomain("H2", H_BITS);
        Z = makeDomain("Z", Z_BITS);
        F = makeDomain("F", F_BITS);
        T1 = makeDomain("T1", T_BITS);
        T2 = makeDomain("T2", T_BITS);
        N = makeDomain("N", N_BITS);
        M = makeDomain("M", M_BITS);
        
        V1c = makeDomain("V1c", VC_BITS);
        V2c = makeDomain("V2c", VC_BITS);
        H1c = makeDomain("H1c", HC_BITS);
        H2c = makeDomain("H2c", HC_BITS);
        
        V3 = makeDomain("V3", V_BITS);
        H3 = makeDomain("H3", H_BITS);
        V3c = makeDomain("V3c", VC_BITS);
        H3c = makeDomain("H3c", HC_BITS);
        I2 = makeDomain("I2", I_BITS);
        M2 = makeDomain("M2", M_BITS);
        F2 = makeDomain("F2", F_BITS);
        
        if (varorder.indexOf("V1c") == -1)
            varorder = varorder.replaceFirst("V1", "V1cxV1");
        if (varorder.indexOf("V2c") == -1)
            varorder = varorder.replaceFirst("V2", "V2cxV2");
        if (varorder.indexOf("H1c") == -1)
            varorder = varorder.replaceFirst("H1", "H1cxH1");
        if (varorder.indexOf("H2c") == -1)
            varorder = varorder.replaceFirst("H2", "H2cxH2");
        
        if (varorder.indexOf("V3") == -1)
            varorder = varorder.replaceFirst("V2", "V3xV2");
        if (varorder.indexOf("V3c") == -1)
            varorder = varorder.replaceFirst("V2c", "V3cxV2c");
        if (varorder.indexOf("H3") == -1)
            varorder = varorder.replaceFirst("H2", "H3xH2");
        if (varorder.indexOf("H3c") == -1)
            varorder = varorder.replaceFirst("H2c", "H3cxH2c");
        if (varorder.indexOf("I2") == -1)
            varorder = varorder.replaceFirst("I", "I2xI");
        if (varorder.indexOf("M2") == -1)
            varorder = varorder.replaceFirst("M", "M2xM");
        if (varorder.indexOf("F2") == -1)
            varorder = varorder.replaceFirst("F", "F2xF");
        
        int[] order = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(order);
        
        V1toV2 = bdd.makePair();
        V1toV2.set(new BDDDomain[] {V1,V1c},
                   new BDDDomain[] {V2,V2c});
        V2toV1 = bdd.makePair();
        V2toV1.set(new BDDDomain[] {V2,V2c},
                   new BDDDomain[] {V1,V1c});
        H1toH2 = bdd.makePair();
        H1toH2.set(new BDDDomain[] {H1,H1c},
                   new BDDDomain[] {H2,H2c});
        H2toH1 = bdd.makePair();
        H2toH1.set(new BDDDomain[] {H2,H2c},
                   new BDDDomain[] {H1,H1c});
        V1H1toV2H2 = bdd.makePair();
        V1H1toV2H2.set(new BDDDomain[] {V1,H1,V1c,H1c},
                       new BDDDomain[] {V2,H2,V2c,H2c});
        V2H2toV1H1 = bdd.makePair();
        V2H2toV1H1.set(new BDDDomain[] {V2,H2,V2c,H2c},
                       new BDDDomain[] {V1,H1,V1c,H1c});
        V1cV2ctoV2cV1c = bdd.makePair();
        V1cV2ctoV2cV1c.set(new BDDDomain[] {V1c,V2c},
                           new BDDDomain[] {V2c,V1c});
        V1ctoV2c = bdd.makePair(V1c, V2c);
        
        V1set = V1.set();
        V2set = V2.set();
        H1set = H1.set();
        H2set = H2.set();
        T1set = T1.set();
        T2set = T2.set();
        Fset = F.set();
        Mset = M.set();
        Nset = N.set();
        Iset = I.set();
        Zset = Z.set();
        V1cV2cset = V1c.set(); V1cV2cset.andWith(V2c.set());
        V1set.andWith(V1c.set());
        V2set.andWith(V2c.set());
        H1set.andWith(H1c.set());
        H2set.andWith(H2c.set());
        V1V2set = V1set.and(V2set);
        V1H1set = V1set.and(H1set);
        IMset = Iset.and(Mset);
        IZset = Iset.and(Zset);
        H1Fset = H1set.and(Fset);
        H2Fset = H2set.and(Fset);
        H1FH2set = H1Fset.and(H2set);
        T2Nset = T2set.and(Nset);
        MZset = Mset.and(Zset);
    }

    void initialize(CSPAResults that) {
        V1 = that.V1;
        V2 = that.V2;
        I = that.I;
        H1 = that.H1;
        H2 = that.H2;
        Z = that.Z;
        F = that.F;
        T1 = that.T1;
        T2 = that.T2;
        N = that.N;
        M = that.M;
        
        V1c = that.V1c;
        V2c = that.V2c;
        H1c = that.H1c;
        H2c = that.H2c;
        
        V3 = that.V3;
        H3 = that.H3;
        V3c = that.V3c;
        H3c = that.H3c;
        I2 = that.I2;
        M2 = that.M2;
        F2 = that.F2;
        
        V1toV2 = that.V1toV2;
        V2toV1 = that.V2toV1;
        V1H1toV2H2 = that.V1H1toV2H2;
        V2H2toV1H1 = that.V2H2toV1H1;
        V1cV2ctoV2cV1c = that.V1cV2ctoV2cV1c;
        
        V1set = that.V1set;
        V2set = that.V2set;
        H1set = that.H1set;
        H2set = that.H2set;
        T1set = that.T1set;
        T2set = that.T2set;
        Fset = that.Fset;
        Mset = that.Mset;
        Nset = that.Nset;
        Iset = that.Iset;
        Zset = that.Zset;
        V1cV2cset = that.V1cV2cset;
        V1V2set = that.V1V2set;
        V1H1set = that.V1H1set;
        IMset = that.IMset;
        IZset = that.IZset;
        H1Fset = that.H1Fset;
        H2Fset = that.H2Fset;
        H1FH2set = that.H1FH2set;
        T2Nset = that.T2Nset;
        MZset = that.MZset;
    }

    public static void main(String[] args) throws IOException {
        CSPAResults r = runAnalysis(args, null);
        r.interactive();
    }
    
    public static CSPAResults runAnalysis(String[] args, String addToClasspath) throws IOException {
        String prefix;
        if (args.length > 0) {
            prefix = args[0];
            String sep = System.getProperty("file.separator");
            if (!prefix.endsWith(sep))
                prefix += sep;
        } else {
            prefix = "";
        }
        String fileName = System.getProperty("bddresults", "pa");
        BDDFactory bdd = initialize(addToClasspath);
        CSPAResults r = runAnalysis(bdd, prefix, fileName);
        return r;
    }
    
    public static BDDFactory initialize(String addToClasspath) {
        // We use bytecode maps.
        CodeCache.AlwaysMap = true;
        HostedVM.initialize();
        
        if (addToClasspath != null)
            PrimordialClassLoader.loader.addToClasspath(addToClasspath);
        
        int nodeCount = 500000;
        int cacheSize = 50000;
        BDDFactory bdd = BDDFactory.init(nodeCount, cacheSize);
        //bdd.setMaxIncrease(nodeCount/4);
        bdd.setIncreaseFactor(2);
        return bdd;
    }
    
    public static CSPAResults runAnalysis(BDDFactory bdd,
                                          String prefix,
                                          String fileName) throws IOException {
        CSPAResults r = new CSPAResults(bdd);
        r.loadCallGraph(prefix+"callgraph");
        r.load(prefix+fileName);
        r.numberPaths();
        r.initializeRelations();
        return r;
    }

    public void initializeRelations() {
        buildContextInsensitive();
        initializeMethodMap();
        buildSCCToVarBDD();
        //buildCallGraphRelation();
        //buildAccessibleLocations();
        sanityCheck();
    }
    
    public static String domainName(BDDDomain d) {
        return d.getName();
    }

    public String elementToString(BDDDomain d, BigInteger i) {
        StringBuffer sb = new StringBuffer();
        sb.append(domainName(d)+"("+i+")");
        Node n = null;
        if (d == V1 || d == V2 || d == V3) {
            n = (Node) getVariableNode(i.intValue());
        } else if (d == H1 || d == H2 || d == H3) {
            n = (Node) getHeapNode(i.intValue());
        } else if (d == F) {
            jq_Field f = getField(i.intValue());
            sb.append(": "+(f==null?"[]":f.getName().toString()));
        } else if (d == M) {
            jq_Method m = getMethod(i.intValue());
            sb.append(": "+m.getDeclaringClass().shortName()+"."+m.getName().toString()+"()");
        }
        if (n != null) {
            sb.append(": ");
            sb.append(n.toString_short());
        }
        return sb.toString();
    }

    public static String domainNames(Set dom) {
        StringBuffer sb = new StringBuffer();
        for (Iterator i=dom.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            sb.append(domainName(d));
            if (i.hasNext()) sb.append(',');
        }
        return sb.toString();
    }
    
    public static final Comparator domain_comparator = new Comparator() {

        public int compare(Object arg0, Object arg1) {
            BDDDomain d1 = (BDDDomain) arg0;
            BDDDomain d2 = (BDDDomain) arg1;
            if (d1.getIndex() < d2.getIndex()) return -1;
            else if (d1.getIndex() > d2.getIndex()) return 1;
            else return 0;
        }
        
    };
    
    public static final boolean USE_BC_LOCATION = false;
    
    public static ProgramLocation mapCall(ProgramLocation callSite) {
        if (USE_BC_LOCATION && callSite instanceof ProgramLocation.QuadProgramLocation) {
            jq_Method m = (jq_Method) callSite.getMethod();
            Map map = CodeCache.getBCMap(m);
            Quad q = ((ProgramLocation.QuadProgramLocation) callSite).getQuad();
            if (q == null) {
                Assert.UNREACHABLE("Error: cannot find call site "+callSite);
            }
            Integer i = (Integer) map.get(q);
            if (i == null) {
                Assert.UNREACHABLE("Error: no mapping for quad "+q);
            }
            int bcIndex = i.intValue();
            callSite = new ProgramLocation.BCProgramLocation(m, bcIndex);
        }
        return callSite;
    }
    
    /** ProgramLocation is the location of the method invocation that you want the return value of. */
    public int getReturnValueIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ReturnValueNode) {
                ReturnValueNode ctn = (ReturnValueNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    public int getThrownExceptionIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ThrownExceptionNode) {
                ThrownExceptionNode ctn = (ThrownExceptionNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    int[] getIndices(Collection c) {
        if (c == null) return null;
        int s = c.size();
        if (s == 0) return null;
        int[] r = new int[s];
        int j = -1;
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            r[++j] = getVariableIndex(n);
        }
        Assert._assert(j == r.length-1);
        return r;
    }
    
    public int[] getInvokeParamIndices(ProgramLocation pl, int k) {
        return getInvokeParamIndices(new PassedParameter(pl, k));
    }
    public int[] getInvokeParamIndices(PassedParameter pp) {
        Collection c = passedParams.getValues(pp);
        return getIndices(c);
    }
    
    public int[] getReturnValueIndices(jq_Method m) {
        Collection c = methodToVariables.getValues(m);
        LinkedList result = new LinkedList();
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (returned.contains(o))
                result.add(o);
        }
        return getIndices(result);
    }
    
    public int[] getThrownValueIndices(jq_Method m) {
        Collection c = methodToVariables.getValues(m);
        LinkedList result = new LinkedList();
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (thrown.contains(o))
                result.add(o);
        }
        return getIndices(result);
    }
    
    /** Multimap between methods and their variables. */ 
    MultiMap methodToVariables;
    
    public void initializeMethodMap() {
        methodToVariables = new GenericMultiMap();
        for (Iterator i = Vmap.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o != null)
                methodToVariables.add(o.getDefiningMethod(), o);
        }
    }
    
    public int getLoadIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof FieldNode) {
                FieldNode ctn = (FieldNode) o;
                if (ctn.getLocations().contains(pl))
                    return getVariableIndex(o);
            }
        }
        return -1;
    }
    
    public int getHeapVariableIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    public int getHeapObjectIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl.equals(pl2)) {
                    return getHeapIndex(o);
                }
            }
        }
        return -1;
    }
    
    public Node getVariableNode(int v) {
        if (v < 0 || v >= Vmap.size())
            return null;
        Node n = (Node) Vmap.get(v);
        return n;
    }
    
    public int getVariableIndex(Node n) {
        int size = Vmap.size();
        int v = Vmap.get(n);
        Assert._assert(size == Vmap.size());
        return v;
    }

    public Node getHeapNode(int v) {
        if (v < 0 || v >= Hmap.size())
            return null;
        Node n = (Node) Hmap.get(v);
        return n;
    }
    
    public int getHeapIndex(Node n) {
        int size = Hmap.size();
        int v = Hmap.get(n);
        Assert._assert(size == Hmap.size());
        return v;
    }

    public jq_Method getMethod(int v) {
        if (v < 0 || v >= Mmap.size())
            return null;
        jq_Method n = (jq_Method) Mmap.get(v);
        return n;
    }
    
    public int getMethodIndex(jq_Method m) {
        int size = Mmap.size();
        int v = Mmap.get(m);
        Assert._assert(size == Mmap.size());
        return v;
    }
    
    public jq_Field getField(int v) {
        if (v < 0 || v >= Fmap.size())
            return null;
        jq_Field n = (jq_Field) Fmap.get(v);
        return n;
    }
    
    public ProgramLocation getHeapProgramLocation(int v) {
        Node n = (Node) getHeapNode(v);
        if (n instanceof ConcreteTypeNode)
            return ((ConcreteTypeNode) n).getLocation();
        return null;
    }

    public static jq_Class getClass(String classname) {
        jq_Class klass = (jq_Class) jq_Type.parseType(classname);
        return klass;
    }

    public static jq_Method getMethod(String classname, String name, String desc) {
        jq_Class klass = (jq_Class) jq_Type.parseType(classname);
        if (klass == null) return null;
        jq_Method m = (jq_Method) klass.getDeclaredMember(name, desc);
        return m;
    }
    
    public int getParameterIndex(jq_Method m, int k) {
        Collection c = methodToVariables.getValues(m);
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof ParamNode) {
                ParamNode pn = (ParamNode) o;
                if (pn.getMethod() == m && k == pn.getIndex())
                    return getVariableIndex(pn);
            }
        }
        return -1;
    }
    
    public class TypedBDD {
        private static final int DEFAULT_NUM_TO_PRINT = 6;
        private static final int PRINT_ALL = -1;
        BDD bdd;
        Set dom;
        
        /**
         * @param bdd
         * @param domains
         */
        public TypedBDD(BDD bdd, BDDDomain[] domains) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.addAll(Arrays.asList(domains));
        }
        
        public TypedBDD(BDD bdd, Set domains) {
            this.bdd = bdd;
            this.dom = domains;
        }
            
        public TypedBDD(BDD bdd, BDDDomain d) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3, BDDDomain d4) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
            this.dom.add(d4);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3, BDDDomain d4, BDDDomain d5) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
            this.dom.add(d4);
            this.dom.add(d5);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3, BDDDomain d4, BDDDomain d5, BDDDomain d6) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
            this.dom.add(d4);
            this.dom.add(d5);
            this.dom.add(d6);
        }
        
        public TypedBDD relprod(TypedBDD bdd1, TypedBDD set) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            newDom.addAll(bdd1.dom);
            if (!newDom.containsAll(set.dom)) {
                System.err.println("Warning! Quantifying domain that doesn't exist: "+domainNames(set.dom));
            }
            newDom.removeAll(set.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.relprod(bdd1.bdd, set.bdd), newDom);
        }
        
        public TypedBDD restrict(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.containsAll(bdd1.dom)) {
                System.err.println("Warning! Restricting domain that doesn't exist: "+domainNames(bdd1.dom));
            }
            if (bdd1.satCount() > 1.0) {
                System.err.println("Warning! Using restrict with more than one value");
            }
            newDom.removeAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.restrict(bdd1.bdd), newDom);
        }
        
        public TypedBDD exist(TypedBDD set) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.containsAll(set.dom)) {
                System.err.println("Warning! Quantifying domain that doesn't exist: "+domainNames(set.dom));
            }
            newDom.removeAll(set.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.exist(set.bdd), newDom);
        }
        
        public TypedBDD and(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            newDom.addAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.and(bdd1.bdd), newDom);
        }
        
        public TypedBDD or(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.equals(bdd1.dom)) {
                System.err.println("Warning! Or'ing BDD with different domains: "+domainNames(bdd1.dom));
            }
            newDom.addAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.or(bdd1.bdd), newDom);
        }
        
        public boolean isZero() {
            return bdd.isZero();
        }
        
        public String getDomainNames() {
            return domainNames(dom);
        }
        
        private BDD getDomains() {
            BDDFactory f = bdd.getFactory();
            BDD r = f.one();
            for (Iterator i=dom.iterator(); i.hasNext(); ) {
                BDDDomain d = (BDDDomain) i.next();
                r.andWith(d.set());
            }
            return r;
        }
        
        public double satCount() {
            return bdd.satCount(getDomains());
        }
        
        public String toString() {
            return toString(DEFAULT_NUM_TO_PRINT);
        }
        
        public String toStringAll() {
            return toString(PRINT_ALL);
        }
        
        public String toString(int numToPrint) {
            BDD dset = getDomains();
            double s = bdd.satCount(dset);
            if (s == 0.) return "<empty>";
            BDD b = bdd.id();
            StringBuffer sb = new StringBuffer();
            int j = 0;
            while (!b.isZero()) {
                if (numToPrint != PRINT_ALL && j > numToPrint - 1) {
                    sb.append("\tand "+(long)b.satCount(dset)+" others.");
                    sb.append(Strings.lineSep);
                    break;
                }
                BigInteger[] val = b.scanAllVar();
                //sb.append((long)b.satCount(dset));
                sb.append("\t(");
                BDD temp = b.getFactory().one();
                for (Iterator i=dom.iterator(); i.hasNext(); ) {
                    BDDDomain d = (BDDDomain) i.next();
                    BigInteger e = val[d.getIndex()];
                    if (e.signum() < 0 || e.compareTo(d.size()) >= 0) {
                        System.out.println("Error: out of range "+e);
                        break;
                    }
                    sb.append(elementToString(d, e));
                    if (i.hasNext()) sb.append(' ');
                    temp.andWith(d.ithVar(e));
                }
                sb.append(')');
                b.applyWith(temp, BDDFactory.diff);
                ++j;
                sb.append(Strings.lineSep);
            }
            //sb.append(bdd.toStringWithDomains());
            return sb.toString();
        }

        public Iterator iterator() {
            Assert._assert(dom.size() == 1);
            final BDD t = this.bdd.id();
            final BDDDomain d = (BDDDomain) this.dom.iterator().next();
            return new UnmodifiableIterator() {

                public boolean hasNext() {
                    return !t.isZero();
                }

                public int nextInt() {
                    int v = t.scanVar(d).intValue();
                    if (v == -1)
                        throw new NoSuchElementException();
                    t.applyWith(d.ithVar(v), BDDFactory.diff);
                    return v;
                }

                public Object next() {
                    return new Integer(nextInt());
                }
            };
        }

        public void free() {
            bdd.free(); bdd = null;
        }
    }

    TypedBDD parseBDD(List a, String s) {
        if (s.equals("vP")) {
            return new TypedBDD(vP, V1c, V1, H1c, H1);
        }
        if (s.equals("vP_ci")) {
            return new TypedBDD(vP_ci, V1, H1);
        }
        if (s.equals("hP")) {
            return new TypedBDD(hP, H1c, H1, F, H2c, H2);
        }
        if (s.equals("S")) {
            return new TypedBDD(S, V1c, V1, F, V2c, V2);
        }
        if (s.equals("L")) {
            return new TypedBDD(L, V1c, V1, F, V2c, V2);
        }
        if (s.equals("A")) {
            return new TypedBDD(A, V1c, V1, V2c, V2);
        }
        if (s.equals("al")) {
            return new TypedBDD(accessibleLocations, V1c, V1, V2c, V2);
        }
        if (s.equals("cg")) {
            if (callGraphRelation == null)
                buildCallGraphRelation();
            return new TypedBDD(callGraphRelation, V1c, V1, V2c, V2);
        }
        if (s.startsWith("V1(")) {
            int x = Integer.parseInt(s.substring(3, s.length()-1));
            return new TypedBDD(V1.ithVar(x), V1);
        }
        if (s.startsWith("V1c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V1c.ithVar(x), V1c);
        }
        if (s.startsWith("V2(")) {
            int x = Integer.parseInt(s.substring(3, s.length()-1));
            return new TypedBDD(V2.ithVar(x), V2);
        }
        if (s.startsWith("V2c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V2c.ithVar(x), V2c);
        }
        if (s.startsWith("H1(")) {
            int x = Integer.parseInt(s.substring(3, s.length()-1));
            return new TypedBDD(H1.ithVar(x), H1);
        }
        if (s.startsWith("H1c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H1c.ithVar(x), H1c);
        }
        if (s.startsWith("H2(")) {
            int x = Integer.parseInt(s.substring(3, s.length()-1));
            return new TypedBDD(H2.ithVar(x), H2);
        }
        if (s.startsWith("H2c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H2c.ithVar(x), H2c);
        }
        if (s.startsWith("F(")) {
            int x = Integer.parseInt(s.substring(2, s.length()-1));
            return new TypedBDD(F.ithVar(x), F);
        }
        if (s.startsWith("M(")) {
            int x = Integer.parseInt(s.substring(2, s.length()-1));
            return new TypedBDD(M.ithVar(x), M);
        }
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return new TypedBDD(d.domain(), d);
        }
        int i = Integer.parseInt(s)-1;
        return (TypedBDD) a.get(i);
    }

    TypedBDD parseBDDset(List a, String s) {
        if (s.equals("V1")) {
            BDD b = V1.set(); b.andWith(V1c.set());
            return new TypedBDD(b, V1c, V1);
        }
        if (s.equals("H1")) {
            BDD b = H1.set(); b.andWith(H1c.set());
            return new TypedBDD(b, H1c, H1);
        }
        if (s.equals("C")) {
            BDD b = V1c.set(); b.andWith(H1c.set());
            return new TypedBDD(b, V1c, H1c);
        }
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return new TypedBDD(d.set(), d);
        }
        int i = Integer.parseInt(s)-1;
        return (TypedBDD) a.get(i);
    }

    BDDDomain parseDomain(String dom) {
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            if (dom.equals(bdd.getDomain(i).getName()))
                return bdd.getDomain(i);
        }
        return null;
    }

    void interactive() {
        int i = 1;
        List results = new ArrayList();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        for (;;) {
            boolean increaseCount = true;
            boolean listAll = false;
            
            try {
                System.out.print(i+"> ");
                String s = in.readLine();
                if (s == null) return;
                StringTokenizer st = new StringTokenizer(s);
                if (!st.hasMoreElements()) continue;
                String command = st.nextToken();
                if (command.equals("relprod")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = bdd1.relprod(bdd2, set);
                    results.add(r);
                } else if (command.equals("restrict")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.restrict(bdd2);
                    results.add(r);
                } else if (command.equals("exist")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = bdd1.exist(set);
                    results.add(r);
                } else if (command.equals("and")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.and(bdd2);
                    results.add(r);
                } else if (command.equals("or")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.or(bdd2);
                    results.add(r);
                } else if (command.equals("var")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = new TypedBDD(V1.ithVar(z), V1);
                    results.add(r);
                } else if (command.equals("heap")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = new TypedBDD(H1.ithVar(z), H1);
                    results.add(r);
                } else if (command.equals("quit") || command.equals("exit")) {
                    break;
                } else if (command.equals("aliased")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = getAliasedLocations(z);
                    results.add(r);
                } else if (command.equals("heapType")) {
                    jq_Reference typeRef = (jq_Reference) jq_Type.parseType(st.nextToken());
                    if (typeRef != null) {
                        TypedBDD r = getAllHeapOfType(typeRef);
                        results.add(r);
                    }
                } else if (command.equals("list")) {
                    TypedBDD r = parseBDD(results, st.nextToken());
                    results.add(r);
                    listAll = true;
                    System.out.println("Domains: " + r.getDomainNames());
                } else if (command.equals("contextvar")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = getVariableNode (varNum);
                    if (n == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        jq_Method m = n.getDefiningMethod();
                        Number c = new BigInteger(st.nextToken(), 10);
                        if (m == null) {
                            System.out.println("No method for node "+n);
                        } else {
                            Path trace = pn.getPath(m, c);
                            System.out.println(m+" context "+c+":\n"+trace);
                        }
                    }
                    increaseCount = false;
                } else if (command.equals("contextheap")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = (Node) getHeapNode(varNum);
                    if (n == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        jq_Method m = n.getDefiningMethod();
                        Number c = new BigInteger(st.nextToken(), 10);
                        if (m == null) {
                            System.out.println("No method for node "+n);
                        } else {
                            Path trace = pn.getPath(m, c);
                            System.out.println(m+" context "+c+": "+trace);
                        }
                    }
                    increaseCount = false;
                } else if (command.equals("method")) {
                    jq_Class c = (jq_Class) jq_Type.parseType(st.nextToken());
                    if (c == null) {
                        System.out.println("Cannot find class");
                        increaseCount = false;
                    } else {
                        jq_Method m = c.getDeclaredMethod(st.nextToken());
                        if (m == null) {
                            System.out.println("Cannot find method");
                            increaseCount = false;
                        } else {
                            System.out.println("Method: "+m);
                            int k = Mmap.get(m);
                            TypedBDD bdd = new TypedBDD(M.ithVar(k), M);
                            results.add(bdd);
                        }
                    }
                } else if (command.equals("methodvars")) {
                    jq_Class c = (jq_Class) jq_Type.parseType(st.nextToken());
                    if (c == null) {
                        System.out.println("Cannot find class");
                        increaseCount = false;
                    } else {
                        jq_Method m = c.getDeclaredMethod(st.nextToken());
                        if (m == null) {
                            System.out.println("Cannot find method");
                            increaseCount = false;
                        } else {
                            System.out.println("Method: "+m);
                            Collection method_nodes = methodToVariables.getValues(m);
                            BDD localVars = bdd.zero();
                            for (Iterator k = method_nodes.iterator(); k.hasNext(); ) {
                                Node node = (Node) k.next();
                                int x = getVariableIndex(node);
                                localVars.orWith(V1.ithVar(x));
                            }
                            results.add(new TypedBDD(localVars, V1));
                        }
                    }
                } else if (command.equals("aliasedparams")) {
                    findAliasedParameters();
                    increaseCount = false;
                } else if (command.equals("findequiv")) {
                    TypedBDD bdd = findEquivalentObjects_fs();
                    results.add(bdd);
                } else if (command.equals("reachableobjs")) {
                    int heapNum = Integer.parseInt(st.nextToken());
                    TypedBDD bdd = getReachableObjects(heapNum);
                    results.add(bdd);
                } else if (command.equals("dumpconnect")) {
                    int heapNum = Integer.parseInt(st.nextToken());
                    String filename = "connect.dot";
                    if (st.hasMoreTokens()) filename = st.nextToken();
                    BufferedWriter dos = new BufferedWriter(new FileWriter(filename));
                    dumpObjectConnectivityGraph(heapNum, dos);
                    increaseCount = false;
                } else if (command.equals("dumpallconnect")) {
                    String filename = "connect.dot";
                    if (st.hasMoreTokens()) filename = st.nextToken();
                    BufferedWriter dos = new BufferedWriter(new FileWriter(filename));
                    dumpObjectConnectivityGraph(dos);
                    increaseCount = false;
                } else if (command.equals("escape")) {
                    escapeAnalysis();
                    increaseCount = false;
                } else if (command.equals("mod")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = getVariableNode (varNum);
                    if (n == null) {
                        System.out.println("No method for node "+n);
                        increaseCount = false;
                    } else {
                        jq_Method m = n.getDefiningMethod();
                        TypedBDD bdd = findMod(m);
                        results.add(bdd);
                    }
                } else if (command.equals("countmod")) {
                    //if (accessibleLocations == null) buildAccessibleLocations();
                    countMod();
                    increaseCount = false;
                } else if (command.equals("ref")) {
                    if (accessibleLocations == null) buildAccessibleLocations();
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd = findRef(bdd1);
                    results.add(bdd);
                } else if (command.equals("whorefs")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd = new TypedBDD(whoReferences(bdd1.bdd), M);
                    results.add(bdd);
                } else if (command.equals("whomods")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd = new TypedBDD(whoModifies(bdd1.bdd), M, F);
                    results.add(bdd);
                } else if (command.equals("whomodsfield")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd = new TypedBDD(whoModifies(bdd1.bdd), M, F);
                    results.add(bdd);
                } else if (command.equals("syncedvars")) {
                    TypedBDD bdd = new TypedBDD(getSyncedVars(), V1);
                    results.add(bdd);
                } else if (command.equals("unnecessarysyncs")) {
                    TypedBDD bdd = new TypedBDD(getUnnecessarySyncs(), V1);
                    results.add(bdd);
                } else if (command.equals("thread")) {
                    int k = Integer.parseInt(st.nextToken());
                    jq_Method run = null;
                    Iterator j = thread_runs.keySet().iterator();
                    while (--k >= 0) {
                        run = (jq_Method) j.next();
                    }
                    System.out.println(k+": "+run);
                    BDD m = M.ithVar(getMethodIndex(run));
                    TypedBDD bdd = new TypedBDD(m, M);
                    results.add(bdd);
                } else if (command.equals("threadlocal")) {
                    TypedBDD bdd = new TypedBDD(getThreadLocalObjects(), H1c, H1);
                    results.add(bdd);
                } else if (command.equals("reachable")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd = new TypedBDD(getReachableVars(bdd1.bdd), V1c, V1);
                    results.add(bdd);
                } else if (command.equals("comparemods")) {
                    compareMods();
                    increaseCount = false;
                } else if (command.equals("printusedef")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    printUseDefChain(bdd1.bdd);
                    increaseCount = false;
                } else if (command.equals("usedef")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    BDD bdd = printUseDef(bdd1.bdd);
                    results.add(new TypedBDD(bdd, V1c, V1));
                } else if (command.equals("printdefuse")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    printDefUseChain(bdd1.bdd);
                    increaseCount = false;
                } else if (command.equals("encapsulation")) {
                    countEncapsulation();
                    increaseCount = false;
                } else if (command.equals("help")) {
                    printHelp();
                    increaseCount = false;
                } else {
                    System.err.println("Unrecognized command");
                    increaseCount = false;
                    //results.add(new TypedBDD(bdd.zero(), Collections.EMPTY_SET));
                }
            } catch (IOException e) {
                System.err.println("Error: IOException");
                increaseCount = false;
            } catch (NumberFormatException e) {
                System.err.println("Parse error: NumberFormatException");
                increaseCount = false;
            } catch (NoSuchElementException e) {
                System.err.println("Parse error: NoSuchElementException");
                increaseCount = false;
            } catch (IndexOutOfBoundsException e) {
                System.err.println("Parse error: IndexOutOfBoundsException");
                increaseCount = false;
            }

            if (increaseCount) {
                TypedBDD r = (TypedBDD) results.get(i-1);
                if (listAll) {
                    System.out.println(i+" -> "+r.toStringAll());
                } 
                else {
                    System.out.println(i+" -> "+r);
                }
                Assert._assert(i == results.size());
                ++i;
            }
        }
    }

    public CSPAResults(BDDFactory bdd) {
        this.bdd = bdd;
    }

    public static void printHelp() {
        System.out.println("dumpconnect # <fn>:   dump heap connectivity graph for heap object # to file fn");
        System.out.println("dumpallconnect <fn>:  dump entire heap connectivity graph to file fn");
        System.out.println("escape:               run escape analysis");
        System.out.println("findequiv:            find equivalent objects");
        System.out.println("relprod b1 b2 bs:     relational product of b1 and b2 w.r.t. set bs");
    }

    public BDD whoReferences(BDD heaps) {
        BDD vars = vP.relprod(heaps, H1set);
        BDD methods = mV.relprod(vars, V1set);
        vars.free();
        return methods;
    }
    
    public BDD whoModifies(BDD heaps) { // H1
        BDD vars = vP.relprod(heaps, H1set); // V1xH1 x H1 = V1
        vars.andWith(S.exist(V2set)); // V1xF
        BDD methods = mV.relprod(vars, V1set); // MxV1 x V1xF = MxF
        vars.free();
        return methods;
    }
    
    public BDD whoModifiesField(BDD fields) { // F
        BDD vars = S.relprod(fields, V2set);
        BDD methods = mV.relprod(vars, V1set); // MxV1 x V1xF = MxF
        vars.free();
        return methods;
    }
    
    public void compareMods() {
        double totalTypeBased = 0;
        double totalPointerBased = 0;
        int[] histogram_type = new int[10];
        int[] histogram_pointer = new int[10];
        int num = 0;
        for (int i = 0; i < Hmap.size(); ++i) {
            Node n = (Node) Hmap.get(i);
            jq_Reference type = n.getDeclaredType();
            if (type == null) continue;
            BDD H_bdd = H1.ithVar(i);
            BDD whoMods = whoModifies(H_bdd);
            for (int F_i = 0; F_i < Fmap.size(); ++F_i) {
                jq_Field f = (jq_Field) Fmap.get(F_i);
                if (f != null) {
                    f.getDeclaringClass().prepare();
                    type.prepare();
                    if (!f.getDeclaringClass().isSubtypeOf(type)) {
                        continue;
                    }
                }
                BDD F_bdd = F.ithVar(F_i);
                BDD typeBased = whoModifiesField(F_bdd);
                typeBased = typeBased.exist(Fset);
                BDD pointerBased = whoMods.restrict(F_bdd);
                int type_num = (int) typeBased.satCount(Mset);
                Assert._assert(type_num == (int) typeBased.satCount(Mset));
                int pointer_num = (int) pointerBased.satCount(Mset);
                Assert._assert(pointer_num == (int) pointerBased.satCount(Mset));
                totalTypeBased += type_num;
                totalPointerBased += pointer_num;
                type_num = Math.min(histogram_type.length-1, type_num);
                histogram_type[type_num]++;
                pointer_num = Math.min(histogram_pointer.length-1, pointer_num);
                histogram_pointer[pointer_num]++;
                ++num;
            }
        }
        System.out.println("Number: "+num);
        System.out.println("Type Based: "+totalTypeBased/num);
        System.out.println("Pointer Based: "+totalPointerBased/num);
        for (int i = 0; i < histogram_type.length; ++i) {
            System.out.print(i+": "+histogram_type[i]);
            System.out.print("\t\t");
            System.out.println(histogram_pointer[i]);
        }
    }
    
    public BDD getUnnecessarySyncs() {
        BDD syncs = getSyncedVars(); // V1
        BDD syncObjs = syncs.and(vP); // V1xH1
        BDD local = getThreadLocalObjects(); // H1
        BDD u = syncObjs.applyAll(local, BDDFactory.imp, H1set);
        u.andWith(syncs);
        return u;
    }
    
    public BDD getSyncedVars() {
        BDD caller_callee = IE.exist(V1cV2cset);
        BDD result = bdd.zero();
        for (int i = 0; i < Mmap.size(); ++i) {
            jq_Method n = (jq_Method) Mmap.get(i);
            if (!n.isSynchronized()) continue;
            BDD M_bdd = M.ithVar(i); // M
            BDD callers = caller_callee.relprod(M_bdd, Mset); // IxM x M = I
            callers.andWith(Z.ithVar(0)); // IxZ
            BDD thispointers = actual.relprod(callers, IZset); // IxZ x IxZxV2 = V2
            callers.free();
            thispointers.replaceWith(V2toV1);
            System.out.println(n+" synced locations:");
            System.out.println(new TypedBDD(thispointers, V1));
            result.orWith(thispointers);
        }
        caller_callee.free();
        return result;
    }
    
    public BDD getReachableVars(BDD method_plus_context0) {
        System.out.println("Method = "+method_plus_context0.toStringWithDomains());
        BDD result = bdd.zero();
        BDD allInvokes = mI.exist(Nset);
        BDD new_m = method_plus_context0.id();
        BDD V2cIset = Iset.and(V2c.set());
        for (int k=1; ; ++k) {
            System.out.println("Iteration "+k);
            BDD vars = new_m.relprod(mV, Mset); // V1cxM x MxV1 = V1cxV1
            result.orWith(vars);
            BDD invokes = new_m.relprod(allInvokes, Mset); // V1cxM x MxI = V1cxI
            invokes.replaceWith(V1ctoV2c); // V2cxI
            BDD methods = invokes.relprod(IEc, V2cIset); // V2cxI x V2cxIxV1cxM = V1cxM
            new_m.orWith(methods);
            new_m.applyWith(method_plus_context0.id(), BDDFactory.diff);
            if (new_m.isZero()) break;
            method_plus_context0.orWith(new_m.id());
        }
        return result;
    }
    
    public BDD getThreadLocalObjects() {
        jq_NameAndDesc main_nd = new jq_NameAndDesc("main", "([Ljava/lang/String;)V");
        jq_Method main = null;
        for (Iterator i = Mmap.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (main_nd.equals(m.getNameAndDesc())) {
                main = m;
                System.out.println("Using main() method: "+main);
                break;
            }
        }
        BDD allObjects = bdd.zero();
        BDD sharedObjects = bdd.zero();
        if (main != null) {
            int M_i = Mmap.get(main);
            BDD m = M.ithVar(M_i);
            m.andWith(V1c.ithVar(0));
            System.out.println("Main: "+m.toStringWithDomains());
            BDD b = getReachableVars(m);
            m.free();
            System.out.println("Reachable vars: "+b.satCount(V1set));
            BDD b2 = b.relprod(vP, V1set);
            b.free();
            System.out.println("Reachable objects: "+b2.satCount(H1set));
            allObjects.orWith(b2);
        }
        for (Iterator i = thread_runs.keySet().iterator(); i.hasNext(); ) {
            jq_Method run = (jq_Method) i.next();
            int M_i = Mmap.get(run);
            Set t_runs = (Set) thread_runs.get(run);
            if (t_runs == null) {
                System.out.println("Unknown run() method: "+run);
                continue;
            }
            Iterator k = t_runs.iterator();
            for (int j = 0; k.hasNext(); ++j) {
                Node q = (Node) k.next();
                BDD m = M.ithVar(M_i);
                m.andWith(V1c.ithVar(j));
                System.out.println("Thread: "+m.toStringWithDomains()+" Object: "+q);
                BDD b = getReachableVars(m);
                m.free();
                System.out.println("Reachable vars: "+b.satCount(V1set));
                BDD b2 = b.relprod(vP, V1set);
                b.free();
                System.out.println("Reachable objects: "+b2.satCount(H1set));
                BDD b3 = allObjects.and(b2);
                System.out.println("Shared objects: "+b3.satCount(H1set));
                sharedObjects.orWith(b3);
                allObjects.orWith(b2);
            }
        }
        
        System.out.println("All shared objects: "+sharedObjects.satCount(H1set));
        allObjects.applyWith(sharedObjects, BDDFactory.diff);
        System.out.println("All local objects: "+allObjects.satCount(H1set));
        
        return allObjects;
    }
    
    public void countEncapsulation() {
        int totalEncapsulated = 0;
        int totalUnencapsulated = 0;
        for (int i = 0; i < Hmap.size(); ++i) {
            Node n = (Node) Hmap.get(i);
            jq_Reference type = n.getDeclaredType();
            if (type == null) continue;
            BDD H_bdd = H1.ithVar(i);
            BDD M_bdd = whoReferences(H_bdd);
            boolean encapsulated = true;
            for (Iterator j = new TypedBDD(M_bdd, M).iterator(); j.hasNext(); ) {
                int M_i = ((Integer) j.next()).intValue();
                jq_Method m = getMethod(M_i);
                m.getDeclaringClass().prepare(); type.prepare();
                if (!m.getDeclaringClass().isSubtypeOf(type)) {
                    encapsulated = false;
                    break;
                }
            }
            if (encapsulated)
                totalEncapsulated++;
            else
                totalUnencapsulated++;
            H_bdd.free();
            M_bdd.free();
        }
        System.out.println("Total encapsulated: "+totalEncapsulated);
        System.out.println("Total unencapsulated: "+totalUnencapsulated);
    }
    
    public void printUseDefChain(BDD vPrelation) {
        BDD visited = bdd.zero();
        vPrelation = vPrelation.id();
        for (int k = 1; !vPrelation.isZero(); ++k) {
            System.out.println("Step "+k+":");
            System.out.println(vPrelation.toStringWithDomains(ts));
            visited.orWith(vPrelation.id());
            // A: v2=v1;
            BDD b = A.relprod(vPrelation, V1set);
            System.out.println("Arguments/Return Values = "+b.satCount(V2set));
            // L: v2=v1.f;
            vPrelation.replaceWith(V1toV2);
            BDD c = L.relprod(vPrelation, V2set); // V1xF
            vPrelation.free();
            BDD d = vP.relprod(c, V1set); // H1xF
            c.free();
            BDD e = hP.relprod(d, H1Fset); // H2
            d.free();
            e.replaceWith(H2toH1);
            BDD f = vP.relprod(e, H1set); // V1
            System.out.println("Loads/Stores = "+f.satCount(V1set));
            e.free();
            vPrelation = b;
            vPrelation.replaceWith(V2toV1);
            vPrelation.orWith(f);
            vPrelation.applyWith(visited.id(), BDDFactory.diff);
        }
    }
    
    public void dumpUseDefChain(BufferedWriter out, BDD vPrelation) throws IOException {
        BDD visited = bdd.zero();
        vPrelation = vPrelation.id();
        HashSet visitedNodes = new HashSet();
        for (int k = 1; !vPrelation.isZero(); ++k) {
            {
                BDD foo = vPrelation.exist(V1c.set());
                for (Iterator i = new TypedBDD(foo, V1).iterator(); i.hasNext(); ) {
                    int V_i = ((Integer) i.next()).intValue();
                    Node n = (Node) Vmap.get(V_i);
                    String name = n.getDefiningMethod().toString();
                    if (!visitedNodes.contains(n)) {
                        out.write("n"+V_i+" [label=\""+name+"\"];\n");
                        visitedNodes.add(n);
                    }
                }
            }
            System.out.println("Step "+k+":");
            System.out.println(vPrelation.toStringWithDomains(ts));
            visited.orWith(vPrelation.id());
            // A: v2=v1;
            BDD b = A.relprod(vPrelation, V1set);
            System.out.println("Arguments/Return Values = "+b.satCount(V2set));
            // L: v2=v1.f;
            vPrelation.replaceWith(V1toV2);
            BDD c = L.relprod(vPrelation, V2set); // V1xF
            vPrelation.free();
            BDD d = vP.relprod(c, V1set); // H1xF
            c.free();
            BDD e = hP.relprod(d, H1Fset); // H2
            d.free();
            e.replaceWith(H2toH1);
            BDD f = vP.relprod(e, H1set); // V1
            System.out.println("Loads/Stores = "+f.satCount(V1set));
            e.free();
            vPrelation = b;
            vPrelation.replaceWith(V2toV1);
            vPrelation.orWith(f);
            vPrelation.applyWith(visited.id(), BDDFactory.diff);
        }
    }
    
    public BDD printUseDef(BDD vPrelation) {
        vPrelation = vPrelation.id();
        // A: v2=v1;
        BDD b = A.relprod(vPrelation, V1set);
        System.out.println("Arguments/Return Values = "+b.satCount(V2set));
        // L: v2=v1.f;
        vPrelation.replaceWith(V1toV2);
        BDD c = L.relprod(vPrelation, V2set); // V1xF
        vPrelation.free();
        BDD d = vP.relprod(c, V1set); // H1xF
        c.free();
        BDD e = hP.relprod(d, H1Fset); // H2
        d.free();
        e.replaceWith(H2toH1);
        BDD f = vP.relprod(e, H1set); // V1
        System.out.println("Loads/Stores = "+f.satCount(V1set));
        e.free();
        vPrelation = b;
        vPrelation.replaceWith(V2toV1);
        vPrelation.orWith(f);
        System.out.println(vPrelation.toStringWithDomains(ts));
        return vPrelation;
    }
    
    public void printDefUseChain(BDD vPrelation) {
        BDD visited = bdd.zero();
        vPrelation = vPrelation.id();
        for (int k = 1; !vPrelation.isZero(); ++k) {
            System.out.println("Step "+k+":");
            System.out.println(vPrelation.toStringWithDomains(ts));
            visited.orWith(vPrelation.id());
            vPrelation.replaceWith(V1toV2);
            // A: v2=v1;
            BDD b = A.relprod(vPrelation, V2set);
            vPrelation.free();
            vPrelation = b;
            vPrelation.applyWith(visited.id(), BDDFactory.diff);
        }
    }
    
    ToString ts = new ToString();
    
    public class ToString extends BDD.BDDToString {
        
        /* (non-Javadoc)
         * @see net.sf.javabdd.BDD.BDDToString#elementName(int, long)
         */
        public String elementName(int arg0, BigInteger arg1) {
            return elementToString(bdd.getDomain(arg0), arg1);
        }
        
        /* (non-Javadoc)
         * @see net.sf.javabdd.BDD.BDDToString#elementNames(int, long, long)
         */
        public String elementNames(int arg0, BigInteger arg1, BigInteger arg2) {
            StringBuffer sb = new StringBuffer();
            while (arg1.compareTo(arg2) < 0) {
                sb.append(elementName(arg0, arg1));
                sb.append(", ");
                arg1 = arg1.add(BigInteger.ONE);
            }
            sb.append(elementName(arg0, arg1));
            return sb.toString();
        }

    }
    
}

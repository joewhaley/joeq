// PA.java, created Oct 16, 2003 3:39:34 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigInteger;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Compil3r.Quad.CachedCallGraph;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.LoadedCallGraph;
import Main.HostedVM;
import Util.Collections.IndexMap;
import Util.Collections.Pair;
import Util.Graphs.PathNumbering;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;
import Util.Graphs.PathNumbering.Range;
import Util.Graphs.PathNumbering.Selector;

/**
 * Context-insensitive pointer analysis using BDDs.
 * This version corresponds exactly to the description in the paper.
 * All of the inference rules are direct copies.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class PA {

    boolean TRACE = false;
    boolean TRACE_SOLVER = false;
    boolean TRACE_BIND = false;
    boolean TRACE_RELATIONS = false;
    PrintStream out = System.out;

    boolean ADD_CLINIT = true;
    boolean ADD_THREADS = true;
    boolean ADD_FINALIZERS = true;
    boolean FILTER_TYPE = true;
    boolean INCREMENTAL1 = true;
    boolean INCREMENTAL2 = true;
    boolean INCREMENTAL3 = true;
    boolean CONTEXT_SENSITIVE = System.getProperty("pa.cs") != null;
    boolean DISCOVER_CALL_GRAPH = System.getProperty("pa.discover") != null;
    boolean DUMP_DOTGRAPH = System.getProperty("pa.dumpdotgraph") != null;
    
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "2500000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "150000"));
    static String resultsFileName = System.getProperty("pa.results", "pa");
    static String callgraphFileName = System.getProperty("pa.callgraph", "callgraph");
    static String initialCallgraphFileName = System.getProperty("pa.icallgraph", callgraphFileName);
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    BDDDomain V1c, V2c, H1c, H2c;
    
    int V_BITS=17, I_BITS=16, H_BITS=15, Z_BITS=5, F_BITS=12, T_BITS=12, N_BITS=13, M_BITS=14;
    int VC_BITS=1, HC_BITS=1;
    int MAX_HC_BITS = Integer.parseInt(System.getProperty("pa.maxhc", "6"));
    
    IndexMap/*Node*/ Vmap;
    IndexMap/*ProgramLocation*/ Imap;
    IndexMap/*Node*/ Hmap;
    IndexMap/*jq_Field*/ Fmap;
    IndexMap/*jq_Reference*/ Tmap;
    IndexMap/*jq_Method*/ Nmap;
    IndexMap/*jq_Method*/ Mmap;
    PathNumbering vCnumbering;
    PathNumbering hCnumbering;
    
    BDD A;      // V1xV2, arguments and return values   (+context)
    BDD vP;     // V1xH1, variable points-to            (+context)
    BDD S;      // (V1xF)xV2, stores                    (+context)
    BDD L;      // (V1xF)xV2, loads                     (+context)
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
    BDD sync;   // V, synced locations                  (no context)
    
    BDD hP;     // H1xFxH2, heap points-to              (+context)
    BDD IE;     // IxM, invocation edges                (no context)
    BDD filter; // V1xH1, type filter                   (no context)
    BDD IEc;    // V2cxIxV1cxM, context-sensitive edges
    
    BDD visited; // M, visited methods
    
    String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2xH2c_T2_H1xH1c");
    //String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_H2_T2_H1");
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDDPairing V1cV2ctoV2cV1c;
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1H1set, IMset, H1Fset, H2Fset, H1FH2set, T2Nset, MZset;
    BDD V1cV2cset;
    
    Set visitedMethods = new HashSet();
    Set roots = new HashSet();
    
    BDDDomain makeDomain(String name, int bits) {
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }
    
    public void initialize() {
        bdd = BDDFactory.init(bddnodes, bddcache);
        bdd.setMaxIncrease(bddnodes/4);
        
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
        
        int[] ordering = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(ordering);
        
        Vmap = makeMap("Vars", V_BITS);
        Imap = makeMap("Invokes", I_BITS);
        Hmap = makeMap("Heaps", H_BITS);
        Fmap = makeMap("Fields", F_BITS);
        Tmap = makeMap("Types", T_BITS);
        Nmap = makeMap("Names", N_BITS);
        Mmap = makeMap("Methods", M_BITS);
        
        if (CONTEXT_SENSITIVE) {
            V1toV2 = bdd.makePair();
            V1toV2.set(new BDDDomain[] {V1,V1c},
                       new BDDDomain[] {V2,V2c});
            V2toV1 = bdd.makePair();
            V2toV1.set(new BDDDomain[] {V2,V2c},
                       new BDDDomain[] {V1,V1c});
            V1H1toV2H2 = bdd.makePair();
            V1H1toV2H2.set(new BDDDomain[] {V1,H1,V1c,H1c},
                           new BDDDomain[] {V2,H2,V2c,H2c});
            V2H2toV1H1 = bdd.makePair();
            V2H2toV1H1.set(new BDDDomain[] {V2,H2,V2c,H2c},
                           new BDDDomain[] {V1,H1,V1c,H1c});
            V1cV2ctoV2cV1c = bdd.makePair();
            V1cV2ctoV2cV1c.set(new BDDDomain[] {V1c,V2c},
                               new BDDDomain[] {V2c,V1c});
        } else {
            V1toV2 = bdd.makePair(V1, V2);
            V2toV1 = bdd.makePair(V2, V1);
            V1H1toV2H2 = bdd.makePair();
            V1H1toV2H2.set(new BDDDomain[] {V1,H1},
                           new BDDDomain[] {V2,H2});
            V2H2toV1H1 = bdd.makePair();
            V2H2toV1H1.set(new BDDDomain[] {V2,H2},
                           new BDDDomain[] {V1,H1});
        }
        
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
        if (CONTEXT_SENSITIVE) {
            V1set.andWith(V1c.set());
            V2set.andWith(V2c.set());
            H1set.andWith(H1c.set());
            H2set.andWith(H2c.set());
        }
        V1V2set = V1set.and(V2set);
        V1H1set = V1set.and(H1set);
        IMset = Iset.and(Mset);
        H1Fset = H1set.and(Fset);
        H2Fset = H2set.and(Fset);
        H1FH2set = H1Fset.and(H2set);
        T2Nset = T2set.and(Nset);
        MZset = Mset.and(Zset);
        
        A = bdd.zero();
        vP = bdd.zero();
        S = bdd.zero();
        L = bdd.zero();
        vT = bdd.zero();
        hT = bdd.zero();
        aT = bdd.zero();
        cha = bdd.zero();
        actual = bdd.zero();
        formal = bdd.zero();
        Iret = bdd.zero();
        Mret = bdd.zero();
        Ithr = bdd.zero();
        Mthr = bdd.zero();
        mI = bdd.zero();
        mV = bdd.zero();
        sync = bdd.zero();
        IE = bdd.zero();
        hP = bdd.zero();
        visited = bdd.zero();
        
        if (INCREMENTAL1) {
            old1_A = bdd.zero();
            old1_S = bdd.zero();
            old1_L = bdd.zero();
            old1_vP = bdd.zero();
            old1_hP = bdd.zero();
        }
        if (INCREMENTAL2) {
            old2_IE = bdd.zero();
            old2_visited = bdd.zero();
        }
        if (INCREMENTAL3) {
            old3_t3 = bdd.zero();
            old3_vP = bdd.zero();
            old3_t4 = bdd.zero();
            old3_hT = bdd.zero();
        }
        if (ADD_THREADS) {
            PrimordialClassLoader.getJavaLangThread().prepare();
            PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;").prepare();
        }
    }
    
    void addToVisited(BDD M_bdd) {
        if (TRACE_RELATIONS) out.println("Adding to visited: "+M_bdd.toStringWithDomains());
        visited.orWith(M_bdd.id());
    }
    
    void addToFormal(BDD M_bdd, int z, Node v) {
        BDD bdd1 = Z.ithVar(z);
        int V_i = Vmap.get(v);
        bdd1.andWith(V1.ithVar(V_i));
        bdd1.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to formal: "+bdd1.toStringWithDomains());
        formal.orWith(bdd1);
    }
    
    void addToIE(BDD V1V2context, BDD I_bdd, jq_Method target) {
        int M2_i = Mmap.get(target);
        BDD bdd1 = M.ithVar(M2_i);
        bdd1.andWith(I_bdd.id());
        if (CONTEXT_SENSITIVE) bdd1.andWith(V1V2context.id());
        if (TRACE_RELATIONS) out.println("Adding to IE: "+bdd1.toStringWithDomains());
        IE.orWith(bdd1);
    }
    
    void addToMI(BDD M_bdd, BDD I_bdd, jq_Method target) {
        int N_i = Nmap.get(target);
        BDD bdd1 = N.ithVar(N_i);
        bdd1.andWith(M_bdd.id());
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to mI: "+bdd1.toStringWithDomains());
        mI.orWith(bdd1);
    }
    
    void addToActual(BDD I_bdd, int z, Set s) {
        BDD bdd1 = bdd.zero();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            int V_i = Vmap.get(j.next());
            if (TRACE_RELATIONS) out.println("Adding to actual: "+bdd1.toStringWithDomains());
            bdd1.orWith(V2.ithVar(V_i));
        }
        bdd1.andWith(Z.ithVar(z));
        bdd1.andWith(I_bdd.id());
        actual.orWith(bdd1);
    }
    
    void addToIret(BDD I_bdd, Node v) {
        int V_i = Vmap.get(v);
        BDD bdd1 = V1.ithVar(V_i);
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Iret: "+bdd1.toStringWithDomains());
        Iret.orWith(bdd1);
    }
    
    void addToIthr(BDD I_bdd, Node v) {
        int V_i = Vmap.get(v);
        BDD bdd1 = V1.ithVar(V_i);
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Ithr: "+bdd1.toStringWithDomains());
        Ithr.orWith(bdd1);
    }
    
    void addToMV(BDD M_bdd, BDD V_bdd) {
        BDD bdd1 = M_bdd.id();
        bdd1.andWith(V_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to mV: "+bdd1.toStringWithDomains());
        mV.orWith(bdd1);
    }
    
    void addToMret(BDD M_bdd, Node v) {
        addToMret(M_bdd, Vmap.get(v));
    }
    
    void addToMret(BDD M_bdd, int V_i) {
        BDD bdd1 = V2.ithVar(V_i);
        bdd1.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Mret: "+bdd1.toStringWithDomains());
        Mret.orWith(bdd1);
    }
    
    void addToMthr(BDD M_bdd, int V_i) {
        BDD bdd1 = V2.ithVar(V_i);
        bdd1.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Mthr: "+bdd1.toStringWithDomains());
        Mthr.orWith(bdd1);
    }
    
    void addToVP(BDD V1H1context, Node p, int H_i) {
        int V1_i = Vmap.get(p);
        BDD bdd1 = V1.ithVar(V1_i);
        bdd1.andWith(H1.ithVar(H_i));
        if (CONTEXT_SENSITIVE) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToVP(BDD V1H1context, BDD V_bdd, Node h) {
        int H_i = Hmap.get(h);
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(V_bdd.id());
        if (CONTEXT_SENSITIVE) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToA(BDD V1V2context, int V1_i, int V2_i) {
        BDD V_bdd = V1.ithVar(V1_i);
        addToA(V1V2context, V_bdd, V2_i);
        V_bdd.free();
    }
    
    void addToA(BDD V1V2context, BDD V_bdd, int V2_i) {
        BDD bdd1 = V2.ithVar(V2_i);
        bdd1.andWith(V_bdd.id());
        bdd1.andWith(V1V2context.id());
        if (TRACE_RELATIONS) out.println("Adding to A: "+bdd1.toStringWithDomains());
        A.orWith(bdd1);
    }
    
    void addToS(BDD V1V2context, BDD V_bdd, jq_Field f, Collection c) {
        int F_i = Fmap.get(f);
        BDD F_bdd = F.ithVar(F_i);
        for (Iterator k = c.iterator(); k.hasNext(); ) {
            Node node2 = (Node) k.next();
            if (node2 instanceof ConcreteTypeNode ||
                node2 instanceof ConcreteObjectNode) {
                jq_Reference type = node2.getDeclaredType();
                if (type == null) {
                    if (TRACE) out.println("Skipping null constant.");
                    continue;
                }
            }
            int V2_i = Vmap.get(node2);
            BDD bdd1 = V2.ithVar(V2_i);
            bdd1.andWith(F_bdd.id());
            bdd1.andWith(V_bdd.id());
            if (CONTEXT_SENSITIVE) bdd1.andWith(V1V2context.id());
            if (TRACE_RELATIONS) out.println("Adding to S: "+bdd1.toStringWithDomains());
            S.orWith(bdd1);
        }
        F_bdd.free();
    }
    
    void addToL(BDD V1V2context, BDD V_bdd, jq_Field f, Collection c) {
        int F_i = Fmap.get(f);
        BDD F_bdd = F.ithVar(F_i);
        for (Iterator k = c.iterator(); k.hasNext(); ) {
            Node node2 = (Node) k.next();
            int V2_i = Vmap.get(node2);
            BDD bdd1 = V2.ithVar(V2_i);
            bdd1.andWith(F_bdd.id());
            bdd1.andWith(V_bdd.id());
            if (CONTEXT_SENSITIVE) bdd1.andWith(V1V2context.id());
            if (TRACE_RELATIONS) out.println("Adding to L: "+bdd1.toStringWithDomains());
            L.orWith(bdd1);
        }
        F_bdd.free();
    }
    
    void addToSync(Node n) {
        int V_i = Vmap.get(n);
        BDD bdd1 = V1.ithVar(V_i);
        if (TRACE_RELATIONS) out.println("Adding to sync: "+bdd1.toStringWithDomains());
        sync.orWith(bdd1);
    }
    
    BDD getVC(ProgramLocation mc, jq_Method callee) {
        Pair p = new Pair(LoadedCallGraph.mapCall(mc), callee);
        Range r_edge = vCnumbering.getEdge(p);
        Range r_caller = vCnumbering.getRange(mc.getMethod());
        if (r_edge == null) {
            out.println("Cannot find edge "+p);
            return bdd.one();
        }
        BDD context = buildContextMap(V2c,
                                      PathNumbering.toBigInt(r_caller.low),
                                      PathNumbering.toBigInt(r_caller.high),
                                      V1c,
                                      PathNumbering.toBigInt(r_edge.low),
                                      PathNumbering.toBigInt(r_edge.high));
        return context;
    }
    
    public static BDD buildContextMap(BDDDomain d1, BigInteger startD1, BigInteger endD1,
                                      BDDDomain d2, BigInteger startD2, BigInteger endD2) {
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
    
    public void visitMethod(jq_Method m) {
        if (visitedMethods.contains(m)) return;
        visitedMethods.add(m);
        
        if (TRACE) out.println("Visiting method "+m);
        m.getDeclaringClass().prepare();
        
        int M_i = Mmap.get(m);
        BDD M_bdd = M.ithVar(M_i);
        addToVisited(M_bdd);
        
        BDD V1V2context = null, V1H1context = null;
        if (CONTEXT_SENSITIVE) {
            Number n1 = vCnumbering.numberOfPathsTo(m);
            int bits = BigInteger.valueOf(n1.longValue()).bitLength();
            V1V2context = V1c.buildAdd(V2c, bits, 0L);
            V1V2context.andWith(V1c.varRange(0, n1.longValue()-1));
            V1H1context = (BDD) V1H1correspondence.get(m);
        }
        
        if (m.isSynchronized()) {
            //addToSync();
        }
        
        if (m.getBytecode() == null) {
            // todo: parameters passed into native methods.
            // build up 'Mret'
            jq_Type retType = m.getReturnType();
            if (retType instanceof jq_Reference) {
                Node node = UnknownTypeNode.get((jq_Reference) retType);
                addToMret(M_bdd, node);
                visitNode(V1V2context, null, node);
            }
            M_bdd.free();
            if (CONTEXT_SENSITIVE) {
                V1V2context.free();
                V1H1context.free();
            }
            return;
        }
        
        MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
        if (TRACE) out.println("Visiting method summary "+ms);
        
        addClassInitializer(ms.getMethod().getDeclaringClass());
        
        // build up 'formal'
        int nParams = ms.getNumOfParams();
        int offset = ms.getMethod().isStatic()?1:0;
        for (int i = 0; i < nParams; ++i) {
            Node node = ms.getParamNode(i);
            if (node == null) continue;
            int Z_i = i + offset;
            addToFormal(M_bdd, Z_i, node);
        }
        
        // build up 'mI', 'actual', 'Iret', 'Ithr'
        for (Iterator i = ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            if (TRACE) out.println("Visiting call site "+mc);
            int I_i = Imap.get(LoadedCallGraph.mapCall(mc));
            BDD I_bdd = I.ithVar(I_i);
            jq_Method target = mc.getTargetMethod();
            if (mc.isSingleTarget()) {
                BDD context = null;
                if (CONTEXT_SENSITIVE) {
                    context = getVC(mc, target);
                }
                addToIE(context, I_bdd, target);
                if (CONTEXT_SENSITIVE) {
                    context.free();
                }
            } else {
                addToMI(M_bdd, I_bdd, target);
            }
            
            if (target.isStatic())
                addClassInitializer(target.getDeclaringClass());
            
            if (target.isStatic()) {
                addToActual(I_bdd, 0, Collections.singleton(GlobalNode.GLOBAL));
                offset = 1;
            } else {
                offset = 0;
            }
            jq_Type[] params = mc.getParamTypes();
            for (int k = 0; k<params.length; ++k) {
                if (!params[k].isReferenceType()) continue;
                Set s = ms.getNodesThatCall(mc, k);
                addToActual(I_bdd, k+offset, s);
            }
            Node node = ms.getRVN(mc);
            if (node != null) {
                addToIret(I_bdd, node);
            }
            node = ms.getTEN(mc);
            if (node != null) {
                addToIthr(I_bdd, node);
            }
            I_bdd.free();
        }
        // build up 'mV', 'vP', 'S', 'L', 'Mret', 'Mthr'
        for (Iterator i = ms.nodeIterator(); i.hasNext(); ) {
            Node node = (Node) i.next();
            
            if (node instanceof ConcreteTypeNode ||
                node instanceof ConcreteObjectNode) {
                jq_Reference type = node.getDeclaredType();
                if (type == null) {
                    if (TRACE) out.println("Skipping null constant.");
                    continue;
                }
            }
            
            int V_i = Vmap.get(node);
            BDD V_bdd = V1.ithVar(V_i);
            addToMV(M_bdd, V_bdd);
            
            if (ms.getReturned().contains(node)) {
                addToMret(M_bdd, V_i);
            }
            
            if (ms.getThrown().contains(node)) {
                addToMthr(M_bdd, V_i);
            }
            
            visitNode(V1V2context, V1H1context, node);
        }
        if (CONTEXT_SENSITIVE) {
            V1V2context.free();
            V1H1context.free();
        }
    }
    
    public void visitNode(BDD V1V2context, BDD V1H1context, Node node) {
        if (TRACE) out.println("Visiting node "+node);
        
        if (node instanceof ConcreteTypeNode ||
            node instanceof ConcreteObjectNode) {
            jq_Reference type = node.getDeclaredType();
            if (type == null) {
                if (TRACE) out.println("Skipping null constant.");
                return;
            }
        }
        
        int V_i = Vmap.get(node);
        BDD V_bdd = V1.ithVar(V_i);
        
        if (node instanceof ConcreteTypeNode) {
            addToVP(V1H1context, V_bdd, node);
        } else if (node instanceof ConcreteObjectNode ||
                   node instanceof UnknownTypeNode ||
                   node == GlobalNode.GLOBAL) {
            BDD context = bdd.one();
            addToVP(context, V_bdd, node);
            context.free();
        } else if (node instanceof GlobalNode) {
            int V2_i = Vmap.get(GlobalNode.GLOBAL);
            BDD context = bdd.one();
            addToA(context, V_bdd, V2_i);
            addToA(context, V2_i, V_i);
            context.free();
        }
        
        for (Iterator j = node.getAllEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Collection c;
            if (e.getValue() instanceof Collection)
                c = (Collection) e.getValue();
            else
                c = Collections.singleton(e.getValue());
            addToS(V1V2context, V_bdd, f, c);
        }
        
        for (Iterator j = node.getAccessPathEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Collection c;
            if (e.getValue() instanceof Collection)
                c = (Collection) e.getValue();
            else
                c = Collections.singleton(e.getValue());
            addToL(V1V2context, V_bdd, f, c);
            if (node instanceof GlobalNode)
                addClassInitializer(f.getDeclaringClass());
        }
    }
    
    void addToVT(int V_i, jq_Reference type) {
        BDD bdd1 = V1.ithVar(V_i);
        int T_i = Tmap.get(type);
        bdd1.andWith(T1.ithVar(T_i));
        if (TRACE_RELATIONS) out.println("Adding to vT: "+bdd1.toStringWithDomains());
        vT.orWith(bdd1);
    }
    
    void addToHT(int H_i, jq_Reference type) {
        BDD bdd1 = H1.ithVar(H_i);
        int T_i = Tmap.get(type);
        bdd1.andWith(T2.ithVar(T_i));
        if (TRACE_RELATIONS) out.println("Adding to hT: "+bdd1.toStringWithDomains());
        hT.orWith(bdd1);
    }
    
    void addToAT(BDD T1_bdd, int T2_i) {
        BDD bdd1 = T2.ithVar(T2_i);
        bdd1.andWith(T1_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to aT: "+bdd1.toStringWithDomains());
        aT.orWith(bdd1);
    }
    
    void addToCHA(BDD T_bdd, int N_i, jq_Method m) {
        BDD bdd1 = N.ithVar(N_i);
        int M_i = Mmap.get(m);
        bdd1.andWith(M.ithVar(M_i));
        bdd1.andWith(T_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to cha: "+bdd1.toStringWithDomains());
        cha.orWith(bdd1);
    }
    
    int last_V = 0;
    int last_H = 0;
    int last_T = 0;
    int last_N = 0;
    
    public void buildTypes() {
        // build up 'vT'
        for (int V_i = last_V; V_i < Vmap.size(); ++V_i) {
            Node n = (Node) Vmap.get(V_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) type.prepare();
            addToVT(V_i, type);
        }
        
        // build up 'hT', and identify clinit, thread run, finalizers.
        for (int H_i = last_H; H_i < Hmap.size(); ++H_i) {
            Node n = (Node) Hmap.get(H_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) {
                type.prepare();
                if (n instanceof ConcreteTypeNode && type instanceof jq_Class) {
                    addClassInitializer((jq_Class) type);
                    addFinalizer((jq_Class) type);
                }
                if (ADD_THREADS &&
                    (type.isSubtypeOf(PrimordialClassLoader.getJavaLangThread()) ||
                     type.isSubtypeOf(PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;")))) {
                    addThreadRun(H_i, (jq_Class) type);
                }
            }
            addToHT(H_i, type);
        }
        
        // build up 'aT'
        for (int T1_i = 0; T1_i < Tmap.size(); ++T1_i) {
            jq_Reference t1 = (jq_Reference) Tmap.get(T1_i);
            int start = (T1_i < last_T)?last_T:0;
            BDD T1_bdd = T1.ithVar(T1_i);
            for (int T2_i = start; T2_i < Tmap.size(); ++T2_i) {
                jq_Reference t2 = (jq_Reference) Tmap.get(T2_i);
                if (t2 == null || (t1 != null && t2.isSubtypeOf(t1))) {
                    addToAT(T1_bdd, T2_i);
                }
            }
            T1_bdd.free();
        }
        
        // make type filter
        if (FILTER_TYPE) {
            BDD t1 = vT.relprod(aT, T1set); // V1xT1 x T1xT2 = V1xT2
            filter = t1.relprod(hT, T2set); // V1xT2 x H1xT2 = V1xH1
            t1.free();
        } else {
            filter = bdd.one();
        }
        
        // build up 'cha'
        for (int T_i = 0; T_i < Tmap.size(); ++T_i) {
            jq_Reference t = (jq_Reference) Tmap.get(T_i);
            BDD T_bdd = T2.ithVar(T_i);
            int start = (T_i < last_T)?last_N:0;
            for (int N_i = start; N_i < Nmap.size(); ++N_i) {
                jq_Method n = (jq_Method) Nmap.get(N_i);
                n.getDeclaringClass().prepare();
                jq_Method m;
                if (n.isStatic()) {
                    if (t != null) continue;
                    m = n;
                } else {
                    if (t == null || !t.isSubtypeOf(n.getDeclaringClass())) continue;
                    m = t.getVirtualMethod(n.getNameAndDesc());
                }
                if (m == null) continue;
                addToCHA(T_bdd, N_i, m);
            }
            T_bdd.free();
        }
        last_V = Vmap.size();
        last_H = Hmap.size();
        last_T = Tmap.size();
        last_N = Nmap.size();
    }
    
    public void addClassInitializer(jq_Class c) {
        if (!ADD_CLINIT) return;
        jq_Method m = c.getClassInitializer();
        if (m != null) {
            visitMethod(m);
            roots.add(m);
        }
    }
    
    jq_NameAndDesc finalizer_method = new jq_NameAndDesc("finalize", "()V");
    public void addFinalizer(jq_Class c) {
        if (!ADD_FINALIZERS) return;
        jq_Method m = c.getVirtualMethod(finalizer_method);
        if (m != null) {
            visitMethod(m);
            roots.add(m);
        }
    }
    
    static jq_NameAndDesc run_method = new jq_NameAndDesc("run", "()V");
    public void addThreadRun(int H_i, jq_Class c) {
        if (!ADD_THREADS) return;
        jq_Method m = c.getVirtualMethod(run_method);
        if (m != null && m.getBytecode() != null) {
            visitMethod(m);
            roots.add(m);
            Node p = MethodSummary.getSummary(CodeCache.getCode(m)).getParamNode(0);
            Node h = (Node) Hmap.get(H_i);
            BDD context = null;
            if (CONTEXT_SENSITIVE) {
                int context_i = getThreadRunIndex(m, h);
                System.out.println("Thread "+h+" index "+context_i);
                context = H1c.ithVar(context_i);
            }
            addToVP(context, p, H_i);
            if (CONTEXT_SENSITIVE) {
                context.free();
            }
        }
    }
    
    public void solvePointsTo() {
        if (INCREMENTAL1) {
            solvePointsTo_incremental();
            return;
        }
        BDD old_vP = vP.id();
        BDD old_hP = hP.id();
        for (int outer = 1; ; ++outer) {
            for (int inner = 1; ; ++inner) {
                old_vP = vP.id();
                
                // Rule 1
                BDD t1 = vP.replace(V1toV2); // V2xH1
                BDD t2 = A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
                t1.free();
                t2.andWith(filter.id());
                vP.orWith(t2);
                if (TRACE_SOLVER) out.println("Inner #"+inner+": vP "+old_vP.satCount(V1H1set)+" -> "+vP.satCount(V1H1set));
                
                boolean done = vP.equals(old_vP); 
                old_vP.free();
                if (done) break;
            }
            
            // Rule 2
            BDD t3 = S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
            BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
            t3.free(); t4.free();
            hP.orWith(t5);

            if (TRACE_SOLVER) out.println("Outer #"+outer+": hP "+old_hP.satCount(H1FH2set)+" -> "+hP.satCount(H1FH2set));
            
            boolean done = hP.equals(old_hP); 
            old_hP.free();
            if (done) break;
            old_hP = hP.id();
            
            // Rule 3
            BDD t6 = L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
            t6.free();
            t7.replaceWith(V2H2toV1H1); // V1xH1
            t7.andWith(filter.id());
            vP.orWith(t7);
        }
    }
    
    BDD old1_A;
    BDD old1_S;
    BDD old1_L;
    BDD old1_vP;
    BDD old1_hP;
    
    public void solvePointsTo_incremental() {
        // handle new A
        BDD new_A = A.apply(old1_A, BDDFactory.diff);
        old1_A.free();
        if (!new_A.isZero()) {
            if (TRACE_SOLVER) out.print("Handling new A: "+new_A.satCount(V1V2set));
            BDD t1 = vP.replace(V1toV2);
            BDD t2 = new_A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
            new_A.free(); t1.free();
            t2.andWith(filter.id());
            if (TRACE_SOLVER) out.print(" vP "+vP.satCount(V1H1set));
            vP.orWith(t2);
            if (TRACE_SOLVER) out.println(" --> "+vP.satCount(V1H1set));
        }
        old1_A = A.id();
        
        // handle new S
        BDD new_S = S.apply(old1_S, BDDFactory.diff);
        old1_S.free();
        if (!new_S.isZero()) {
            if (TRACE_SOLVER) out.print("Handling new S: "+new_S.satCount(V1V2set.and(Fset)));
            BDD t3 = new_S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            new_S.free();
            BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
            BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
            t3.free(); t4.free();
            if (TRACE_SOLVER) out.print(" hP "+hP.satCount(H1FH2set));
            hP.orWith(t5);
            if (TRACE_SOLVER) out.println(" --> "+hP.satCount(H1FH2set));
        }
        old1_S = S.id();
        
        // handle new L
        BDD new_L = L.apply(old1_L, BDDFactory.diff);
        old1_L.free();
        if (!new_L.isZero()) {
            if (TRACE_SOLVER) out.print("Handling new L: "+new_L.satCount(V1V2set.and(Fset)));
            BDD t6 = new_L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
            t6.free();
            t7.replaceWith(V2H2toV1H1); // V1xH1
            t7.andWith(filter.id());
            if (TRACE_SOLVER) out.print(" vP "+vP.satCount(V1H1set));
            vP.orWith(t7);
            if (TRACE_SOLVER) out.println(" --> "+vP.satCount(V1H1set));
        }
        old1_L = S.id();
        
        for (int outer = 1; ; ++outer) {
            BDD new_vP_inner = vP.apply(old1_vP, BDDFactory.diff);
            for (int inner = 1; !new_vP_inner.isZero(); ++inner) {
                if (TRACE_SOLVER)
                    out.print("Inner #"+inner+": new vP "+new_vP_inner.satCount(V1H1set));
                
                // Rule 1
                BDD t1 = new_vP_inner.replace(V1toV2); // V2xH1
                new_vP_inner.free();
                BDD t2 = A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
                t1.free();
                t2.andWith(filter.id());
                
                BDD old_vP_inner = vP.id();
                vP.orWith(t2);
                if (TRACE_SOLVER)
                    out.println(", vP "+old_vP_inner.satCount(V1H1set)+
                                " -> "+vP.satCount(V1H1set));
                new_vP_inner = vP.apply(old_vP_inner, BDDFactory.diff);
                old_vP_inner.free();
            }
            
            BDD new_vP = vP.apply(old1_vP, BDDFactory.diff);
            old1_vP.free();
            
            if (TRACE_SOLVER)
                out.print("Outer #"+outer+": new vP "+new_vP.satCount(V1H1set));
            
            {
                // Rule 2
                BDD t3 = S.relprod(new_vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
                BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
                t3.free(); t4.free();
                hP.orWith(t5);
            }
            {
                // Rule 2
                BDD t3 = S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t4 = new_vP.replace(V1H1toV2H2); // V2xH2
                BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
                t3.free(); t4.free();
                hP.orWith(t5);
            }

            if (TRACE_SOLVER)
                out.println(", hP "+old1_hP.satCount(H1FH2set)+" -> "+hP.satCount(H1FH2set));
            
            old1_vP = vP.id();
            
            BDD new_hP = hP.apply(old1_hP, BDDFactory.diff);
            if (new_hP.isZero()) break;
            old1_hP = hP.id();
            
            if (TRACE_SOLVER)
                out.print("        : new hP "+new_hP.satCount(H1FH2set));
            
            {
                // Rule 3
                BDD t6 = L.relprod(new_vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                t7.andWith(filter.id());
                vP.orWith(t7);
            }
            {
                // Rule 3
                BDD t6 = L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t7 = t6.relprod(new_hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                t7.andWith(filter.id());
                vP.orWith(t7);
            }
            if (TRACE_SOLVER)
                out.println(", vP "+old1_vP.satCount(V1H1set)+
                            " -> "+vP.satCount(V1H1set));
        }
    }
    
    public void bindInvocations() {
        if (INCREMENTAL3) {
            bindInvocations_incremental();
            return;
        }
        BDD t1 = actual.restrict(Z.ithVar(0)); // IxV2
        t1.replaceWith(V2toV1); // IxV1
        if (TRACE_BIND) out.println("t1: "+t1.satCount(Iset.and(V1set)));
        if (TRACE_BIND) out.println("t1: "+t1.toStringWithDomains(TS));
        BDD t2 = mI.exist(Mset); // IxN
        if (TRACE_BIND) out.println("t2: "+t2.satCount(Iset.and(Nset)));
        if (TRACE_BIND) out.println("t2: "+t2.toStringWithDomains(TS));
        BDD t3 = t1.and(t2); // IxV1 & IxN = IxV1xN 
        if (TRACE_BIND) out.println("t3: "+t3.satCount(Iset.and(Nset).and(V1set)));
        if (TRACE_BIND) out.println("t3: "+t3.toStringWithDomains(TS));
        t1.free(); t2.free();
        BDD t4 = t3.relprod(vP, V1set); // IxV1xN x V1xH1 = IxH1xN
        if (TRACE_BIND) out.println("t4: "+t4.satCount(Iset.and(Nset).and(H1set)));
        if (TRACE_BIND) out.println("t4: "+t4.toStringWithDomains(TS));
        t3.free();
        BDD t5 = t4.relprod(hT, H1set); // IxH1xN x H1xT2 = IxT2xN
        if (TRACE_BIND) out.println("t5: "+t5.satCount(Iset.and(Nset).and(T2set)));
        if (TRACE_BIND) out.println("t5: "+t5.toStringWithDomains(TS));
        t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // IxT2xN x T2xNxM = IxM
        if (TRACE_BIND) out.println("t6: "+t6.satCount(Iset.and(Mset)));
        if (TRACE_BIND) out.println("t6: "+t6.toStringWithDomains(TS));
        t5.free();
        if (CONTEXT_SENSITIVE)
            t6.andWith(IEc.id());
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        IE.orWith(t6);
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
    }
    
    BDD old3_t3;
    BDD old3_vP;
    BDD old3_t4;
    BDD old3_hT;
    
    public void bindInvocations_incremental() {
        BDD t1 = actual.restrict(Z.ithVar(0)); // IxV2
        t1.replaceWith(V2toV1); // IxV1
        BDD t2 = mI.exist(Mset); // IxN
        BDD t3 = t1.and(t2); // IxV1 & IxN = IxV1xN 
        t1.free(); t2.free();
        BDD new_t3 = t3.apply(old3_t3, BDDFactory.diff);
        old3_t3.free();
        BDD new_vP = vP.apply(old3_vP, BDDFactory.diff);
        old3_vP.free();
        BDD t4 = t3.relprod(new_vP, V1set); // IxV1xN x V1xH1 = IxH1xN
        new_vP.free();
        old3_t3 = t3;
        t4.orWith(new_t3.relprod(vP, V1set));
        new_t3.free();
        BDD new_t4 = t4.apply(old3_t4, BDDFactory.diff);
        old3_t4.free();
        BDD new_hT = hT.apply(old3_hT, BDDFactory.diff);
        old3_hT.free();
        BDD t5 = t4.relprod(new_hT, H1set); // IxH1xN x H1xT2 = IxT2xN
        new_hT.free();
        old3_t4 = t4;
        t5.orWith(new_t4.relprod(hT, H1set));
        new_t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // IxT2xN x T2xNxM = V2cxIxV1cxM
        t5.free();
        if (CONTEXT_SENSITIVE)
            t6.andWith(IEc.id());
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        IE.orWith(t6);
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        old3_vP = vP.id();
        old3_hT = hT.id();
    }
    
    public boolean handleNewTargets() {
        if (TRACE_SOLVER) out.println("Handling new target methods...");
        BDD targets = IE.exist(Iset);
        if (CONTEXT_SENSITIVE) {
            BDD t2 = targets.exist(V1cV2cset);
            targets.free(); targets = t2;
        }
        targets.applyWith(visited.id(), BDDFactory.diff);
        if (targets.isZero()) return false;
        if (TRACE) out.println("New target methods: "+targets.satCount(Mset));
        while (!targets.isZero()) {
            BDD target = targets.satOneSet(Mset, bdd.zero());
            int M_i = target.scanVar(M);
            jq_Method method = (jq_Method) Mmap.get(M_i);
            if (TRACE) out.println("New target method: "+method);
            visitMethod(method);
            targets.applyWith(target, BDDFactory.diff);
        }
        return true;
    }
    
    BDD old2_IE;
    BDD old2_visited;
    
    public void bindParameters() {
        if (INCREMENTAL2) {
            bindParameters_incremental();
            return;
        }
        
        if (TRACE_SOLVER) out.println("Binding parameters...");
        
        BDD t1 = IE.relprod(actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD IEr;
        if (CONTEXT_SENSITIVE) IEr = IE.replace(V1cV2ctoV2cV1c);
        else IEr = IE;
        BDD t3 = IEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = IEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (CONTEXT_SENSITIVE) IEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
    }
    
    public void bindParameters_incremental() {
        if (TRACE_SOLVER) out.println("Binding parameters...");
        
        BDD new_IE = IE.apply(old2_IE, BDDFactory.diff);
        BDD new_visited = visited.apply(old2_visited, BDDFactory.diff);
        // add in any old edges targetting newly-visited methods, because the
        // argument/retval binding doesn't occur until the method has been visited.
        new_IE.orWith(old2_IE.and(new_visited));
        old2_IE.free();
        old2_visited.free();
        new_visited.free();
        
        BDD t1 = new_IE.relprod(actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD new_IEr;
        if (CONTEXT_SENSITIVE) new_IEr = new_IE.replace(V1cV2ctoV2cV1c);
        else new_IEr = new_IE;
        BDD t3 = new_IEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = new_IEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (CONTEXT_SENSITIVE) new_IEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
        new_IE.free();
        old2_IE = IE.id();
        old2_visited = visited.id();
    }
    
    public void assumeKnownCallGraph() {
        if (IEc != null) IE = IEc;
        handleNewTargets();
        buildTypes();
        bindParameters();
        solvePointsTo();
    }
    
    public void iterate() {
        BDD IE_old = IE.id();
        boolean change;
        for (int major = 1; ; ++major) {
            change = false;
            
            out.println("Discovering call graph, iteration "+major+": "+visitedMethods.size()+" methods.");
            long time = System.currentTimeMillis();
            buildTypes();
            solvePointsTo();
            bindInvocations();
            if (handleNewTargets())
                change = true;
            if (!change && IE.equals(IE_old)) {
                if (TRACE_SOLVER) out.println("Finished after "+major+" iterations.");
                break;
            }
            IE_old.free(); IE_old = IE.id();
            bindParameters();
            if (TRACE_SOLVER)
                out.println("Time spent: "+(System.currentTimeMillis()-time)/1000.);
        }
    }
    
    public void numberPaths(CallGraph cg) {
        System.out.print("Counting size of call graph...");
        long time = System.currentTimeMillis();
        vCnumbering = countCallGraph(cg);
        hCnumbering = countHeapNumbering(cg);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
    }
    
    static CallGraph loadCallGraph(Collection roots) {
        if (new File(initialCallgraphFileName).exists()) {
            try {
                System.out.print("Loading initial call graph...");
                long time = System.currentTimeMillis();
                CallGraph cg = new LoadedCallGraph(initialCallgraphFileName);
                time = System.currentTimeMillis() - time;
                System.out.println("done. ("+time/1000.+" seconds)");
                if (cg.getRoots().containsAll(roots)) {
                    roots = cg.getRoots();
                    //LOADED_CALLGRAPH = true;
                    return cg;
                } else {
                    System.out.println("Call graph doesn't match named class, rebuilding...");
                    cg = null;
                }
            } catch (IOException x) {
                x.printStackTrace();
            }
        }
        return null;
    }
    
    public void run(CallGraph cg, Collection rootMethods) throws IOException {
        if (cg != null) {
            numberPaths(cg);
        }
        
        initialize();
        roots.addAll(rootMethods);
        
        if (cg != null) {
            System.out.print("Calculating call graph relation...");
            long time = System.currentTimeMillis();
            calculateIEc(cg);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
            
            if (CONTEXT_SENSITIVE) {
                System.out.print("Building var-heap context correspondence...");
                time = System.currentTimeMillis();
                buildVarHeapCorrespondence(cg);
                time = System.currentTimeMillis() - time;
                System.out.println("done. ("+time/1000.+" seconds)");
            }
        }
        
        long time = System.currentTimeMillis();
        
        GlobalNode.GLOBAL.addDefaultStatics();
        BDD context = null;
        if (CONTEXT_SENSITIVE) {
            context = bdd.one();
        }
        visitNode(context, context, GlobalNode.GLOBAL);
        for (Iterator i = ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            visitNode(context, context, (ConcreteObjectNode) i.next());
        }
        if (CONTEXT_SENSITIVE) {
            context.free();
        }
        
        for (Iterator i = roots.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            visitMethod(m);
        }
        
        if (DISCOVER_CALL_GRAPH) {
            iterate();
        } else {
            assumeKnownCallGraph();
        }
        
        System.out.println("Time spent solving: "+(System.currentTimeMillis()-time)/1000.);

        printSizes();
        
        System.out.println("Writing results...");
        time = System.currentTimeMillis();
        dumpResults(resultsFileName);
        System.out.println("Time spent writing: "+(System.currentTimeMillis()-time)/1000.);
    }
    
    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        
        Collection rootMethods = Arrays.asList(c.getDeclaredStaticMethods());
        if (args.length > 1) {
            for (Iterator i = rootMethods.iterator(); i.hasNext(); ) {
                jq_Method sm = (jq_Method) i.next();
                if (args[1].equals(sm.getName().toString())) {
                    rootMethods = Collections.singleton(sm);
                    break;
                }
            }
        }
        
        PA dis = new PA();
        CallGraph cg = null;
        if (dis.CONTEXT_SENSITIVE || !dis.DISCOVER_CALL_GRAPH) {
            cg = loadCallGraph(rootMethods);
            if (cg == null) {
                if (dis.CONTEXT_SENSITIVE) {
                    System.out.println("Discovering call graph first...");
                    dis.CONTEXT_SENSITIVE = false;
                    dis.DISCOVER_CALL_GRAPH = true;
                    dis.run(cg, rootMethods);
                    System.out.println("Finished discovering call graph.");
                    dis = new PA();
                    cg = loadCallGraph(rootMethods);
                    rootMethods = cg.getRoots();
                } else if (!dis.DISCOVER_CALL_GRAPH) {
                    System.out.println("Call graph doesn't exist yet, so turning on call graph discovery.");
                    dis.DISCOVER_CALL_GRAPH = true;
                }
            } else {
                rootMethods = cg.getRoots();
            }
        }
        dis.run(cg, rootMethods);
    }
    
    public void printSizes() {
        System.out.println("V = "+Vmap.size()+", bits = "+
                           BigInteger.valueOf(Vmap.size()).bitLength());
        System.out.println("I = "+Imap.size()+", bits = "+
                           BigInteger.valueOf(Imap.size()).bitLength());
        System.out.println("H = "+Hmap.size()+", bits = "+
                           BigInteger.valueOf(Hmap.size()).bitLength());
        System.out.println("F = "+Fmap.size()+", bits = "+
                           BigInteger.valueOf(Fmap.size()).bitLength());
        System.out.println("T = "+Tmap.size()+", bits = "+
                           BigInteger.valueOf(Tmap.size()).bitLength());
        System.out.println("N = "+Nmap.size()+", bits = "+
                           BigInteger.valueOf(Nmap.size()).bitLength());
        System.out.println("M = "+Mmap.size()+", bits = "+
                           BigInteger.valueOf(Mmap.size()).bitLength());
    }
    
    ToString TS = new ToString();
    
    public class ToString extends BDD.BDDToString {
        public String elementName(int i, long j) {
            switch (i) {
                case 0: // fallthrough
                case 1: return Vmap.get((int)j).toString();
                case 2: return Imap.get((int)j).toString();
                case 3: // fallthrough
                case 4: return Hmap.get((int)j).toString();
                case 5: return Long.toString(j);
                case 6: return ""+Fmap.get((int)j);
                case 7: // fallthrough
                case 8: return ""+Tmap.get((int)j);
                case 9: return Nmap.get((int)j).toString();
                case 10: return Mmap.get((int)j).toString();
                default: return "??";
            }
        }
    }
   
    private void dumpCallGraphAsDot(CallGraph callgraph, String dotFileName) throws IOException {
	DataOutputStream dos = new DataOutputStream(new FileOutputStream(dotFileName));
	countCallGraph(callgraph).dotGraph(dos);
	dos.close();
    }

    public void dumpResults(String dumpfilename) throws IOException {
        
        //CallGraph callgraph = CallGraph.makeCallGraph(roots, new PACallTargetMap());
        CallGraph callgraph = new CachedCallGraph(new PACallGraph(this));
        //CallGraph callgraph = callGraph;
        DataOutputStream dos;
        dos = new DataOutputStream(new FileOutputStream(callgraphFileName));
        LoadedCallGraph.write(callgraph, dos);
        dos.close();

        if (DUMP_DOTGRAPH)
            dumpCallGraphAsDot(callgraph, callgraphFileName + ".dot");
        
        bdd.save(dumpfilename+".A", A);
        bdd.save(dumpfilename+".vP", vP);
        bdd.save(dumpfilename+".S", S);
        bdd.save(dumpfilename+".L", L);
        bdd.save(dumpfilename+".vT", vT);
        bdd.save(dumpfilename+".hT", hT);
        bdd.save(dumpfilename+".aT", aT);
        bdd.save(dumpfilename+".cha", cha);
        bdd.save(dumpfilename+".actual", actual);
        bdd.save(dumpfilename+".formal", formal);
        bdd.save(dumpfilename+".Iret", Iret);
        bdd.save(dumpfilename+".Mret", Mret);
        bdd.save(dumpfilename+".Ithr", Ithr);
        bdd.save(dumpfilename+".Mthr", Mthr);
        bdd.save(dumpfilename+".mI", mI);
        bdd.save(dumpfilename+".mV", mV);
        
        bdd.save(dumpfilename+".hP", hP);
        bdd.save(dumpfilename+".IE", IE);
        bdd.save(dumpfilename+".filter", filter);
        if (IEc != null) bdd.save(dumpfilename+".IEc", IEc);
        bdd.save(dumpfilename+".visited", visited);
        
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
        dumpConfig(dos);
        dos.close();
        
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Vmap"));
        Vmap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Imap"));
        Imap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Hmap"));
        Hmap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Fmap"));
        Fmap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Tmap"));
        Tmap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Nmap"));
        Nmap.dump(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Mmap"));
        Mmap.dump(dos);
        dos.close();
    }

    private void dumpConfig(DataOutput out) throws IOException {
        out.writeBytes("V="+V_BITS+"\n");
        out.writeBytes("I="+I_BITS+"\n");
        out.writeBytes("H="+H_BITS+"\n");
        out.writeBytes("Z="+Z_BITS+"\n");
        out.writeBytes("F="+F_BITS+"\n");
        out.writeBytes("T="+T_BITS+"\n");
        out.writeBytes("N="+N_BITS+"\n");
        out.writeBytes("M="+M_BITS+"\n");
        out.writeBytes("VC="+VC_BITS+"\n");
        out.writeBytes("HC="+HC_BITS+"\n");
        out.writeBytes("Order="+varorder+"\n");
        out.writeBytes("Reverse="+reverseLocal+"\n");
    }
    
    public static class ThreadRootMap extends AbstractMap {
        Map map;
        ThreadRootMap(Map s) {
            map = s;
        }
        public Object get(Object o) {
            Set s = (Set) map.get(o);
            if (s == null) return new Integer(0);
            return new Integer(s.size()-1);
        }
        /* (non-Javadoc)
         * @see java.util.AbstractMap#entrySet()
         */
        public Set entrySet() {
            throw new UnsupportedOperationException();
        }
    }
    
    static Map thread_runs = new HashMap();
    
    public static int getThreadRunIndex(jq_Method m, Node n) {
        Set s = (Set) thread_runs.get(m);
        if (s != null) {
            Iterator i = s.iterator();
            for (int k = 0; i.hasNext(); ++k) {
                if (i.next() == n) return k;
            }
        }
        return 0;
    }
    
    public PathNumbering countCallGraph(CallGraph cg) {
        jq_Class jlt = PrimordialClassLoader.getJavaLangThread();
        jq_Class jlr = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;");
        
        Set fields = new HashSet();
        Set classes = new HashSet();
        int vars = 0, heaps = 0, bcodes = 0, methods = 0, calls = 0;
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            ++methods;
            if (m.getBytecode() == null) continue;
            bcodes += m.getBytecode().length;
            MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
            for (Iterator j = ms.nodeIterator(); j.hasNext(); ) {
                Node n = (Node) j.next();
                ++vars;
                if (n instanceof ConcreteTypeNode ||
                    n instanceof UnknownTypeNode ||
                    n instanceof ConcreteObjectNode) {
                    ++heaps;
                    jq_Reference type = n.getDeclaredType(); 
                    if (type != null) {
                        type.prepare();
                        if (type.isSubtypeOf(jlt) ||
                            type.isSubtypeOf(jlr)) {
                            jq_Method rm = type.getVirtualMethod(run_method);
                            Set s = (Set) thread_runs.get(rm);
                            if (s == null) thread_runs.put(rm, s = new HashSet());
                            s.add(n);
                        }
                    }
                }
                fields.addAll(n.getAccessPathEdgeFields());
                fields.addAll(n.getNonEscapingEdgeFields());
                if (n instanceof GlobalNode) continue;
                jq_Reference r = (jq_Reference) n.getDeclaredType();
                classes.add(r);
            }
            calls += ms.getCalls().size();
        }
        System.out.println();
        System.out.println("Methods="+methods+" Bytecodes="+bcodes+" Call sites="+calls);
        PathNumbering pn = new PathNumbering();
        System.out.println("Thread runs="+thread_runs);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        System.out.println("Vars="+vars+" Heaps="+heaps+" Classes="+classes.size()+" Fields="+fields.size()+" Paths="+paths);
        V_BITS = BigInteger.valueOf(vars+256).bitLength();
        I_BITS = BigInteger.valueOf(calls).bitLength();
        H_BITS = BigInteger.valueOf(heaps+256).bitLength();
        F_BITS = BigInteger.valueOf(fields.size()+64).bitLength();
        T_BITS = BigInteger.valueOf(classes.size()+64).bitLength();
        N_BITS = I_BITS;
        M_BITS = BigInteger.valueOf(methods).bitLength() + 1;
        VC_BITS = paths.bitLength();
        VC_BITS = Math.min(60, VC_BITS);
        System.out.println(" V="+V_BITS+" I="+I_BITS+" H="+H_BITS+
                           " F="+F_BITS+" T="+T_BITS+" N="+N_BITS+
                           " M="+M_BITS+" VC="+VC_BITS);
        return pn;
    }

    public final HeapPathSelector heapPathSelector = new HeapPathSelector();
    
    public class HeapPathSelector implements Selector {

        jq_Class collection_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/Collection;");
        jq_Class map_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/Map;");
        HeapPathSelector() {
            collection_class.prepare();
            map_class.prepare();
        }
        
        /* (non-Javadoc)
         * @see Util.Graphs.PathNumbering.Selector#isImportant(Util.Graphs.SCComponent, Util.Graphs.SCComponent)
         */
        public boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num) {
            if (num.bitLength() > MAX_HC_BITS) return false;
            Set s = scc2.nodeSet();
            Iterator i = s.iterator();
            Object o = i.next();
            if (i.hasNext()) return false;
            if (o instanceof ProgramLocation) return true;
            jq_Method m = (jq_Method) o;
            if (!m.getReturnType().isReferenceType()) return false;
            if (m.getBytecode() == null) return false;
            MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
            for (i = ms.getReturned().iterator(); i.hasNext(); ) {
                Node n = (Node) i.next();
                if (!(n instanceof ConcreteTypeNode)) {
                    //return false;
                }
                jq_Reference type = n.getDeclaredType();
                if (type == null) {
                    return false;
                }
                type.prepare();
                //if (!type.isSubtypeOf(collection_class) &&
                //    !type.isSubtypeOf(map_class))
                //    return false;
            }
            return true;
        }
    }

    public PathNumbering countHeapNumbering(CallGraph cg) {
        PathNumbering pn = new PathNumbering(heapPathSelector);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        HC_BITS = paths.bitLength();
        System.out.println("Heap context bits="+HC_BITS);
        return pn;
    }
    
    void calculateIEc(CallGraph cg) {
        IEc = bdd.zero();
        for (Iterator i = cg.getAllCallSites().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            mc = LoadedCallGraph.mapCall(mc);
            int I_i = Imap.get(mc);
            for (Iterator j = cg.getTargetMethods(mc).iterator(); j.hasNext(); ) {
                jq_Method callee = (jq_Method) j.next();
                int M_i = Mmap.get(callee);
                BDD context;
                if (CONTEXT_SENSITIVE) {
                    Pair p = new Pair(mc, callee);
                    Range r_edge = vCnumbering.getEdge(p);
                    Range r_caller = vCnumbering.getRange(mc.getMethod());
                    context = buildContextMap(V2c,
                                              PathNumbering.toBigInt(r_caller.low),
                                              PathNumbering.toBigInt(r_caller.high),
                                              V1c,
                                              PathNumbering.toBigInt(r_edge.low),
                                              PathNumbering.toBigInt(r_edge.high));
                } else {
                    context = bdd.one();
                }
                context.andWith(I.ithVar(I_i));
                context.andWith(M.ithVar(M_i));
                IEc.orWith(context);
            }
        }
    }
    
    Map V1H1correspondence;
    public void buildVarHeapCorrespondence(CallGraph cg) {
        BDDPairing V2cH2ctoV1cH1c = bdd.makePair();
        V2cH2ctoV1cH1c.set(new BDDDomain[] {V2c, H2c}, new BDDDomain[] {V1c, H1c});
        BDDPairing V2ctoV1c = bdd.makePair(V2c, V1c);
        BDDPairing H2ctoH1c = bdd.makePair(H2c, H1c);
        BDD V1cset = V1c.set();
        BDD H1cset = H1c.set();
        
        V1H1correspondence = new HashMap();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            jq_Method root = (jq_Method) i.next();
            Number n1 = vCnumbering.numberOfPathsTo(root);
            Number n2 = hCnumbering.numberOfPathsTo(root);
            BDD relation;
            if (n1.equals(n2)) {
                relation = V1c.buildAdd(H1c, BigInteger.valueOf(n1.longValue()).bitLength(), 0);
                relation.andWith(V1c.varRange(0, n1.longValue()-1));
                System.out.println("Root "+root+" numbering: "+relation.toStringWithDomains());
            } else {
                System.out.println("Root numbering doesn't match: "+root);
                // just intermix them all, because we don't know the mapping.
                relation = V1c.varRange(0, n1.longValue()-1);
                relation.andWith(H1c.varRange(0, n2.longValue()-1));
            }
            V1H1correspondence.put(root, relation);
        }
        List rpo = Traversals.reversePostOrder(cg.getMethodNavigator(), cg.getRoots());
        for (Iterator i = rpo.iterator(); i.hasNext(); ) {
            jq_Method callee = (jq_Method) i.next();
            //Assert._assert(!V1H1correspondence.containsKey(callee));
            BDD calleeRelation;
            calleeRelation = (BDD) V1H1correspondence.get(callee);
            if (calleeRelation == null)
                calleeRelation = bdd.zero();
            for (Iterator j = cg.getCallers(callee).iterator(); j.hasNext(); ) {
                ProgramLocation cs = (ProgramLocation) j.next();
                jq_Method caller = cs.getMethod();
                BDD callerRelation = (BDD) V1H1correspondence.get(caller);
                if (callerRelation == null) continue;
                Range r1_caller = vCnumbering.getRange(caller);
                Range r1_edge = vCnumbering.getEdge(cs, callee);
                Range r2_caller = hCnumbering.getRange(caller);
                Range r2_edge = hCnumbering.getEdge(cs, callee);
                BDD cm1;
                BDD tmpRel;
                boolean r1_same = r1_caller.equals(r1_edge);
                boolean r2_same = r2_caller.equals(r2_edge);
                if (!r1_same) {
                    cm1 = buildContextMap(V1c,
                                          PathNumbering.toBigInt(r1_caller.low),
                                          PathNumbering.toBigInt(r1_caller.high),
                                          V2c,
                                          PathNumbering.toBigInt(r1_edge.low),
                                          PathNumbering.toBigInt(r1_edge.high));
                    tmpRel = callerRelation.relprod(cm1, V1cset);
                    cm1.free();
                } else {
                    tmpRel = callerRelation.id();
                }
                BDD tmpRel2;
                if (!r2_same) {
                    cm1 = buildContextMap(H1c,
                                          PathNumbering.toBigInt(r2_caller.low),
                                          PathNumbering.toBigInt(r2_caller.high),
                                          H2c,
                                          PathNumbering.toBigInt(r2_edge.low),
                                          PathNumbering.toBigInt(r2_edge.high));
                    tmpRel2 = tmpRel.relprod(cm1, H1cset);
                    tmpRel.free();
                    cm1.free();
                } else {
                    tmpRel2 = tmpRel;
                }
                if (!r1_same) {
                    if (!r2_same) {
                        tmpRel2.replaceWith(V2cH2ctoV1cH1c);
                    } else {
                        tmpRel2.replaceWith(V2ctoV1c);
                    }
                } else if (!r2_same) {
                    tmpRel2.replaceWith(H2ctoH1c);
                }
                calleeRelation.orWith(tmpRel2);
            }
            V1H1correspondence.put(callee, calleeRelation);
        }
    }
}

// PA.java, created Oct 16, 2003 3:39:34 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
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
import org.sf.javabdd.BDDBitVector;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_InstanceField;
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
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operator.Invoke;
import Main.HostedVM;
import Util.Assert;
import Util.Collections.IndexMap;
import Util.Collections.Pair;
import Util.Graphs.Navigator;
import Util.Graphs.PathNumbering;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;
import Util.Graphs.PathNumbering.Range;
import Util.Graphs.PathNumbering.Selector;

/**
 * Pointer analysis using BDDs.  Includes both context-insensitive and context-sensitive
 * analyses.  This version corresponds exactly to the description in the paper.
 * All of the inference rules are direct copies.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class PA {

    public static final boolean VerifyAssertions = false;
    boolean TRACE = !System.getProperty("pa.trace", "no").equals("no");
    boolean TRACE_SOLVER = !System.getProperty("pa.tracesolver", "no").equals("no");
    boolean TRACE_BIND = !System.getProperty("pa.tracebind", "no").equals("no");
    boolean TRACE_RELATIONS = !System.getProperty("pa.tracerelations", "no").equals("no");
    boolean TRACE_OBJECT = !System.getProperty("pa.traceobject", "no").equals("no");
    PrintStream out = System.out;

    boolean INCREMENTAL1 = !System.getProperty("pa.inc1", "yes").equals("no"); // incremental points-to
    boolean INCREMENTAL2 = !System.getProperty("pa.inc2", "yes").equals("no"); // incremental parameter binding
    boolean INCREMENTAL3 = !System.getProperty("pa.inc3", "yes").equals("no"); // incremental invocation binding
    
    boolean ADD_CLINIT = !System.getProperty("pa.clinit", "yes").equals("no");
    boolean ADD_THREADS = !System.getProperty("pa.threads", "yes").equals("no");
    boolean ADD_FINALIZERS = !System.getProperty("pa.finalizers", "yes").equals("no");
    boolean IGNORE_EXCEPTIONS = !System.getProperty("pa.ignoreexceptions", "no").equals("no");
    boolean FILTER_VP = !System.getProperty("pa.vpfilter", "yes").equals("no");
    boolean FILTER_HP = !System.getProperty("pa.hpfilter", "no").equals("no");
    boolean OBJECT_SENSITIVE = !System.getProperty("pa.os", "no").equals("no");
    boolean CONTEXT_SENSITIVE = !System.getProperty("pa.cs", "no").equals("no");
    boolean DISCOVER_CALL_GRAPH = !System.getProperty("pa.discover", "no").equals("no");
    boolean DUMP_DOTGRAPH = !System.getProperty("pa.dumpdotgraph", "no").equals("no");
    boolean FILTER_NULL = !System.getProperty("pa.filternull", "yes").equals("no");
    
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "2500000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "150000"));
    static String resultsFileName = System.getProperty("pa.results", "pa");
    static String callgraphFileName = System.getProperty("pa.callgraph", "callgraph");
    static String initialCallgraphFileName = System.getProperty("pa.icallgraph", callgraphFileName);
    
    Map newMethodSummaries = new HashMap();
    Set rootMethods = new HashSet();
    
    CallGraph cg;
    ObjectCreationGraph ocg;
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    BDDDomain V1c, V2c, H1c, H2c;
    
    int V_BITS=18, I_BITS=16, H_BITS=15, Z_BITS=5, F_BITS=13, T_BITS=12, N_BITS=13, M_BITS=14;
    int VC_BITS=1, HC_BITS=1;
    int MAX_VC_BITS = Integer.parseInt(System.getProperty("pa.maxvc", "48"));
    int MAX_HC_BITS = Integer.parseInt(System.getProperty("pa.maxhc", "6"));
    
    IndexMap/*Node*/ Vmap;
    IndexMap/*ProgramLocation*/ Imap;
    IndexMap/*Node*/ Hmap;
    IndexMap/*jq_Field*/ Fmap;
    IndexMap/*jq_Reference*/ Tmap;
    IndexMap/*jq_Method*/ Nmap;
    IndexMap/*jq_Method*/ Mmap;
    PathNumbering vCnumbering; // for context-sensitive
    PathNumbering hCnumbering; // for context-sensitive
    PathNumbering oCnumbering; // for object-sensitive
    
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

    BDD fT;     // FxT2, field types                    (no context)
    BDD fC;     // FxT2, field containing types         (no context)

    BDD hP;     // H1xFxH2, heap points-to              (+context)
    BDD IE;     // IxM, invocation edges                (no context)
    BDD IEcs;   // V2cxIxV1cxM, context-sensitive invocation edges
    BDD vPfilter; // V1xH1, type filter                 (no context)
    BDD hPfilter; // H1xFxH2, type filter               (no context)
    BDD IEfilter; // V2cxIxV1cxM, context-sensitive edge filter
    
    BDD visited; // M, visited methods
    
    BDD staticCalls; // V1xIxM, statically-bound calls
    
    String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2xH2c_T2_H1xH1c");
    //String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_H2_T2_H1");
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDDPairing V1ctoV2c, V1cV2ctoV2cV1c, V1cH1ctoV2cV1c;
    BDDPairing T2toT1, T1toT2;
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1Fset, V2Fset, V1FV2set, V1H1set, H1Fset, H2Fset, H1H2set, H1FH2set;
    BDD IMset, INset, IV1set, INV1set, INH1set, INT2set, T2Nset, MZset;
    BDD V1cV2cset, V1cH1cset, H1cH2cset;
    
    BDDDomain makeDomain(String name, int bits) {
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }
    
    public void initializeBDD(String bddfactory) {
        if (CONTEXT_SENSITIVE) bddnodes *= 2;
        
        if (bddfactory == null)
            bdd = BDDFactory.init(bddnodes, bddcache);
        else
            bdd = BDDFactory.init(bddfactory, bddnodes, bddcache);
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
        
        V1ctoV2c = bdd.makePair(V1c, V2c);
        V1cV2ctoV2cV1c = bdd.makePair();
        V1cV2ctoV2cV1c.set(new BDDDomain[] {V1c,V2c},
                           new BDDDomain[] {V2c,V1c});
        if (OBJECT_SENSITIVE) {
            V1cH1ctoV2cV1c = bdd.makePair();
            V1cH1ctoV2cV1c.set(new BDDDomain[] {V1c,H1c},
                               new BDDDomain[] {V2c,V1c});
        }
        T2toT1 = bdd.makePair(T2, T1);
        T1toT2 = bdd.makePair(T1, T2);
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) {
            V1toV2 = bdd.makePair();
            V1toV2.set(new BDDDomain[] {V1,V1c},
                       new BDDDomain[] {V2,V2c});
            V2toV1 = bdd.makePair();
            V2toV1.set(new BDDDomain[] {V2,V2c},
                       new BDDDomain[] {V1,V1c});
            H2toH1 = bdd.makePair();
            H2toH1.set(new BDDDomain[] {H2,H2c},
                       new BDDDomain[] {H1,H1c});
            V1H1toV2H2 = bdd.makePair();
            V1H1toV2H2.set(new BDDDomain[] {V1,H1,V1c,H1c},
                           new BDDDomain[] {V2,H2,V2c,H2c});
            V2H2toV1H1 = bdd.makePair();
            V2H2toV1H1.set(new BDDDomain[] {V2,H2,V2c,H2c},
                           new BDDDomain[] {V1,H1,V1c,H1c});
            if (FILTER_HP) {
                H1toH2 = bdd.makePair();
                H1toH2.set(new BDDDomain[] {H1,H1c},
                           new BDDDomain[] {H2,H2c});
            }
        } else {
            V1toV2 = bdd.makePair(V1, V2);
            V2toV1 = bdd.makePair(V2, V1);
            H2toH1 = bdd.makePair(H2, H1);
            V1H1toV2H2 = bdd.makePair();
            V1H1toV2H2.set(new BDDDomain[] {V1,H1},
                           new BDDDomain[] {V2,H2});
            V2H2toV1H1 = bdd.makePair();
            V2H2toV1H1.set(new BDDDomain[] {V2,H2},
                           new BDDDomain[] {V1,H1});
            if (FILTER_HP) {
                H1toH2 = bdd.makePair(H1, H2);
            }
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
        V1cH1cset = V1c.set(); V1cH1cset.andWith(H1c.set());
        H1cH2cset = H1c.set(); H1cH2cset.andWith(H2c.set());
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) {
            V1set.andWith(V1c.set());
            V2set.andWith(V2c.set());
            H1set.andWith(H1c.set());
            H2set.andWith(H2c.set());
        }
        V1V2set = V1set.and(V2set);
        V1FV2set = V1V2set.and(Fset);
        V1H1set = V1set.and(H1set);
        V1Fset = V1set.and(Fset);
        V2Fset = V2set.and(Fset);
        IV1set = Iset.and(V1.set());
        IMset = Iset.and(Mset);
        INset = Iset.and(Nset);
        INV1set = INset.and(V1.set());
        INH1set = INset.and(H1set);
        INT2set = INset.and(T2set);
        H1Fset = H1set.and(Fset);
        H2Fset = H2set.and(Fset);
        H1H2set = H1set.and(H2set);
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
        if (FILTER_HP) {
            fT = bdd.zero();
            fC = bdd.zero();
        }
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
        
        if (OBJECT_SENSITIVE) staticCalls = bdd.zero();
        
        if (INCREMENTAL1) {
            old1_A = bdd.zero();
            old1_S = bdd.zero();
            old1_L = bdd.zero();
            old1_vP = bdd.zero();
            old1_hP = bdd.zero();
        }
        if (INCREMENTAL2) {
            old2_myIE = bdd.zero();
            old2_visited = bdd.zero();
        }
        if (INCREMENTAL3) {
            old3_t3 = bdd.zero();
            old3_vP = bdd.zero();
            old3_t4 = bdd.zero();
            old3_hT = bdd.zero();
        }
    }
    
    void initializeMaps() {
        Vmap = makeMap("Vars", V_BITS);
        Imap = makeMap("Invokes", I_BITS);
        Hmap = makeMap("Heaps", H_BITS);
        Fmap = makeMap("Fields", F_BITS);
        Tmap = makeMap("Types", T_BITS);
        Nmap = makeMap("Names", N_BITS);
        Mmap = makeMap("Methods", M_BITS);
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
    
    void addSingleTargetCall(Set thisptr, ProgramLocation mc, BDD I_bdd, jq_Method target) {
        addToIE(I_bdd, target);
        if (OBJECT_SENSITIVE) {
            BDD bdd1 = bdd.zero();
            for (Iterator j = thisptr.iterator(); j.hasNext(); ) {
                int V_i = Vmap.get(j.next());
                bdd1.orWith(V1.ithVar(V_i));
            }
            bdd1.andWith(I_bdd.id());
            int M_i = Mmap.get(target);
            bdd1.andWith(M.ithVar(M_i));
            staticCalls.orWith(bdd1);
        }
    }
    
    void addToIE(BDD I_bdd, jq_Method target) {
        int M2_i = Mmap.get(target);
        BDD bdd1 = M.ithVar(M2_i);
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to IE: "+bdd1.toStringWithDomains());
        if (CONTEXT_SENSITIVE) {
            // Add the context for the new call graph edge.
            IEcs.orWith(bdd1.and(IEfilter));
        }
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
            bdd1.orWith(V2.ithVar(V_i));
        }
        bdd1.andWith(Z.ithVar(z));
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to actual: "+bdd1.toStringWithDomains());
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
    
    void addToVP(Node p, int H_i) {
        BDD context = CONTEXT_SENSITIVE||OBJECT_SENSITIVE?bdd.one():null;
        addToVP(context, p, H_i);
        if (CONTEXT_SENSITIVE||OBJECT_SENSITIVE) context.free();
    }
    
    void addToVP(BDD V1H1context, Node p, int H_i) {
        int V1_i = Vmap.get(p);
        BDD bdd1 = V1.ithVar(V1_i);
        bdd1.andWith(H1.ithVar(H_i));
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToVP(BDD V_bdd, Node h) {
        BDD context = CONTEXT_SENSITIVE||OBJECT_SENSITIVE?bdd.one():null;
        addToVP(context, V_bdd, h);
        if (CONTEXT_SENSITIVE||OBJECT_SENSITIVE) context.free();
    }
    
    void addToVP(BDD V1H1context, BDD V_bdd, Node h) {
        int H_i = Hmap.get(h);
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(V_bdd.id());
        if (CONTEXT_SENSITIVE) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToA(int V1_i, int V2_i) {
        BDD context = CONTEXT_SENSITIVE||OBJECT_SENSITIVE?bdd.one():null;
        addToA(context, V1_i, V2_i);
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) context.free();
    }
    
    void addToA(BDD V1V2context, int V1_i, int V2_i) {
        BDD V_bdd = V1.ithVar(V1_i);
        addToA(V1V2context, V_bdd, V2_i);
        V_bdd.free();
    }
    
    void addToA(BDD V_bdd, int V2_i) {
        BDD context = CONTEXT_SENSITIVE||OBJECT_SENSITIVE?bdd.one():null;
        addToA(context, V_bdd, V2_i);
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) context.free();
    }
    
    void addToA(BDD V1V2context, BDD V_bdd, int V2_i) {
        BDD bdd1 = V2.ithVar(V2_i);
        bdd1.andWith(V_bdd.id());
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) bdd1.andWith(V1V2context.id());
        if (TRACE_RELATIONS) out.println("Adding to A: "+bdd1.toStringWithDomains());
        A.orWith(bdd1);
    }
    
    void addToS(BDD V_bdd, jq_Field f, Collection c) {
        BDD context = CONTEXT_SENSITIVE||OBJECT_SENSITIVE?bdd.one():null;
        addToS(context, V_bdd, f, c);
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) context.free();
    }
    
    void addToS(BDD V1V2context, BDD V_bdd, jq_Field f, Collection c) {
        int F_i = Fmap.get(f);
        BDD F_bdd = F.ithVar(F_i);
        for (Iterator k = c.iterator(); k.hasNext(); ) {
            Node node2 = (Node) k.next();
            if (FILTER_NULL && isNullConstant(node2))
		continue;

            int V2_i = Vmap.get(node2);
            BDD bdd1 = V2.ithVar(V2_i);
            bdd1.andWith(F_bdd.id());
            bdd1.andWith(V_bdd.id());
            if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) bdd1.andWith(V1V2context.id());
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
            if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE) bdd1.andWith(V1V2context.id());
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
        if (CONTEXT_SENSITIVE) {
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
        } else if (OBJECT_SENSITIVE) {
            // One-to-one match if call is on 'this' pointer.
            boolean one_to_one;
            jq_Method caller = mc.getMethod();
            jq_Method target = mc.getTargetMethod();
            if (target.isStatic()) {
                one_to_one = caller.isStatic();
            } else {
                Quad q = ((ProgramLocation.QuadProgramLocation) mc).getQuad();
                RegisterOperand rop = Invoke.getParam(q, 0);
                System.out.println("rop = "+rop);
                one_to_one = rop.getType() == caller.getDeclaringClass();
            }
            jq_Class c;
            if (caller.isStatic()) c = null;
            else c = caller.getDeclaringClass();
            Range r = (Range) rangeMap.get(c);
            System.out.println("Method call: "+mc);
            System.out.println("Range of "+c+" = "+r);
            BDD V1V2context;
            if (r == null) {
                System.out.println("Warning: when getting VC, "+c+" is not in object creation graph.");
                V1V2context = bdd.one();
                return V1V2context;
            }
            if (one_to_one) {
                int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
                V1V2context = V1c.buildAdd(V2c, bits, 0L);
                V1V2context.andWith(V1c.varRange(r.low.longValue(), r.high.longValue()));
            } else {
                V1V2context = V1c.varRange(r.low.longValue(), r.high.longValue());
                V1V2context.andWith(V2c.varRange(r.low.longValue(), r.high.longValue()));
            }
            return V1V2context;
        } else {
            return null;
        }
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
    
    public boolean alreadyVisited(jq_Method m) {
        int M_i = Mmap.get(m);
        BDD M_bdd = M.ithVar(M_i);
        M_bdd.andWith(visited.id());
        boolean result = !M_bdd.isZero();
        M_bdd.free();
        return result;
    }
    
    public void visitMethod(jq_Method m) {
        if (alreadyVisited(m)) return;
        if (VerifyAssertions && cg != null)
            Assert._assert(cg.getAllMethods().contains(m), m.toString());
        PAMethodSummary s = new PAMethodSummary(this, m);
        if (VerifyAssertions) Assert._assert(newMethodSummaries.get(m) == s);
    }
    
    public void addAllMethods() {
        for (Iterator i = newMethodSummaries.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            jq_Method m = (jq_Method) e.getKey();
            PAMethodSummary s = (PAMethodSummary) e.getValue();
            BDD V1V2context = getV1V2Context(m);
            BDD V1H1context = getV1H1Context(m);
            s.registerRelations(V1V2context, V1H1context);
            if (V1V2context != null) V1V2context.free();
            if (V1H1context != null) V1H1context.free();
            s.free();
            i.remove();
        }
    }
    
    Map rangeMap;
    
    public BDD getV1V2Context(jq_Method m) {
        if (CONTEXT_SENSITIVE) {
            Range r = vCnumbering.getRange(m);
            int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
            BDD V1V2context = V1c.buildAdd(V2c, bits, 0L);
            V1V2context.andWith(V1c.varRange(r.low.longValue(), r.high.longValue()));
            return V1V2context;
        } else if (OBJECT_SENSITIVE) {
            jq_Class c;
            if (m.isStatic()) c = null;
            else c = m.getDeclaringClass();
            Range r = (Range) rangeMap.get(c);
            if (TRACE_OBJECT) out.println("Range to "+c+" = "+r);
            BDD V1V2context;
            if (r == null) {
                System.out.println("Warning: when getting V1V2, "+c+" is not in object creation graph!  Assuming global only.");
                V1V2context = V1c.ithVar(0);
                V1V2context.andWith(V2c.ithVar(0));
                return V1V2context;
            }
            int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
            V1V2context = V1c.buildAdd(V2c, bits, 0L);
            V1V2context.andWith(V1c.varRange(r.low.longValue(), r.high.longValue()));
            return V1V2context;
        } else {
            return null;
        }
    }
    
    public void visitGlobalNode(Node node) {
        if (TRACE) out.println("Visiting node "+node);
       
        if (FILTER_NULL && isNullConstant(node))
	    return;
        
        int V_i = Vmap.get(node);
        BDD V_bdd = V1.ithVar(V_i);
        
        if (VerifyAssertions)
            Assert._assert(node instanceof ConcreteObjectNode ||
                           node instanceof UnknownTypeNode ||
                           node == GlobalNode.GLOBAL);
        addToVP(V_bdd, node);
        
        for (Iterator j = node.getAllEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Collection c;
            if (e.getValue() instanceof Collection)
                c = (Collection) e.getValue();
            else
                c = Collections.singleton(e.getValue());
            addToS(V_bdd, f, c);
        }
        
        if (VerifyAssertions)
            Assert._assert(!node.hasAccessPathEdges());
    }

    public boolean isNullConstant(Node node) {
	if (node instanceof ConcreteTypeNode || node instanceof ConcreteObjectNode) {
            jq_Reference type = node.getDeclaredType();
            if (type == null) {
                if (TRACE) out.println("Skipping null constant.");
                return true;
            }
        }
	return false;
    }
    
    void addToVT(int V_i, jq_Reference type) {
        BDD bdd1 = V1.ithVar(V_i);
        int T_i = Tmap.get(type);
        bdd1.andWith(T1.ithVar(T_i));
        if (TRACE_RELATIONS) out.println("Adding to vT: "+bdd1.toStringWithDomains());
        vT.orWith(bdd1);
    }
    
    void addToHT(int H_i, jq_Reference type) {
        int T_i = Tmap.get(type);
        BDD T_bdd = T2.ithVar(T_i);
        addToHT(H_i, T_bdd);
        T_bdd.free();
    }
    
    void addToHT(int H_i, BDD T_bdd) {
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(T_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to hT: "+bdd1.toStringWithDomains());
        hT.orWith(bdd1);
    }
    
    void addToAT(BDD T1_bdd, int T2_i) {
        BDD bdd1 = T2.ithVar(T2_i);
        bdd1.andWith(T1_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to aT: "+bdd1.toStringWithDomains());
        aT.orWith(bdd1);
    }
    
    void addToFC(BDD T2_bdd, int F_i) {
        BDD bdd1 = F.ithVar(F_i);
        bdd1.andWith(T2_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to fC: "+bdd1.toStringWithDomains(TS));
        fC.orWith(bdd1);
    }
    
    void addToFT(BDD F_bdd, BDD T2_bdd) {
        BDD bdd1 = F_bdd.and(T2_bdd);
        if (TRACE_RELATIONS) out.println("Adding to fT: "+bdd1.toStringWithDomains(TS));
        fT.orWith(bdd1);
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
    int last_F = 0;
    
    public void buildTypes() {
        // build up 'vT'
        int Vsize = Vmap.size();
        for (int V_i = last_V; V_i < Vsize; ++V_i) {
            Node n = (Node) Vmap.get(V_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) type.prepare();
            addToVT(V_i, type);
        }
        
        // build up 'hT', and identify clinit, thread run, finalizers.
        int Hsize = Hmap.size();
        for (int H_i = last_H; H_i < Hsize; ++H_i) {
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
            if (false && n instanceof UnknownTypeNode) {
                // conservatively say that it can be any type.
                BDD Tdom = T2.domain();
                addToHT(H_i, Tdom);
                Tdom.free();
            } else {
                addToHT(H_i, type);
            }
        }
        
        int Fsize = Fmap.size();
        int Tsize = Tmap.size();
        // build up 'aT'
        for (int T1_i = 0; T1_i < Tsize; ++T1_i) {
            jq_Reference t1 = (jq_Reference) Tmap.get(T1_i);
            int start = (T1_i < last_T)?last_T:0;
            BDD T1_bdd = T1.ithVar(T1_i);
            for (int T2_i = start; T2_i < Tsize; ++T2_i) {
                jq_Reference t2 = (jq_Reference) Tmap.get(T2_i);
                if (t2 == null || (t1 != null && t2.isSubtypeOf(t1))) {
                    addToAT(T1_bdd, T2_i);
                }
            }
            if (FILTER_HP) {
                BDD T2_bdd = T2.ithVar(T1_i);
                if (T1_i >= last_T && t1 == null) {
                    BDD Fdom = F.domain();
                    addToFT(Fdom, T2_bdd);
                    Fdom.free();
                }
                int start2 = (T1_i < last_T)?last_F:0;
                for (int F_i = start2; F_i < Fsize; ++F_i) {
                    jq_Field f = (jq_Field) Fmap.get(F_i);
                    if (f != null) {
                        f.getDeclaringClass().prepare();
                        f.getType().prepare();
                    }
                    BDD F_bdd = F.ithVar(F_i);
                    if ((t1 == null && f != null && f.isStatic()) ||
                        (t1 != null && ((f == null && t1 instanceof jq_Array && ((jq_Array) t1).getElementType().isReferenceType()) ||
                                        (f != null && t1.isSubtypeOf(f.getDeclaringClass()))))) {
                        addToFC(T2_bdd, F_i);
                    }
                    if (f != null && t1 != null && t1.isSubtypeOf(f.getType())) {
                        addToFT(F_bdd, T2_bdd);
                    }
                }
                T2_bdd.free();
            }
            T1_bdd.free();
        }
        
        // make type filters
        if (FILTER_VP) {
            if (vPfilter != null) vPfilter.free();
            BDD t1 = vT.relprod(aT, T1set); // V1xT1 x T1xT2 = V1xT2
            vPfilter = t1.relprod(hT, T2set); // V1xT2 x H1xT2 = V1xH1
            t1.free();
        }

        if (FILTER_HP) {
            for (int F_i = last_F; F_i < Fsize; ++F_i) {
                jq_Field f = (jq_Field) Fmap.get(F_i);
                if (f == null) {
                    BDD F_bdd = F.ithVar(F_i);
                    BDD T2dom = T2.domain();
                    addToFT(F_bdd, T2dom);
                    T2dom.free();
                    F_bdd.free();
                }
            }
            if (hPfilter != null) hPfilter.free();
            BDD t1 = hT.relprod(fC, T2set); // H1xT2 x FxT2 = H1xF
            hPfilter = hT.relprod(fT, T2set); // H1xT2 x FxT2 = H1xF
            hPfilter.replaceWith(H1toH2); // H2xF
            hPfilter.andWith(t1); // H1xFxH2
        }
        
        // build up 'cha'
        int Nsize = Nmap.size();
        for (int T_i = 0; T_i < Tsize; ++T_i) {
            jq_Reference t = (jq_Reference) Tmap.get(T_i);
            BDD T_bdd = T2.ithVar(T_i);
            int start = (T_i < last_T)?last_N:0;
            for (int N_i = start; N_i < Nsize; ++N_i) {
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
        last_V = Vsize;
        last_H = Hsize;
        last_T = Tsize;
        last_N = Nsize;
        last_F = Fsize;
        if (Vsize != Vmap.size() ||
            Hsize != Hmap.size() ||
            Tsize != Tmap.size() ||
            Nsize != Nmap.size() ||
            Fsize != Fmap.size()) {
            if (TRACE) out.println("Elements added, recalculating types...");
            buildTypes();
        }
    }
    
    public void addClassInitializer(jq_Class c) {
        if (!ADD_CLINIT) return;
        jq_Method m = c.getClassInitializer();
        if (m != null) {
            visitMethod(m);
            rootMethods.add(m);
        }
    }
    
    jq_NameAndDesc finalizer_method = new jq_NameAndDesc("finalize", "()V");
    public void addFinalizer(jq_Class c) {
        if (!ADD_FINALIZERS) return;
        jq_Method m = c.getVirtualMethod(finalizer_method);
        if (m != null) {
            visitMethod(m);
            rootMethods.add(m);
        }
    }
    
    static jq_NameAndDesc main_method = new jq_NameAndDesc("main", "([Ljava/lang/String;)V");
    static jq_NameAndDesc run_method = new jq_NameAndDesc("run", "()V");
    public void addThreadRun(int H_i, jq_Class c) {
        if (!ADD_THREADS) return;
        jq_Method m = c.getVirtualMethod(run_method);
        if (m != null && m.getBytecode() != null) {
            visitMethod(m);
            rootMethods.add(m);
            Node p = MethodSummary.getSummary(CodeCache.getCode(m)).getParamNode(0);
            Node h = (Node) Hmap.get(H_i);
            BDD context = null;
            if (CONTEXT_SENSITIVE && MAX_HC_BITS > 1) {
                int context_i = getThreadRunIndex(m, h);
                System.out.println("Thread "+h+" index "+context_i);
                context = H1c.ithVar(context_i);
                context.andWith(V1c.ithVar(context_i));
                addToVP(context, p, H_i);
                context.free();
            } else {
                addToVP(p, H_i);
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
                if (FILTER_VP) t2.andWith(vPfilter.id());
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
            if (FILTER_HP) t5.andWith(hPfilter.id());
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
            if (FILTER_VP) t7.andWith(vPfilter.id());
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
            if (FILTER_VP) t2.andWith(vPfilter.id());
            if (TRACE_SOLVER) out.print(" vP "+vP.satCount(V1H1set));
            vP.orWith(t2);
            if (TRACE_SOLVER) out.println(" --> "+vP.satCount(V1H1set));
        }
        old1_A = A.id();
        
        // handle new S
        BDD new_S = S.apply(old1_S, BDDFactory.diff);
        old1_S.free();
        if (!new_S.isZero()) {
            if (TRACE_SOLVER) out.print("Handling new S: "+new_S.satCount(V1FV2set));
            BDD t3 = new_S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            new_S.free();
            BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
            BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
            t3.free(); t4.free();
            if (FILTER_HP) t5.andWith(hPfilter.id());
            if (TRACE_SOLVER) out.print(" hP "+hP.satCount(H1FH2set));
            hP.orWith(t5);
            if (TRACE_SOLVER) out.println(" --> "+hP.satCount(H1FH2set));
        }
        old1_S = S.id();
        
        // handle new L
        BDD new_L = L.apply(old1_L, BDDFactory.diff);
        old1_L.free();
        if (!new_L.isZero()) {
            if (TRACE_SOLVER) out.print("Handling new L: "+new_L.satCount(V1FV2set));
            BDD t6 = new_L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
            t6.free();
            t7.replaceWith(V2H2toV1H1); // V1xH1
            if (FILTER_VP) t7.andWith(vPfilter.id());
            if (TRACE_SOLVER) out.print(" vP "+vP.satCount(V1H1set));
            vP.orWith(t7);
            if (TRACE_SOLVER) out.println(" --> "+vP.satCount(V1H1set));
        }
        old1_L = S.id();
        
        for (int outer = 1; ; ++outer) {
            BDD new_vP_inner = vP.apply(old1_vP, BDDFactory.diff);
            int inner;
            for (inner = 1; !new_vP_inner.isZero() && inner < 256; ++inner) {
                if (TRACE_SOLVER)
                    out.print("Inner #"+inner+": new vP "+new_vP_inner.satCount(V1H1set));
                
                // Rule 1
                BDD t1 = new_vP_inner.replace(V1toV2); // V2xH1
                new_vP_inner.free();
                BDD t2 = A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
                t1.free();
                if (FILTER_VP) t2.andWith(vPfilter.id());
                
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
                if (FILTER_HP) t5.andWith(hPfilter.id());
                hP.orWith(t5);
            }
            {
                // Rule 2
                BDD t3 = S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t4 = new_vP.replace(V1H1toV2H2); // V2xH2
                BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
                t3.free(); t4.free();
                if (FILTER_HP) t5.andWith(hPfilter.id());
                hP.orWith(t5);
            }

            if (TRACE_SOLVER)
                out.println(", hP "+old1_hP.satCount(H1FH2set)+" -> "+hP.satCount(H1FH2set));
            
            old1_vP = vP.id();
            
            BDD new_hP = hP.apply(old1_hP, BDDFactory.diff);
            if (new_hP.isZero() && inner < 256) break;
            old1_hP = hP.id();
            
            if (TRACE_SOLVER)
                out.print("        : new hP "+new_hP.satCount(H1FH2set));
            
            {
                // Rule 3
                BDD t6 = L.relprod(new_vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                if (FILTER_VP) t7.andWith(vPfilter.id());
                vP.orWith(t7);
            }
            {
                // Rule 3
                BDD t6 = L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                BDD t7 = t6.relprod(new_hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                if (FILTER_VP) t7.andWith(vPfilter.id());
                vP.orWith(t7);
            }
            if (TRACE_SOLVER)
                out.println(", vP "+old1_vP.satCount(V1H1set)+
                            " -> "+vP.satCount(V1H1set));
        }
    }
    
    public void bindInvocations() {
        if (INCREMENTAL3 && !OBJECT_SENSITIVE) {
            bindInvocations_incremental();
            return;
        }
        BDD t1 = actual.restrict(Z.ithVar(0)); // IxV2
        t1.replaceWith(V2toV1); // IxV1
        if (TRACE_BIND) out.println("t1: "+t1.satCount(IV1set));
        if (TRACE_BIND) out.println("t1: "+t1.toStringWithDomains(TS));
        BDD t2 = mI.exist(Mset); // IxN
        if (TRACE_BIND) out.println("t2: "+t2.satCount(INset));
        if (TRACE_BIND) out.println("t2: "+t2.toStringWithDomains(TS));
        BDD t3 = t1.and(t2); // IxV1 & IxN = IxV1xN
        if (TRACE_BIND) out.println("t3: "+t3.satCount(INV1set));
        if (TRACE_BIND) out.println("t3: "+t3.toStringWithDomains(TS));
        t1.free(); t2.free();
        BDD t4 = t3.relprod(vP, V1set); // IxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
        if (TRACE_BIND) out.println("t4: "+t4.satCount(INH1set));
        if (TRACE_BIND) out.println("t4: "+t4.toStringWithDomains(TS));
        BDD t5 = t4.relprod(hT, H1set); // IxH1cxH1xN x H1xT2 = IxT2xN
        if (TRACE_BIND) out.println("t5: "+t5.satCount(INT2set));
        if (TRACE_BIND) out.println("t5: "+t5.toStringWithDomains(TS));
        t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // IxT2xN x T2xNxM = IxM
        if (TRACE_BIND) out.println("t6: "+t6.satCount(IMset));
        if (TRACE_BIND) out.println("t6: "+t6.toStringWithDomains(TS));
        t5.free();
        
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        IE.orWith(t6.id());
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        if (CONTEXT_SENSITIVE) {
            // Add the context for the new call graph edges.
            t6.andWith(IEfilter.id());
            IEcs.orWith(t6.id());
        } else if (OBJECT_SENSITIVE) {
            // Add the context for the new edges.
            t4 = t3.relprod(vP, V1.set()); // IxV1xN x V1cxV1xH1cxH1 = V1cxIxH1cxH1xN
            t5 = t4.relprod(hT, H1.set()); // V1cxIxH1cxH1xN x H1xT2 = V1cxIxH1cxT2xN
            t4.free();
            BDD t7 = t5.relprod(cha, T2Nset); // V1cxIxH1cxT2xN x T2xNxM = V1cxIxH1cxM
            t5.free();
            t7.replaceWith(V1cH1ctoV2cV1c); // V2cxIxV1cxM
            IEcs.orWith(t7);
            
            // Add the context for statically-bound call edges.
            BDD t8 = staticCalls.relprod(vP, V1.set().and(H1.set())); // V1xIxM x V1cxV1xH1cxH1 = V1cxIxH1cxM
            t8.replaceWith(V1cH1ctoV2cV1c); // V2cxIxV1cxM
            IEcs.orWith(t8);
        }
        t3.free();
        t6.free();
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
        if (false) out.println("New invokes: "+new_t3.toStringWithDomains());
        BDD new_vP = vP.apply(old3_vP, BDDFactory.diff);
        old3_vP.free();
        if (false) out.println("New vP: "+new_vP.toStringWithDomains());
        BDD t4 = t3.relprod(new_vP, V1set); // IxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
        new_vP.free();
        old3_t3 = t3;
        t4.orWith(new_t3.relprod(vP, V1set)); // IxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
        new_t3.free();
        BDD new_t4 = t4.apply(old3_t4, BDDFactory.diff);
        old3_t4.free();
        if (false) out.println("New 'this' objects: "+new_t4.toStringWithDomains());
        BDD new_hT = hT.apply(old3_hT, BDDFactory.diff);
        old3_hT.free();
        BDD t5 = t4.relprod(new_hT, H1set); // IxH1cxH1xN x H1xT2 = IxT2xN
        new_hT.free();
        old3_t4 = t4;
        t5.orWith(new_t4.relprod(hT, H1set)); // IxH1cxH1xN x H1xT2 = IxT2xN
        new_t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // IxT2xN x T2xNxM = IxM
        t5.free();
        
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        IE.orWith(t6.id());
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        if (CONTEXT_SENSITIVE) {
            t6.andWith(IEfilter.id());
            IEcs.orWith(t6.id());
        }
        t6.free();
        
        old3_vP = vP.id();
        old3_hT = hT.id();
    }
    
    public boolean handleNewTargets() {
        if (TRACE_SOLVER) out.println("Handling new target methods...");
        BDD targets = IE.exist(Iset); // IxM -> M
        targets.applyWith(visited.id(), BDDFactory.diff);
        if (targets.isZero()) return false;
        if (TRACE_SOLVER) out.println("New target methods: "+targets.satCount(Mset));
        while (!targets.isZero()) {
            BDD target = targets.satOne(Mset, bdd.zero());
            int M_i = (int) target.scanVar(M);
            jq_Method method = (jq_Method) Mmap.get(M_i);
            if (TRACE) out.println("New target method: "+method);
            visitMethod(method);
            targets.applyWith(target, BDDFactory.diff);
        }
        return true;
    }
    
    public void bindParameters() {
        if (INCREMENTAL2) {
            bindParameters_incremental();
            return;
        }
        
        if (TRACE_SOLVER) out.println("Binding parameters...");
        
        BDD my_IE = CONTEXT_SENSITIVE ? IEcs : IE;
        
        if (TRACE_SOLVER) out.println("Number of call graph edges: "+my_IE.satCount(IMset));
        
        BDD t1 = my_IE.relprod(actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD my_IEr = CONTEXT_SENSITIVE ? IEcs.replace(V1cV2ctoV2cV1c) : IE;
        BDD t3 = my_IEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = my_IEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (CONTEXT_SENSITIVE) my_IEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
    }
    
    BDD old2_myIE;
    BDD old2_visited;
    
    public void bindParameters_incremental() {
        if (TRACE_SOLVER) out.println("Binding parameters...");
        
        BDD my_IE = CONTEXT_SENSITIVE ? IEcs : IE;
        
        BDD new_myIE = my_IE.apply(old2_myIE, BDDFactory.diff);
        BDD new_visited = visited.apply(old2_visited, BDDFactory.diff);
        // add in any old edges targetting newly-visited methods, because the
        // argument/retval binding doesn't occur until the method has been visited.
        new_myIE.orWith(old2_myIE.and(new_visited));
        old2_myIE.free();
        old2_visited.free();
        new_visited.free();
        
        if (TRACE_SOLVER) out.println("Number of new call graph edges: "+new_myIE.satCount(IMset));
        
        BDD t1 = new_myIE.relprod(actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (false) out.println("New edges for param binding: "+t2.toStringWithDomains());
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD new_myIEr = CONTEXT_SENSITIVE ? new_myIE.replace(V1cV2ctoV2cV1c) : new_myIE;
        BDD t3 = new_myIEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (false) out.println("New edges for return binding: "+t4.toStringWithDomains());
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = new_myIEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (CONTEXT_SENSITIVE) new_myIEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
        new_myIE.free();
        old2_myIE = my_IE.id();
        old2_visited = visited.id();
    }
    
    public void assumeKnownCallGraph() {
        if (VerifyAssertions)
            Assert._assert(!IE.isZero());
        handleNewTargets();
        addAllMethods();
        buildTypes();
        bindParameters();
        solvePointsTo();
    }
    
    public void iterate() {
        BDD vP_old = vP.id();
        BDD IE_old = IE.id();
        boolean change;
        for (int major = 1; ; ++major) {
            change = false;
            
            out.println("Discovering call graph, iteration "+major+": "+(int)visited.satCount(Mset)+" methods.");
            long time = System.currentTimeMillis();
            buildTypes();
            solvePointsTo();
            bindInvocations();
            if (handleNewTargets())
                change = true;
            if (!change && vP.equals(vP_old) && IE.equals(IE_old)) {
                if (TRACE_SOLVER) out.println("Finished after "+major+" iterations.");
                break;
            }
            vP_old.free(); vP_old = vP.id();
            IE_old.free(); IE_old = IE.id();
            addAllMethods();
            bindParameters();
            if (TRACE_SOLVER)
                out.println("Time spent: "+(System.currentTimeMillis()-time)/1000.);
        }
    }
    
    public void numberPaths(CallGraph cg, ObjectCreationGraph ocg, boolean updateBits) {
        System.out.print("Counting size of call graph...");
        long time = System.currentTimeMillis();
        vCnumbering = countCallGraph(cg, ocg, updateBits);
        if (OBJECT_SENSITIVE) {
            oCnumbering = new PathNumbering(objectPathSelector);
            BigInteger paths = (BigInteger) oCnumbering.countPaths(ocg);
            if (updateBits) {
                HC_BITS = VC_BITS = paths.bitLength();
                System.out.print("Object paths="+paths+" ("+VC_BITS+" bits), ");
            }
        }
        if (CONTEXT_SENSITIVE && MAX_HC_BITS > 1) {
            hCnumbering = countHeapNumbering(cg, updateBits);
        }
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
    
    public void addDefaults() {
        // Add the default static variables (System in/out/err...)
        GlobalNode.GLOBAL.addDefaultStatics();
        
        // If using object-sensitive, initialize the object creation graph.
        this.ocg = null;
        if (OBJECT_SENSITIVE) {
            this.ocg = new ObjectCreationGraph();
            //ocg.handleCallGraph(cg);
            this.ocg.addRoot(null);
            for (Iterator i = ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
                ConcreteObjectNode con = (ConcreteObjectNode) i.next();
                if (con.getDeclaredType() == null) continue;
                this.ocg.addEdge(null, (Node) null, con.getDeclaredType());
            }
        }
    }
    
    public void run(CallGraph cg, Collection rootMethods) throws IOException {
        run(null, cg, rootMethods);
    }
    public void run(String bddfactory, CallGraph cg, Collection rootMethods) throws IOException {
        addDefaults();
        
        // If we have a call graph, use it for numbering and calculating domain sizes.
        if (cg != null) {
            numberPaths(cg, ocg, true);
        }
        
        // Now we know domain sizes, so initialize the BDD package.
        initializeBDD(bddfactory);
        initializeMaps();
        this.rootMethods.addAll(rootMethods);
        
        // Use the existing call graph to calculate IE filter
        if (cg != null) {
            System.out.print("Calculating call graph relation...");
            long time = System.currentTimeMillis();
            calculateIEfilter(cg);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
            
            // Build up var-heap correspondence in context-sensitive case.
            if (CONTEXT_SENSITIVE && HC_BITS > 1) {
                System.out.print("Building var-heap context correspondence...");
                time = System.currentTimeMillis();
                buildVarHeapCorrespondence(cg);
                time = System.currentTimeMillis() - time;
                System.out.println("done. ("+time/1000.+" seconds)");
            }
            
            // Use the IE filter as the set of invocation edges.
            if (!DISCOVER_CALL_GRAPH) {
                if (VerifyAssertions)
                    Assert._assert(IEfilter != null);
                if (CONTEXT_SENSITIVE) {
                    IEcs = IEfilter;
                    IE = IEcs.exist(V1cV2cset);
                } else {
                    IE = IEfilter;
                }
            }
        }
        
        // Start timing.
        long time = System.currentTimeMillis();
        
        // Add the global relations first.
        visitGlobalNode(GlobalNode.GLOBAL);
        for (Iterator i = ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            ConcreteObjectNode con = (ConcreteObjectNode) i.next();
            visitGlobalNode(con);
        }
        
        // Calculate the relations for the root methods.
        for (Iterator i = rootMethods.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            visitMethod(m);
        }
        
        // Calculate the relations for any other methods we know about.
        handleNewTargets();
        
        // For object-sensitivity, build up the context mapping.
        if (OBJECT_SENSITIVE) {
            buildTypes();
            buildObjectSensitiveV1H1(ocg);
        }
        
        // Now that contexts are calculated, add the relations for all methods
        // to the global relation.
        addAllMethods();
        
        System.out.println("Time spent initializing: "+(System.currentTimeMillis()-time)/1000.);
        
        // Start timing solver.
        time = System.currentTimeMillis();
        
        if (DISCOVER_CALL_GRAPH || OBJECT_SENSITIVE) {
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
        dis.cg = null;
        if (dis.CONTEXT_SENSITIVE || !dis.DISCOVER_CALL_GRAPH) {
            dis.cg = loadCallGraph(rootMethods);
            if (dis.cg == null) {
                if (dis.CONTEXT_SENSITIVE || dis.OBJECT_SENSITIVE) {
                    System.out.println("Discovering call graph first...");
                    dis.CONTEXT_SENSITIVE = false;
                    dis.OBJECT_SENSITIVE = false;
                    dis.DISCOVER_CALL_GRAPH = true;
                    dis.run("java", dis.cg, rootMethods);
                    System.out.println("Finished discovering call graph.");
                    dis = new PA();
                    dis.cg = loadCallGraph(rootMethods);
                    rootMethods = dis.cg.getRoots();
                } else if (!dis.DISCOVER_CALL_GRAPH) {
                    System.out.println("Call graph doesn't exist yet, so turning on call graph discovery.");
                    dis.DISCOVER_CALL_GRAPH = true;
                }
            } else {
                rootMethods = dis.cg.getRoots();
            }
        }
        dis.run(dis.cg, rootMethods);
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
    
    String findInMap(IndexMap map, int j) {
        String jp = "("+j+")";
        if (j < map.size()) {
            Object o = map.get(j);
            return jp+o;
        } else {
            return jp+"<index not in map>";
        }
    }

    public class ToString extends BDD.BDDToString {
        public String elementName(int i, long j) {
            switch (i) {
                case 0: // fallthrough
                case 1: return findInMap(Vmap, (int)j);
                case 2: return findInMap(Imap, (int)j);
                case 3: // fallthrough
                case 4: return findInMap(Hmap, (int)j);
                case 5: return Long.toString(j);
                case 6: return findInMap(Fmap, (int)j);
                case 7: // fallthrough
                case 8: return findInMap(Tmap, (int)j);
                case 9: return findInMap(Nmap, (int)j);
                case 10: return findInMap(Mmap, (int)j);
                case 11: return Long.toString(j);
                case 12: return Long.toString(j);
                case 13: return Long.toString(j);
                case 14: return Long.toString(j);
                default: return "("+j+")"+"??";
            }
        }
        public String elementNames(int i, long j, long k) {
            // TODO: don't bother printing out long form of big sets.
            return super.elementNames(i, j, k);
        }
    }
   
    private void dumpCallGraphAsDot(CallGraph callgraph, String dotFileName) throws IOException {
	DataOutputStream dos = new DataOutputStream(new FileOutputStream(dotFileName));
	countCallGraph(callgraph, null, false).dotGraph(dos);
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
        
        System.out.println("A: "+(long) A.satCount(V1V2set)+" relations, "+A.nodeCount()+" nodes");
        bdd.save(dumpfilename+".A", A);
        System.out.println("vP: "+(long) vP.satCount(V1H1set)+" relations, "+vP.nodeCount()+" nodes");
        bdd.save(dumpfilename+".vP", vP);
        System.out.println("S: "+(long) S.satCount(V1FV2set)+" relations, "+S.nodeCount()+" nodes");
        bdd.save(dumpfilename+".S", S);
        System.out.println("L: "+(long) L.satCount(V1FV2set)+" relations, "+L.nodeCount()+" nodes");
        bdd.save(dumpfilename+".L", L);
        System.out.println("vT: "+(long) vT.satCount(V1.set().and(T1set))+" relations, "+vT.nodeCount()+" nodes");
        bdd.save(dumpfilename+".vT", vT);
        System.out.println("hT: "+(long) hT.satCount(H1.set().and(T2set))+" relations, "+hT.nodeCount()+" nodes");
        bdd.save(dumpfilename+".hT", hT);
        System.out.println("aT: "+(long) aT.satCount(T1set.and(T2set))+" relations, "+aT.nodeCount()+" nodes");
        bdd.save(dumpfilename+".aT", aT);
        System.out.println("cha: "+(long) cha.satCount(T2Nset.and(Mset))+" relations, "+cha.nodeCount()+" nodes");
        bdd.save(dumpfilename+".cha", cha);
        System.out.println("actual: "+(long) actual.satCount(Iset.and(Zset).and(V2.set()))+" relations, "+actual.nodeCount()+" nodes");
        bdd.save(dumpfilename+".actual", actual);
        System.out.println("formal: "+(long) formal.satCount(this.MZset.and(V2.set()))+" relations, "+formal.nodeCount()+" nodes");
        bdd.save(dumpfilename+".formal", formal);
        System.out.println("Iret: "+(long) Iret.satCount(IV1set)+" relations, "+Iret.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Iret", Iret);
        System.out.println("Mret: "+(long) Mret.satCount(Mset.and(V2.set()))+" relations, "+Mret.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Mret", Mret);
        System.out.println("Ithr: "+(long) Ithr.satCount(IV1set)+" relations, "+Ithr.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Ithr", Ithr);
        System.out.println("Mthr: "+(long) Mthr.satCount(Mset.and(V2.set()))+" relations, "+Mthr.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Mthr", Mthr);
        System.out.println("mI: "+(long) mI.satCount(INset.and(Mset))+" relations, "+mI.nodeCount()+" nodes");
        bdd.save(dumpfilename+".mI", mI);
        System.out.println("mV: "+(long) mV.satCount(Mset.and(V1.set()))+" relations, "+mV.nodeCount()+" nodes");
        bdd.save(dumpfilename+".mV", mV);
        
        System.out.println("hP: "+(long) hP.satCount(H1FH2set)+" relations, "+hP.nodeCount()+" nodes");
        bdd.save(dumpfilename+".hP", hP);
        System.out.println("IE: "+(long) IE.satCount(IMset)+" relations, "+IE.nodeCount()+" nodes");
        bdd.save(dumpfilename+".IE", IE);
        if (IEcs != null) {
            System.out.println("IEcs: "+(long) IEcs.satCount(IMset.and(V1cV2cset))+" relations, "+IEcs.nodeCount()+" nodes");
            bdd.save(dumpfilename+".IEcs", IEcs);
        }
        if (vPfilter != null) {
            System.out.println("vPfilter: "+(long) vPfilter.satCount(V1.set().and(H1.set()))+" relations, "+vPfilter.nodeCount()+" nodes");
            bdd.save(dumpfilename+".vPfilter", vPfilter);
        }
        if (hPfilter != null) {
            System.out.println("hPfilter: "+(long) hPfilter.satCount(H1.set().and(Fset).and(H1.set()))+" relations, "+hPfilter.nodeCount()+" nodes");
            bdd.save(dumpfilename+".hPfilter", hPfilter);
        }
        if (IEfilter != null) {
            System.out.println("IEfilter: "+IEfilter.nodeCount()+" nodes");
            bdd.save(dumpfilename+".IEfilter", IEfilter);
        }
        System.out.println("visited: "+(long) visited.satCount(Mset)+" relations, "+visited.nodeCount()+" nodes");
        bdd.save(dumpfilename+".visited", visited);
        
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
        dumpConfig(dos);
        dos.close();
        
        System.out.print("Dumping maps...");
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
        System.out.println("done.");
    }

    public static PA loadResults(String bddfactory, String loaddir, String loadfilename) throws IOException {
        PA pa = new PA();
        DataInputStream di;
        di = new DataInputStream(new FileInputStream(loaddir+loadfilename+".config"));
        pa.loadConfig(di);
        di.close();
        System.out.print("Initializing...");
        pa.initializeBDD(bddfactory);
        System.out.println("done.");
        
        System.out.print("Loading results from "+loaddir+loadfilename+"...");
        if (loaddir.length() == 0) loaddir = "."+System.getProperty("file.separator");
        File dir = new File(loaddir);
        final String prefix = loadfilename + ".";
        File[] files = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(prefix);
            }
        });
        for (int i = 0; i < files.length; ++i) {
            File f = files[i];
            if (f.isDirectory()) continue;
            String name = f.getName().substring(prefix.length());
            try {
                Field field = PA.class.getDeclaredField(name);
                if (field == null) continue;
                if (field.getType() == org.sf.javabdd.BDD.class) {
                    System.out.print(name+": ");
                    BDD b = pa.bdd.load(f.getAbsolutePath());
                    System.out.print(b.nodeCount()+" nodes, ");
                    field.set(pa, b);
                } else if (field.getType() == IndexMap.class) {
                    System.out.print(name+": ");
                    di = new DataInputStream(new FileInputStream(f));
                    IndexMap m = IndexMap.load(name, di);
                    di.close();
                    System.out.print(m.size()+" entries, ");
                    field.set(pa, m);
                } else {
                    System.out.println();
                    System.out.println("Cannot load field: "+field);
                }
            } catch (NoSuchFieldException e) {
            } catch (IllegalArgumentException e) {
                Assert.UNREACHABLE();
            } catch (IllegalAccessException e) {
                Assert.UNREACHABLE();
            }
        }
        System.out.println("done.");
        
        // Set types for loaded BDDs.
        if (pa.A instanceof TypedBDD)
            if (pa.CONTEXT_SENSITIVE || pa.OBJECT_SENSITIVE)
                ((TypedBDD) pa.A).setDomains(pa.V1, pa.V1c, pa.V2, pa.V2c);
            else
                ((TypedBDD) pa.A).setDomains(pa.V1, pa.V2);
        if (pa.vP instanceof TypedBDD)
            if (pa.CONTEXT_SENSITIVE || pa.OBJECT_SENSITIVE)
                ((TypedBDD) pa.vP).setDomains(pa.V1, pa.V1c, pa.H1, pa.H1c);
            else
                ((TypedBDD) pa.vP).setDomains(pa.V1, pa.H1);
        if (pa.S instanceof TypedBDD)
            if (pa.CONTEXT_SENSITIVE || pa.OBJECT_SENSITIVE)
                ((TypedBDD) pa.S).setDomains(pa.V1, pa.V1c, pa.F, pa.V2, pa.V2c);
            else
                ((TypedBDD) pa.S).setDomains(pa.V1, pa.F, pa.V2);
        if (pa.L instanceof TypedBDD)
            if (pa.CONTEXT_SENSITIVE || pa.OBJECT_SENSITIVE)
                ((TypedBDD) pa.L).setDomains(pa.V1, pa.V1c, pa.F, pa.V2, pa.V2c);
            else
                ((TypedBDD) pa.L).setDomains(pa.V1, pa.F, pa.V2);
        if (pa.vT instanceof TypedBDD)
            ((TypedBDD) pa.vT).setDomains(pa.V1, pa.T1);
        if (pa.hT instanceof TypedBDD)
            ((TypedBDD) pa.hT).setDomains(pa.H1, pa.T2);
        if (pa.aT instanceof TypedBDD)
            ((TypedBDD) pa.aT).setDomains(pa.T1, pa.T2);
        if (pa.cha instanceof TypedBDD)
            ((TypedBDD) pa.cha).setDomains(pa.T2, pa.N, pa.M);
        if (pa.actual instanceof TypedBDD)
            ((TypedBDD) pa.actual).setDomains(pa.I, pa.Z, pa.V2);
        if (pa.formal instanceof TypedBDD)
            ((TypedBDD) pa.formal).setDomains(pa.M, pa.Z, pa.V1);
        if (pa.Iret instanceof TypedBDD)
            ((TypedBDD) pa.Iret).setDomains(pa.I, pa.V1);
        if (pa.Mret instanceof TypedBDD)
            ((TypedBDD) pa.Mret).setDomains(pa.M, pa.V2);
        if (pa.Ithr instanceof TypedBDD)
            ((TypedBDD) pa.Ithr).setDomains(pa.I, pa.V1);
        if (pa.Mthr instanceof TypedBDD)
            ((TypedBDD) pa.Mthr).setDomains(pa.M, pa.V2);
        if (pa.mI instanceof TypedBDD)
            ((TypedBDD) pa.mI).setDomains(pa.M, pa.I, pa.N);
        if (pa.mV instanceof TypedBDD)
            ((TypedBDD) pa.mV).setDomains(pa.M, pa.V1);
        if (pa.sync instanceof TypedBDD)
            ((TypedBDD) pa.sync).setDomains(pa.V1);
        
        if (pa.fT instanceof TypedBDD)
            ((TypedBDD) pa.fT).setDomains(pa.F, pa.T2);
        if (pa.fC instanceof TypedBDD)
            ((TypedBDD) pa.fC).setDomains(pa.F, pa.T2);

        if (pa.hP instanceof TypedBDD)
            if (pa.CONTEXT_SENSITIVE || pa.OBJECT_SENSITIVE)
                ((TypedBDD) pa.hP).setDomains(pa.H1, pa.H1c, pa.F, pa.H2, pa.H2c);
            else
                ((TypedBDD) pa.hP).setDomains(pa.H1, pa.F, pa.H2);
        if (pa.IE instanceof TypedBDD)
            ((TypedBDD) pa.IE).setDomains(pa.I, pa.M);
        if (pa.IEcs instanceof TypedBDD)
            ((TypedBDD) pa.IEcs).setDomains(pa.V2c, pa.I, pa.V1c, pa.M);
        if (pa.vPfilter instanceof TypedBDD)
            ((TypedBDD) pa.vPfilter).setDomains(pa.V1, pa.H1);
        if (pa.hPfilter instanceof TypedBDD)
            ((TypedBDD) pa.hPfilter).setDomains(pa.H1, pa.F, pa.H2);
        if (pa.IEfilter instanceof TypedBDD)
            ((TypedBDD) pa.IEfilter).setDomains(pa.V2c, pa.I, pa.V1c, pa.M);
        
        if (pa.visited instanceof TypedBDD)
            ((TypedBDD) pa.visited).setDomains(pa.M);
        
        return pa;
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
        out.writeBytes("CS="+(CONTEXT_SENSITIVE?"yes":"no")+"\n");
        out.writeBytes("OS="+(OBJECT_SENSITIVE?"yes":"no")+"\n");
        out.writeBytes("Order="+varorder+"\n");
        out.writeBytes("Reverse="+reverseLocal+"\n");
    }
    
    private void loadConfig(DataInput in) throws IOException {
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
            } else if (s1.equals("CS")) {
                CONTEXT_SENSITIVE = s2.equals("yes");
            } else if (s1.equals("OS")) {
                OBJECT_SENSITIVE = s2.equals("yes");
            } else if (s1.equals("Order")) {
                varorder = s2;
            } else if (s1.equals("Reverse")) {
                reverseLocal = s2.equals("true");
            } else {
                System.err.println("Unknown config option "+s);
            }
        }
        if (VC_BITS > 1 || HC_BITS > 1) {
            MAX_VC_BITS = VC_BITS;
            MAX_HC_BITS = HC_BITS;
        }
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
    
    // Map between thread run() methods and the ConcreteTypeNodes of the corresponding threads.
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
    
    public PathNumbering countCallGraph(CallGraph cg, ObjectCreationGraph ocg, boolean updateBits) {
        jq_Class jlt = PrimordialClassLoader.getJavaLangThread();
        jlt.prepare();
        jq_Class jlr = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;");
        jlr.prepare();
        Set fields = new HashSet();
        Set classes = new HashSet();
        int vars = 0, heaps = 0, bcodes = 0, methods = 0, calls = 0;
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (TRACE_OBJECT) out.println("Counting "+m);
            ++methods;
            jq_Class c = m.isStatic() ? null : m.getDeclaringClass();
            if (m.getBytecode() == null) {
                jq_Type retType = m.getReturnType();
                if (retType instanceof jq_Reference) {
                    boolean b = classes.add(retType);
                    if (b)
                        ++heaps;
                    if (ocg != null) {
                        ocg.addEdge(null, (Node) null, (jq_Reference) retType);
                    }
                }
                continue;
            }
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
                        if (ocg != null) {
                            ocg.addEdge(c, n, type);
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
        PathNumbering pn = new PathNumbering(varPathSelector);
        System.out.println("Thread runs="+thread_runs);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        System.out.println("Vars="+vars+" Heaps="+heaps+" Classes="+classes.size()+" Fields="+fields.size()+" Paths="+paths);
        if (updateBits) {
            V_BITS = BigInteger.valueOf(vars+256).bitLength();
            I_BITS = BigInteger.valueOf(calls).bitLength();
            H_BITS = BigInteger.valueOf(heaps+256).bitLength();
            F_BITS = BigInteger.valueOf(fields.size()+64).bitLength();
            T_BITS = BigInteger.valueOf(classes.size()+64).bitLength();
            N_BITS = I_BITS;
            M_BITS = BigInteger.valueOf(methods).bitLength() + 1;
            if (CONTEXT_SENSITIVE) {
                VC_BITS = paths.bitLength();
                VC_BITS = Math.min(MAX_VC_BITS, VC_BITS);
            }
            System.out.println(" V="+V_BITS+" I="+I_BITS+" H="+H_BITS+
                               " F="+F_BITS+" T="+T_BITS+" N="+N_BITS+
                               " M="+M_BITS+" VC="+VC_BITS);
        }
        return pn;
    }

    public final VarPathSelector varPathSelector = new VarPathSelector(MAX_VC_BITS);
    
    public static class VarPathSelector implements Selector {

        int maxBits;
        
        VarPathSelector(int max_bits) {
            this.maxBits = max_bits;
        }
        
        /* (non-Javadoc)
         * @see Util.Graphs.PathNumbering.Selector#isImportant(Util.Graphs.SCComponent, Util.Graphs.SCComponent)
         */
        public boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num) {
            if (num.bitLength() > maxBits) return false;
            return true;
        }
    }
    
    public final HeapPathSelector heapPathSelector = new HeapPathSelector();
    
    static Set polyClasses;
    static void initPolyClasses() {
        if (polyClasses != null) return;
        polyClasses = new HashSet();
        File f = new File("polyclasses");
        if (f.exists()) {
            try {
                DataInput in = new DataInputStream(new FileInputStream(f));
                for (;;) {
                    String s = in.readLine();
                    if (s == null) break;
                    polyClasses.add(jq_Type.parseType(s));
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    
    public class HeapPathSelector implements Selector {

        jq_Class collection_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/Collection;");
        jq_Class map_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/Map;");
        jq_Class throwable_class = (jq_Class) PrimordialClassLoader.getJavaLangThrowable();
        HeapPathSelector() {
            initPolyClasses();
            collection_class.prepare();
            map_class.prepare();
            throwable_class.prepare();
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
            if (m.getNameAndDesc() == main_method) return true;
            if (m.getNameAndDesc() == run_method) return true;
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
                if (!polyClasses.isEmpty() && !polyClasses.contains(type))
                    return false;
                if (type.isSubtypeOf(throwable_class))
                    return false;
                //if (!type.isSubtypeOf(collection_class) &&
                //    !type.isSubtypeOf(map_class))
                //    return false;
            }
            return true;
        }
    }

    public final ObjectPathSelector objectPathSelector = new ObjectPathSelector();
    
    public class ObjectPathSelector implements Selector {

        jq_Class throwable_class = (jq_Class) PrimordialClassLoader.getJavaLangThrowable();
        ObjectPathSelector() {
            throwable_class.prepare();
        }
        
        /* (non-Javadoc)
         * @see Util.Graphs.PathNumbering.Selector#isImportant(Util.Graphs.SCComponent, Util.Graphs.SCComponent)
         */
        public boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num) {
            Set s = scc2.nodeSet();
            Iterator i = s.iterator();
            Object o = i.next();
            if (i.hasNext()) {
                if (TRACE_OBJECT) out.println("No object sensitivity for "+s+": CYCLE");
                return false;
            }
            if (o instanceof jq_Array) {
                if (!((jq_Array) o).getElementType().isReferenceType()) {
                    if (TRACE_OBJECT) out.println("No object sensitivity for "+o+": PRIMITIVE ARRAY");
                    return false;
                }
            } else if (o instanceof jq_Class) {
                jq_Class c = (jq_Class) o;
                if (c == PrimordialClassLoader.getJavaLangString()) {
                    if (TRACE_OBJECT) out.println("No object sensitivity for "+c+": STRING");
                    return false;
                }
                c.prepare();
                if (c.isSubtypeOf(throwable_class)) {
                    if (TRACE_OBJECT) out.println("No object sensitivity for "+c+": THROWABLE");
                    return false;
                }
                boolean hasReferenceMember = false;
                jq_InstanceField[] f = c.getInstanceFields();
                for (int j = 0; j < f.length; ++j) {
                    if (f[j].getType().isReferenceType()) {
                        hasReferenceMember = true;
                        break;
                    }
                }
                if (!hasReferenceMember) {
                    if (TRACE_OBJECT) out.println("No object sensitivity for "+c+": NO REF FIELDS");
                    return false;
                }
            }
            return true;
        }
    }
    
    public PathNumbering countHeapNumbering(CallGraph cg, boolean updateBits) {
        if (VerifyAssertions)
            Assert._assert(CONTEXT_SENSITIVE);
        PathNumbering pn = new PathNumbering(heapPathSelector);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        System.out.println("Number of paths for heap context sensitivity: "+paths);
        if (updateBits) {
            HC_BITS = paths.bitLength();
            System.out.println("Heap context bits="+HC_BITS);
        }
        return pn;
    }
    
    void calculateIEfilter(CallGraph cg) {
        IEfilter = bdd.zero();
        IEcs = bdd.zero();
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
                IEfilter.orWith(context);
            }
        }
    }
    
    public BDD getV1H1Context(jq_Method m) {
        if (CONTEXT_SENSITIVE) {
            if (V1H1correspondence != null)
                return (BDD) V1H1correspondence.get(m);
            Range r1 = vCnumbering.getRange(m);
            BDD b = V1c.varRange(r1.low.longValue(), r1.high.longValue());
            b.andWith(H1c.ithVar(0));
            return b;
        } else if (OBJECT_SENSITIVE) {
            jq_Class c = m.isStatic() ? null : m.getDeclaringClass(); 
            BDD result = (BDD) V1H1correspondence.get(c);
            if (result == null) {
                if (TRACE_OBJECT) out.println("Note: "+c+" is not in object creation graph.");
                //result = V1c.ithVar(0);
                //result.andWith(H1c.ithVar(0));
                return result;
            }
            return result.id();
        } else {
            return null;
        }
    }
    
    public void buildObjectSensitiveV1H1(ObjectCreationGraph g) {
        if (TRACE_OBJECT) out.println("Building object-sensitive V1H1");
        V1H1correspondence = new HashMap();
        rangeMap = new HashMap();
        rangeMap.put(null, new Range(BigInteger.ZERO, BigInteger.ZERO));
        Navigator nav = g.getNavigator();
        for (Iterator i = Traversals.reversePostOrder(nav, g.getRoots()).iterator();
             i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof Node) {
                if (TRACE_OBJECT) out.println("Skipping "+o);
                continue;
            }
            jq_Reference c1 = (jq_Reference) o;
            Range r1 = oCnumbering.getRange(c1);
            if (c1 instanceof jq_Class) {
                jq_Class c = (jq_Class) c1;
                while (c != null) {
                    Range r = (Range) rangeMap.get(c);
                    if (r == null || r.high.longValue() < r1.high.longValue()) {
                        rangeMap.put(c, r1);
                    }
                    c = c.getSuperclass();
                }
            }
            if (TRACE_OBJECT) out.println(c1+" Range "+r1);
            
            BDD b = bdd.zero();
            for (Iterator j = nav.next(c1).iterator(); j.hasNext(); ) {
                Object p = j.next();
                Node node;
                jq_Reference c2;
                Range r2;
                if (TRACE_OBJECT) out.println("Edge "+c1+" -> "+p);
                if (p instanceof jq_Reference) {
                    // unknown creation site.
                    node = null;
                    c2 = (jq_Reference) p;
                    r2 = oCnumbering.getEdge(c1, c2);
                } else {
                    node = (Node) p;
                    Collection next = nav.next(node);
                    if (VerifyAssertions)
                        Assert._assert(next.size() == 1);
                    if (VerifyAssertions)
                        Assert._assert(r1.equals(oCnumbering.getEdge(c1, node)));
                    c2 = (jq_Reference) next.iterator().next();
                    r2 = oCnumbering.getEdge(node, c2);
                }
                
                int T_i = Tmap.get(c2);
                // class c1 creates a c2 object
                BDD T_bdd = T2.ithVar(T_i);
                BDD heap;
                if (node == null) {
                    // we don't know which creation site, so just use all sites that
                    // have the same type.
                    heap = hT.restrict(T_bdd);
                    if (VerifyAssertions)
                        Assert._assert(!heap.isZero(), c2.toString());
                } else {
                    int H_i = Hmap.get(node);
                    heap = H1.ithVar(H_i);
                }
                T_bdd.free();
                if (TRACE_OBJECT) out.println(c1+" creation site "+node+" "+c2+" Range: "+r2);
                BDD cm;
                cm = buildContextMap(V1c,
                                     PathNumbering.toBigInt(r1.low),
                                     PathNumbering.toBigInt(r1.high),
                                     H1c,
                                     PathNumbering.toBigInt(r2.low),
                                     PathNumbering.toBigInt(r2.high));
                cm.andWith(heap);
                b.orWith(cm);
            }
            if (TRACE_OBJECT) out.println("Registering V1H1 for "+c1);
            V1H1correspondence.put(c1, b);
        }
    }
    
    public void buildObjectSensitiveV1H1_(ObjectCreationGraph g) {
        V1H1correspondence = new HashMap();
        rangeMap = new HashMap();
        rangeMap.put(null, new Range(BigInteger.ZERO, BigInteger.ZERO));
        Navigator nav = g.getNavigator();
        for (Iterator i = Traversals.reversePostOrder(nav, g.getRoots()).iterator();
             i.hasNext(); ) {
            jq_Reference c1 = (jq_Reference) i.next();
            Range r1 = oCnumbering.getRange(c1);
            if (c1 instanceof jq_Class) {
                jq_Class c = (jq_Class) c1;
                while (c != null) {
                    Range r = (Range) rangeMap.get(c);
                    if (r == null || r.high.longValue() < r1.high.longValue()) {
                        rangeMap.put(c, r1);
                    }
                    c = c.getSuperclass();
                }
            }
            
            BDD b = bdd.zero();
            for (Iterator j = nav.next(c1).iterator(); j.hasNext(); ) {
                jq_Reference c2 = (jq_Reference) j.next();
                int T_i = Tmap.get(c2);
                // class c1 creates a c2 object
                BDD T_bdd = T2.ithVar(T_i);
                BDD heap = hT.restrict(T_bdd);
                T_bdd.free();
                Range r2 = oCnumbering.getEdge(c1, c2);
                BDD cm;
                cm = buildContextMap(V1c,
                                     PathNumbering.toBigInt(r1.low),
                                     PathNumbering.toBigInt(r1.high),
                                     H1c,
                                     PathNumbering.toBigInt(r2.low),
                                     PathNumbering.toBigInt(r2.high));
                cm.andWith(heap);
                b.orWith(cm);
            }
            V1H1correspondence.put(c1, b);
        }
    }
    
    Map V1H1correspondence;
    
    public void buildVarHeapCorrespondence(CallGraph cg) {
        if (VerifyAssertions)
            Assert._assert(CONTEXT_SENSITIVE);
        BDDPairing V2cH2ctoV1cH1c = bdd.makePair();
        V2cH2ctoV1cH1c.set(new BDDDomain[] {V2c, H2c}, new BDDDomain[] {V1c, H1c});
        
        V1H1correspondence = new HashMap();
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            Range r1 = vCnumbering.getRange(m);
            Range r2 = hCnumbering.getRange(m);
            BDD relation;
            if (r1.equals(r2)) {
                relation = V1c.buildAdd(H1c, BigInteger.valueOf(r1.high.longValue()).bitLength(), 0);
                relation.andWith(V1c.varRange(r1.low.longValue(), r1.high.longValue()));
            } else {
                long v_val = r1.high.longValue()+1;
                long h_val = r2.high.longValue()+1;
                
                if (h_val == 1L) {
                    relation = V1c.varRange(r1.low.longValue(), r1.high.longValue());
                    relation.andWith(H1c.ithVar(0));
                } else {
                    int v_bits = BigInteger.valueOf(v_val).bitLength();
                    int h_bits = BigInteger.valueOf(h_val).bitLength();
                    // make it faster.
                    h_val = 1 << h_bits;
                    
                    int[] v = new int[v_bits];
                    for (int j = 0; j < v_bits; ++j) {
                        v[j] = V1c.vars()[j];
                    }
                    BDDBitVector v_vec = bdd.buildVector(v);
                    BDDBitVector z = v_vec.divmod(h_val, false);
                    
                    //int h_bits = BigInteger.valueOf(h_val).bitLength();
                    //int[] h = new int[h_bits];
                    //for (int j = 0; j < h_bits; ++j) {
                    //    h[j] = H1c.vars()[j];
                    //}
                    //BDDBitVector h_vec = bdd.buildVector(h);
                    BDDBitVector h_vec = bdd.buildVector(H1c);
                    
                    relation = bdd.one();
                    int n;
                    for (n = 0; n < h_vec.size() || n < v_vec.size(); n++) {
                        BDD a = (n < v_vec.size()) ? z.getBit(n) : bdd.zero();
                        BDD b = (n < h_vec.size()) ? h_vec.getBit(n) : bdd.zero();
                        relation.andWith(a.biimp(b));
                    }
                    for ( ; n < V1c.varNum() || n < H1c.varNum(); n++) {
                        if (n < V1c.varNum())
                            relation.andWith(bdd.nithVar(V1c.vars()[n]));
                        if (n < H1c.varNum())
                            relation.andWith(bdd.nithVar(H1c.vars()[n]));
                    }
                    relation.andWith(V1c.varRange(r1.low.longValue(), r1.high.longValue()));
                    //System.out.println(v_val+" / "+h_val+" = "+relation.and(V1c.varRange(0, 100)).toStringWithDomains());
                    v_vec.free(); h_vec.free(); z.free();
                }
            }
            V1H1correspondence.put(m, relation);
        }
    }
    
    public void buildExactVarHeapCorrespondence(CallGraph cg) {
        if (VerifyAssertions)
            Assert._assert(CONTEXT_SENSITIVE);
        BDDPairing V2cH2ctoV1cH1c = bdd.makePair();
        V2cH2ctoV1cH1c.set(new BDDDomain[] {V2c, H2c}, new BDDDomain[] {V1c, H1c});
        BDDPairing V2ctoV1c = bdd.makePair(V2c, V1c);
        BDDPairing H2ctoH1c = bdd.makePair(H2c, H1c);
        BDD V1cset = V1c.set();
        BDD H1cset = H1c.set();
        
        V1H1correspondence = new HashMap();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            jq_Method root = (jq_Method) i.next();
            Range r1 = vCnumbering.getRange(root);
            Range r2 = hCnumbering.getRange(root);
            BDD relation;
            if (r1.equals(r2)) {
                relation = V1c.buildAdd(H1c, BigInteger.valueOf(r1.high.longValue()).bitLength(), 0);
                relation.andWith(V1c.varRange(r1.low.longValue(), r1.high.longValue()));
                System.out.println("Root "+root+" numbering: "+relation.toStringWithDomains());
            } else {
                System.out.println("Root numbering doesn't match: "+root);
                // just intermix them all, because we don't know the mapping.
                relation = V1c.varRange(r1.low.longValue(), r1.high.longValue());
                relation.andWith(H1c.varRange(r2.low.longValue(), r2.high.longValue()));
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

// PA.java, created Oct 16, 2003 3:39:34 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.IPA;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigInteger;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_FakeInstanceMethod;
import joeq.Class.jq_Field;
import joeq.Class.jq_InstanceField;
import joeq.Class.jq_Method;
import joeq.Class.jq_NameAndDesc;
import joeq.Class.jq_Reference;
import joeq.Class.jq_Type;
import joeq.Compiler.Analysis.BDD.BuildBDDIR;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.Node;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import joeq.Compiler.Quad.CachedCallGraph;
import joeq.Compiler.Quad.CallGraph;
import joeq.Compiler.Quad.CodeCache;
import joeq.Compiler.Quad.LoadedCallGraph;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.Operand.RegisterOperand;
import joeq.Compiler.Quad.Operator.Invoke;
import joeq.Main.HostedVM;
import joeq.Util.Assert;
import joeq.Util.Collections.IndexMap;
import joeq.Util.Collections.IndexedMap;
import joeq.Util.Collections.Pair;
import joeq.Util.Graphs.Navigator;
import joeq.Util.Graphs.PathNumbering;
import joeq.Util.Graphs.RootPathNumbering;
import joeq.Util.Graphs.SCCPathNumbering;
import joeq.Util.Graphs.SCComponent;
import joeq.Util.Graphs.Traversals;
import joeq.Util.Graphs.PathNumbering.Range;
import joeq.Util.Graphs.SCCPathNumbering.Selector;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDBitVector;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

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

    static boolean WRITE_PARESULTS_BATCHFILE = !System.getProperty("pa.writeparesults", "yes").equals("no");

    boolean TRACE = !System.getProperty("pa.trace", "no").equals("no");
    boolean TRACE_SOLVER = !System.getProperty("pa.tracesolver", "no").equals("no");
    boolean TRACE_BIND = !System.getProperty("pa.tracebind", "no").equals("no");
    boolean TRACE_RELATIONS = !System.getProperty("pa.tracerelations", "no").equals("no");
    boolean TRACE_OBJECT = !System.getProperty("pa.traceobject", "no").equals("no");
    boolean TRACE_CONTEXT = !System.getProperty("pa.tracecontext", "no").equals("no");
    PrintStream out = System.out;
    boolean DUMP_INITIAL = !System.getProperty("pa.dumpinitial", "no").equals("no");
    boolean DUMP_RESULTS = !System.getProperty("pa.dumpresults", "yes").equals("no");
    static boolean USE_JOEQ_CLASSLIBS = !System.getProperty("pa.usejoeqclasslibs", "no").equals("no");

    boolean INCREMENTAL1 = !System.getProperty("pa.inc1", "yes").equals("no"); // incremental points-to
    boolean INCREMENTAL2 = !System.getProperty("pa.inc2", "yes").equals("no"); // incremental parameter binding
    boolean INCREMENTAL3 = !System.getProperty("pa.inc3", "yes").equals("no"); // incremental invocation binding
    
    boolean ADD_CLINIT = !System.getProperty("pa.clinit", "yes").equals("no");
    boolean ADD_THREADS = !System.getProperty("pa.threads", "yes").equals("no");
    boolean ADD_FINALIZERS = !System.getProperty("pa.finalizers", "yes").equals("no");
    boolean IGNORE_EXCEPTIONS = !System.getProperty("pa.ignoreexceptions", "no").equals("no");
    boolean FILTER_VP = !System.getProperty("pa.vpfilter", "yes").equals("no");
    boolean FILTER_HP = !System.getProperty("pa.hpfilter", "no").equals("no");
    boolean CARTESIAN_PRODUCT = !System.getProperty("pa.cp", "no").equals("no");
    boolean THREAD_SENSITIVE = !System.getProperty("pa.ts", "no").equals("no");
    boolean OBJECT_SENSITIVE = !System.getProperty("pa.os", "no").equals("no");
    boolean CONTEXT_SENSITIVE = !System.getProperty("pa.cs", "no").equals("no");
    boolean CS_CALLGRAPH = !System.getProperty("pa.cscg", "no").equals("no");
    boolean DISCOVER_CALL_GRAPH = !System.getProperty("pa.discover", "no").equals("no");
    boolean DUMP_DOTGRAPH = !System.getProperty("pa.dumpdotgraph", "no").equals("no");
    boolean FILTER_NULL = !System.getProperty("pa.filternull", "yes").equals("no");
    boolean LONG_LOCATIONS = !System.getProperty("pa.longlocations", "no").equals("no");
    boolean INCLUDE_UNKNOWN_TYPES = !System.getProperty("pa.unknowntypes", "yes").equals("no");
    boolean INCLUDE_ALL_UNKNOWN_TYPES = !System.getProperty("pa.allunknowntypes", "no").equals("no");
    int MAX_PARAMS = Integer.parseInt(System.getProperty("pa.maxparams", "4"));
    
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "2500000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "150000"));
    static String resultsFileName = System.getProperty("pa.results", "pa");
    static String callgraphFileName = System.getProperty("pa.callgraph", "callgraph");
    static String initialCallgraphFileName = System.getProperty("pa.icallgraph", callgraphFileName);
    
    boolean USE_VCONTEXT;
    boolean USE_HCONTEXT;
    
    Map newMethodSummaries = new HashMap();
    Set rootMethods = new HashSet();
    
    CallGraph cg;
    ObjectCreationGraph ocg;
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    BDDDomain V1c[], V2c[], H1c[], H2c[];
    
    int V_BITS=18, I_BITS=16, H_BITS=14, Z_BITS=5, F_BITS=13, T_BITS=12, N_BITS=13, M_BITS=14;
    int VC_BITS=0, HC_BITS=0;
    int MAX_VC_BITS = Integer.parseInt(System.getProperty("pa.maxvc", "48"));
    int MAX_HC_BITS = Integer.parseInt(System.getProperty("pa.maxhc", "6"));
    
    IndexMap/*Node*/ Vmap;
    IndexMap/*ProgramLocation*/ Imap;
    IndexedMap/*Node*/ Hmap;
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
    BDD NNfilter; // H1, non-null filter                (no context)
    BDD IEfilter; // V2cxIxV1cxM, context-sensitive edge filter
    
    BDD visited; // M, visited methods
    
    BDD staticCalls; // V1xIxM, statically-bound calls, only used for object-sensitive and cartesian product
    
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
    String varorder = System.getProperty("bddordering");
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDDPairing V1ctoV2c, V1cV2ctoV2cV1c, V1cH1ctoV2cV1c;
    BDDPairing T2toT1, T1toT2;
    BDDPairing H1toV1c[], V1ctoH1[]; BDD V1csets[], V1cH1equals[];
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1Fset, V2Fset, V1FV2set, V1H1set, H1Fset, H2Fset, H1H2set, H1FH2set;
    BDD IMset, INset, INH1set, INT2set, T2Nset, MZset;
    BDD V1cset, V2cset, H1cset, H2cset, V1cV2cset, V1cH1cset, H1cH2cset;
    BDD V1cdomain, V2cdomain, H1cdomain, H2cdomain;
    
    BDDDomain makeDomain(String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }
    
    public void initializeBDD(String bddfactory) {
        USE_VCONTEXT = VC_BITS > 0;
        USE_HCONTEXT = HC_BITS > 0;
        
        if (USE_VCONTEXT || USE_HCONTEXT) bddnodes *= 2;
        
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
        
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE || THREAD_SENSITIVE) {
            V1c = new BDDDomain[1];
            V2c = new BDDDomain[1];
            V1c[0] = makeDomain("V1c", VC_BITS);
            V2c[0] = makeDomain("V2c", VC_BITS);
        } else if (CARTESIAN_PRODUCT) {
            V1c = new BDDDomain[MAX_PARAMS];
            V2c = new BDDDomain[MAX_PARAMS];
            for (int i = 0; i < V1c.length; ++i) {
                V1c[i] = makeDomain("V1c"+i, H_BITS + HC_BITS);
            }
            for (int i = 0; i < V2c.length; ++i) {
                V2c[i] = makeDomain("V2c"+i, H_BITS + HC_BITS);
            }
        } else {
            V1c = V2c = new BDDDomain[0];
        }
        if (USE_HCONTEXT) {
            H1c = new BDDDomain[] { makeDomain("H1c", HC_BITS) };
            H2c = new BDDDomain[] { makeDomain("H2c", HC_BITS) };
        } else {
            H1c = H2c = new BDDDomain[0];
        }
        
        if (TRACE) out.println("Variable context domains: "+V1c.length);
        if (TRACE) out.println("Heap context domains: "+H1c.length);
        
        if (varorder == null) {
            // default variable orderings.
            if (CONTEXT_SENSITIVE || THREAD_SENSITIVE || OBJECT_SENSITIVE) {
                if (HC_BITS > 0) {
                    varorder = "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2xH2c_T2_H1xH1c";
                } else {
                    //varorder = "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2_T2_H1";
                    varorder = "N_F_I_M_Z_V2xV1_V2cxV1c_T1_H2_T2_H1";
                }
            } else if (CARTESIAN_PRODUCT) {
                varorder = "N_F_Z_I_M_T1_V2xV1_T2_H2xH1";
                for (int i = 0; i < V1c.length; ++i) {
                    varorder += "xV1c"+i+"xV2c"+i;
                }
            } else {
                //varorder = "N_F_Z_I_M_T1_V2xV1_H2_T2_H1";
                varorder = "N_F_I_M_Z_V2xV1_T1_H2_T2_H1";
            }
        }
        
        System.out.println("Using variable ordering "+varorder);
        int[] ordering = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(ordering);
        
        V1ctoV2c = bdd.makePair();
        V1ctoV2c.set(V1c, V2c);
        V1cV2ctoV2cV1c = bdd.makePair();
        V1cV2ctoV2cV1c.set(V1c, V2c);
        V1cV2ctoV2cV1c.set(V2c, V1c);
        if (OBJECT_SENSITIVE) {
            V1cH1ctoV2cV1c = bdd.makePair();
            V1cH1ctoV2cV1c.set(V1c, V2c);
            V1cH1ctoV2cV1c.set(H1c, V1c);
        }
        T2toT1 = bdd.makePair(T2, T1);
        T1toT2 = bdd.makePair(T1, T2);
        V1toV2 = bdd.makePair();
        V1toV2.set(V1, V2);
        V1toV2.set(V1c, V2c);
        V2toV1 = bdd.makePair();
        V2toV1.set(V2, V1);
        V2toV1.set(V2c, V1c);
        H1toH2 = bdd.makePair();
        H1toH2.set(H1, H2);
        H1toH2.set(H1c, H2c);
        H2toH1 = bdd.makePair();
        H2toH1.set(H2, H1);
        H2toH1.set(H2c, H1c);
        V1H1toV2H2 = bdd.makePair();
        V1H1toV2H2.set(V1, V2);
        V1H1toV2H2.set(H1, H2);
        V1H1toV2H2.set(V1c, V2c);
        V1H1toV2H2.set(H1c, H2c);
        V2H2toV1H1 = bdd.makePair();
        V2H2toV1H1.set(V2, V1);
        V2H2toV1H1.set(H2, H1);
        V2H2toV1H1.set(V2c, V1c);
        V2H2toV1H1.set(H2c, H1c);
        
        V1set = V1.set();
        if (V1c.length > 0) {
            V1cset = bdd.one();
            V1cdomain = bdd.one();
            for (int i = 0; i < V1c.length; ++i) {
                V1cset.andWith(V1c[i].set());
                V1cdomain.andWith(V1c[i].domain());
            }
            V1set.andWith(V1cset.id());
        }
        V2set = V2.set();
        if (V2c.length > 0) {
            V2cset = bdd.one();
            V2cdomain = bdd.one();
            for (int i = 0; i < V2c.length; ++i) {
                V2cset.andWith(V2c[i].set());
                V2cdomain.andWith(V2c[i].domain());
            }
            V2set.andWith(V2cset.id());
        }
        H1set = H1.set();
        if (H1c.length > 0) {
            H1cset = bdd.one();
            H1cdomain = bdd.one();
            for (int i = 0; i < H1c.length; ++i) {
                H1cset.andWith(H1c[i].set());
                H1cdomain.andWith(H1c[i].domain());
            }
            H1set.andWith(H1cset.id());
        }
        H2set = H2.set();
        if (H2c.length > 0) {
            H2cset = bdd.one();
            H2cdomain = bdd.one();
            for (int i = 0; i < H2c.length; ++i) {
                H2cset.andWith(H2c[i].set());
                H2cdomain.andWith(H2c[i].domain());
            }
            H2set.andWith(H2cset.id());
        }
        T1set = T1.set();
        T2set = T2.set();
        Fset = F.set();
        Mset = M.set();
        Nset = N.set();
        Iset = I.set();
        Zset = Z.set();
        V1cV2cset = (V1c.length > 0) ? V1cset.and(V2cset) : bdd.zero();
        H1cH2cset = (H1c.length > 0) ? H1cset.and(H2cset) : bdd.zero();
        if (V1c.length > 0) {
            V1cH1cset = (H1c.length > 0) ? V1cset.and(H1cset) : V1cset;
        } else {
            V1cH1cset = (H1c.length > 0) ? H1cset : bdd.zero();
        }
        V1V2set = V1set.and(V2set);
        V1FV2set = V1V2set.and(Fset);
        V1H1set = V1set.and(H1set);
        V1Fset = V1set.and(Fset);
        V2Fset = V2set.and(Fset);
        IMset = Iset.and(Mset);
        INset = Iset.and(Nset);
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
        
        if (OBJECT_SENSITIVE || CARTESIAN_PRODUCT) staticCalls = bdd.zero();
        
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
            old3_t6 = bdd.zero();
            old3_t9 = new BDD[MAX_PARAMS];
            for (int i = 0; i < old3_t9.length; ++i) {
                old3_t9[i] = bdd.zero();
            }
        }
        
        if (CARTESIAN_PRODUCT) {
            H1toV1c = new BDDPairing[MAX_PARAMS];
            V1ctoH1 = new BDDPairing[MAX_PARAMS];
            V1csets = new BDD[MAX_PARAMS];
            V1cH1equals = new BDD[MAX_PARAMS];
            for (int i = 0; i < MAX_PARAMS; ++i) {
                H1toV1c[i] = bdd.makePair(H1, V1c[i]);
                V1ctoH1[i] = bdd.makePair(V1c[i], H1);
                V1csets[i] = V1c[i].set();
                V1cH1equals[i] = H1.buildEquals(V1c[i]);
            }
        }
        
        if (USE_VCONTEXT) {
            IEcs = bdd.zero();
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
        if (OBJECT_SENSITIVE || CARTESIAN_PRODUCT) {
            BDD bdd1 = bdd.zero();
            for (Iterator j = thisptr.iterator(); j.hasNext(); ) {
                int V_i = Vmap.get(j.next());
                bdd1.orWith(V1.ithVar(V_i));
            }
            bdd1.andWith(I_bdd.id());
            int M_i = Mmap.get(target);
            bdd1.andWith(M.ithVar(M_i));
            if (TRACE_RELATIONS) out.println("Adding single-target call: "+bdd1.toStringWithDomains());
            staticCalls.orWith(bdd1);
        }
    }
    
    void addToIE(BDD I_bdd, jq_Method target) {
        int M2_i = Mmap.get(target);
        BDD bdd1 = M.ithVar(M2_i);
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to IE: "+bdd1.toStringWithDomains());
        if (USE_VCONTEXT && IEfilter != null) {
            // When doing context-sensitive analysis, we need to add to IEcs too.
            // This call edge is true under all contexts for this invocation.
            // "and"-ing with IEfilter achieves this.
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
    
    void addEmptyActual(BDD I_bdd, int z) {
        if (CARTESIAN_PRODUCT) {
            BDD bdd1 = V2.ithVar(0); // global node
            bdd1.andWith(Z.ithVar(z));
            bdd1.andWith(I_bdd.id());
            if (TRACE_RELATIONS) out.println("Adding empty to actual: "+bdd1.toStringWithDomains());
            actual.orWith(bdd1);
        }
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
        BDD context = bdd.one();
        if (USE_VCONTEXT) context.andWith(V1cdomain.id());
        if (USE_HCONTEXT) context.andWith(H1cdomain.id());
        addToVP(context, p, H_i);
        context.free();
    }
    
    void addToVP(BDD V1H1context, Node p, int H_i) {
        int V1_i = Vmap.get(p);
        BDD bdd1 = V1.ithVar(V1_i);
        bdd1.andWith(H1.ithVar(H_i));
        if (V1H1context != null) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToVP(BDD V_bdd, Node h) {
        BDD context = bdd.one();
        if (USE_VCONTEXT) context.andWith(V1cdomain.id());
        if (USE_HCONTEXT) context.andWith(H1cdomain.id());
        addToVP(context, V_bdd, h);
        context.free();
    }
    
    void addToVP(BDD V1H1context, BDD V_bdd, Node h) {
        int H_i = Hmap.get(h);
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(V_bdd.id());
        if (V1H1context != null) bdd1.andWith(V1H1context.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToA(int V1_i, int V2_i) {
        BDD context = USE_VCONTEXT ? V1cdomain.and(V2cdomain) : null;
        addToA(context, V1_i, V2_i);
        if (USE_VCONTEXT) context.free();
    }
    
    void addToA(BDD V1V2context, int V1_i, int V2_i) {
        BDD V_bdd = V1.ithVar(V1_i);
        addToA(V1V2context, V_bdd, V2_i);
        V_bdd.free();
    }
    
    void addToA(BDD V_bdd, int V2_i) {
        BDD context = USE_VCONTEXT ? V1cdomain.and(V2cdomain) : null;
        addToA(context, V_bdd, V2_i);
        if (USE_VCONTEXT) context.free();
    }
    
    void addToA(BDD V1V2context, BDD V_bdd, int V2_i) {
        BDD bdd1 = V2.ithVar(V2_i);
        bdd1.andWith(V_bdd.id());
        if (USE_VCONTEXT) bdd1.andWith(V1V2context.id());
        if (TRACE_RELATIONS) out.println("Adding to A: "+bdd1.toStringWithDomains());
        A.orWith(bdd1);
    }
    
    void addToS(BDD V_bdd, jq_Field f, Collection c) {
        BDD context = USE_VCONTEXT ? V1cdomain.and(V2cdomain) : null;
        addToS(context, V_bdd, f, c);
        if (USE_VCONTEXT) context.free();
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
            if (USE_VCONTEXT) bdd1.andWith(V1V2context.id());
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
            if (USE_VCONTEXT) bdd1.andWith(V1V2context.id());
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
        if (CONTEXT_SENSITIVE || THREAD_SENSITIVE) {
            Pair p = new Pair(LoadedCallGraph.mapCall(mc), callee);
            Range r_edge = vCnumbering.getEdge(p);
            Range r_caller = vCnumbering.getRange(mc.getMethod());
            if (r_edge == null) {
                out.println("Cannot find edge "+p);
                return V1cdomain.and(V2cdomain);
            }
            BDD context = buildContextMap(V2c[0],
                                          PathNumbering.toBigInt(r_caller.low),
                                          PathNumbering.toBigInt(r_caller.high),
                                          V1c[0],
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
                V1V2context = V1cdomain.and(V2cdomain);
                return V1V2context;
            }
            if (one_to_one) {
                int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
                V1V2context = V1c[0].buildAdd(V2c[0], bits, 0L);
                V1V2context.andWith(V1c[0].varRange(r.low.longValue(), r.high.longValue()));
            } else {
                V1V2context = V1c[0].varRange(r.low.longValue(), r.high.longValue());
                V1V2context.andWith(V2c[0].varRange(r.low.longValue(), r.high.longValue()));
            }
            return V1V2context;
        } else if (CARTESIAN_PRODUCT) {
            throw new Error();
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
        if (CONTEXT_SENSITIVE || THREAD_SENSITIVE) {
            Range r = vCnumbering.getRange(m);
            int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
            if (TRACE_CONTEXT) out.println("Range to "+m+" = "+r+" ("+bits+" bits)");
            BDD V1V2context = V1c[0].buildAdd(V2c[0], bits, 0L);
            V1V2context.andWith(V1c[0].varRange(r.low.longValue(), r.high.longValue()));
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
                V1V2context = V1c[0].ithVar(0);
                V1V2context.andWith(V2c[0].ithVar(0));
                return V1V2context;
            }
            int bits = BigInteger.valueOf(r.high.longValue()).bitLength();
            V1V2context = V1c[0].buildAdd(V2c[0], bits, 0L);
            V1V2context.andWith(V1c[0].varRange(r.low.longValue(), r.high.longValue()));
            return V1V2context;
        } else if (CARTESIAN_PRODUCT) {
            BDD V1V2context = bdd.one();
            for (int i = 0; i < MAX_PARAMS; ++i) {
                V1V2context.andWith(V1c[i].buildEquals(V2c[i]));
            }
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
    
    jq_Class object_class = PrimordialClassLoader.getJavaLangObject();
    jq_Method javaLangObject_clone;
    {
        object_class.prepare();
        javaLangObject_clone = object_class.getDeclaredInstanceMethod(new jq_NameAndDesc("clone", "()Ljava/lang/Object;"));
    }
    jq_Class cloneable_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Cloneable;");
    jq_Class throwable_class = (jq_Class) PrimordialClassLoader.getJavaLangThrowable();
    jq_Method javaLangObject_fakeclone = jq_FakeInstanceMethod.fakeMethod(object_class, 
                                                MethodSummary.fakeCloneName, "()Ljava/lang/Object;");

    private jq_Method fakeCloneIfNeeded(jq_Type t) {
        jq_Method m = javaLangObject_clone;
        if (t instanceof jq_Class) {
            jq_Class c = (jq_Class)t;
            if (!c.isInterface() && c.implementsInterface(cloneable_class)) {
                m = jq_FakeInstanceMethod.fakeMethod(c, MethodSummary.fakeCloneName, "()"+t.getDesc());
                boolean mustvisit = (cg != null) ? cg.getAllMethods().contains(m) : true;
                if (mustvisit)
                    visitMethod(m);
            }
        }
        // TODO: handle cloning of arrays
        return m;
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
        
        // build up 'hT', 'NNfilter', and identify clinit, thread run, finalizers.
        if (!FILTER_NULL && NNfilter == null) NNfilter = bdd.zero();
        int Hsize = Hmap.size();
        for (int H_i = last_H; H_i < Hsize; ++H_i) {
            Node n = (Node) Hmap.get(H_i);

            if (!FILTER_NULL && !isNullConstant(n))
                NNfilter.orWith(H1.ithVar(H_i));

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
        
        // add types for UnknownTypeNodes to 'hT'
        if (INCLUDE_UNKNOWN_TYPES) {
            for (int H_i = last_H; H_i < Hsize; ++H_i) {
                Node n = (Node) Hmap.get(H_i);
                if (!(n instanceof UnknownTypeNode))
                    continue;
                jq_Reference type = n.getDeclaredType();
                if (type == null)
                    continue;
                if (!INCLUDE_ALL_UNKNOWN_TYPES && (type == object_class || type == throwable_class)) {
                    System.out.println("warning: excluding UnknownTypeNode "+type.getName()+"* from hT: H1("+H_i+")");
                } else {
                    // conservatively say that it can be any known subtype.
                    BDD T_i = T1.ithVar(Tmap.get(type));
                    BDD Tsub = aT.relprod(T_i, T1set);
                    addToHT(H_i, Tsub);
                    Tsub.free();
                    T_i.free();
                }
            }
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
                if ((m == javaLangObject_clone && t != object_class) || n == javaLangObject_fakeclone) {
                    m = fakeCloneIfNeeded(t);                                   // for t.clone()
                    addToCHA(T_bdd, Nmap.get(javaLangObject_fakeclone), m);     // for super.clone()
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
                int context_j = context_i + vCnumbering.getRange(m).low.intValue();
                System.out.println("Thread "+h+" index "+context_j);
                //context = H1c.ithVar(context_i);
                context = H1cdomain.id();
                //context.andWith(V1c.ithVar(context_j));
                context.andWith(V1cdomain.id());
                addToVP(context, p, H_i);
                context.free();
            } else {
                addToVP(p, H_i);
            }
        }
    }
    
    public void solvePointsTo() {
        try {
            dumpBDDRelations();
        } catch (IOException x) {}
        if (INCREMENTAL1) {
            solvePointsTo_incremental();
            return;
        }
        
        BDD old_vP;
        BDD old_hP = bdd.zero();
        for (int outer = 1; ; ++outer) {
            for (int inner = 1; ; ++inner) {
                old_vP = vP.id();
                
                // Rule 1
                BDD t1 = vP.replace(V1toV2); // V2xH1
                if (TRACE_SOLVER) out.println("Inner #"+inner+": rename V1toV2: vP "+vP.nodeCount()+" -> "+t1.nodeCount());
                BDD t2 = A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
                if (TRACE_SOLVER) out.println("Inner #"+inner+": relprod A "+A.nodeCount()+" -> "+t2.nodeCount());
                t1.free();
                if (FILTER_VP) t2.andWith(vPfilter.id());
                if (FILTER_VP && TRACE_SOLVER) out.println("Inner #"+inner+": and vPfilter "+vPfilter.nodeCount()+" -> "+t2.nodeCount());
                if (TRACE_SOLVER) out.print("Inner #"+inner+": or vP "+vP.nodeCount()+" -> ");
                vP.orWith(t2);
                if (TRACE_SOLVER) out.println(vP.nodeCount());
                
                boolean done = vP.equals(old_vP); 
                old_vP.free();
                if (done) break;
            }
            
            // Rule 2
            BDD t3 = S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            if (!FILTER_NULL) t3.andWith(NNfilter.id());
            BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
            BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
            t3.free(); t4.free();
            if (FILTER_HP) t5.andWith(hPfilter.id());
            hP.orWith(t5);

            if (TRACE_SOLVER) out.println("Outer #"+outer+": hP "+hP.nodeCount());
            
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
            if (TRACE_SOLVER) out.println("Outer #"+outer+": vP "+vP.nodeCount());
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
            if (TRACE_SOLVER) out.println("New A: "+new_A.nodeCount());
            BDD t1 = vP.replace(V1toV2); // V2xH1
            if (TRACE_SOLVER) out.println("New A: rename V1toV2: vP "+vP.nodeCount()+" -> "+t1.nodeCount());
            BDD t2 = new_A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
            if (TRACE_SOLVER) out.println("New A: relprod new_A "+new_A.nodeCount()+" -> "+t2.nodeCount());
            new_A.free(); t1.free();
            if (FILTER_VP) t2.andWith(vPfilter.id());
            if (FILTER_VP && TRACE_SOLVER) out.println("New A: and vPfilter "+vPfilter.nodeCount()+" -> "+t2.nodeCount());
            if (TRACE_SOLVER) out.print("New A: or vP "+vP.nodeCount()+" -> ");
            vP.orWith(t2);
            if (TRACE_SOLVER) out.println(vP.nodeCount());
        }
        old1_A = A.id();
        
        // handle new S
        BDD new_S = S.apply(old1_S, BDDFactory.diff);
        old1_S.free();
        if (!new_S.isZero()) {
            if (TRACE_SOLVER) out.println("New S: "+new_S.nodeCount());
            BDD t3 = new_S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            if (TRACE_SOLVER) out.println("New S: relprod: vP "+vP.nodeCount()+" -> "+t3.nodeCount());
            new_S.free();
            if (!FILTER_NULL) t3.andWith(NNfilter.id());
            if (!FILTER_NULL && TRACE_SOLVER) out.println("New S: and NNfilter "+NNfilter.nodeCount()+" -> "+t3.nodeCount());
            BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
            if (TRACE_SOLVER) out.println("New S: replace vP "+vP.nodeCount()+" -> "+t4.nodeCount());
            BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
            if (TRACE_SOLVER) out.println("New S: relprod -> "+t5.nodeCount());
            t3.free(); t4.free();
            if (FILTER_HP) t5.andWith(hPfilter.id());
            if (FILTER_HP && TRACE_SOLVER) out.println("New S: and hPfilter "+hPfilter.nodeCount()+" -> "+t5.nodeCount());
            if (TRACE_SOLVER) out.print("New S: or hP "+hP.nodeCount()+" -> ");
            hP.orWith(t5);
            if (TRACE_SOLVER) out.println(hP.nodeCount());
        }
        old1_S = S.id();
        
        // handle new L
        BDD new_L = L.apply(old1_L, BDDFactory.diff);
        old1_L.free();
        if (!new_L.isZero()) {
            if (TRACE_SOLVER) out.println("New L: "+new_L.nodeCount());
            BDD t6 = new_L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
            if (TRACE_SOLVER) out.println("New L: relprod: vP "+vP.nodeCount()+" -> "+t6.nodeCount());
            BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
            if (TRACE_SOLVER) out.println("New L: relprod: hP "+hP.nodeCount()+" -> "+t7.nodeCount());
            t6.free();
            t7.replaceWith(V2H2toV1H1); // V1xH1
            if (TRACE_SOLVER) out.println("New L: replace: "+t7.nodeCount());
            if (FILTER_VP) t7.andWith(vPfilter.id());
            if (TRACE_SOLVER) out.print("New L: or vP "+vP.nodeCount()+" -> ");
            vP.orWith(t7);
            if (TRACE_SOLVER) out.println(vP.nodeCount());
        }
        old1_L = L.id();
        
        for (int outer = 1; ; ++outer) {
            BDD new_vP_inner = vP.apply(old1_vP, BDDFactory.diff);
            int inner;
            for (inner = 1; !new_vP_inner.isZero() && inner < 256; ++inner) {
                if (TRACE_SOLVER)
                    out.println("Inner #"+inner+": new vP "+new_vP_inner.nodeCount());
                
                // Rule 1
                BDD t1 = new_vP_inner.replace(V1toV2); // V2xH1
                if (TRACE_SOLVER) out.println("Inner #"+inner+": rename V1toV2: "+t1.nodeCount());
                new_vP_inner.free();
                BDD t2 = A.relprod(t1, V2set); // V1xV2 x V2xH1 = V1xH1
                if (TRACE_SOLVER) out.println("Inner #"+inner+": relprod A: "+A.nodeCount()+" -> "+t2.nodeCount());
                t1.free();
                if (FILTER_VP) t2.andWith(vPfilter.id());
                if (FILTER_VP && TRACE_SOLVER) out.println("Inner #"+inner+": and vPfilter "+vPfilter.nodeCount()+" -> "+t2.nodeCount());
                
                BDD old_vP_inner = vP.id();
                vP.orWith(t2);
                if (TRACE_SOLVER) out.println("Inner #"+inner+": or vP "+old_vP_inner.nodeCount()+" -> "+vP.nodeCount());
                new_vP_inner = vP.apply(old_vP_inner, BDDFactory.diff);
                if (TRACE_SOLVER) out.println("Inner #"+inner+": diff vP -> "+new_vP_inner.nodeCount());
                old_vP_inner.free();
            }
            
            BDD new_vP = vP.apply(old1_vP, BDDFactory.diff);
            if (TRACE_SOLVER) out.println("Outer #"+outer+": diff vP "+vP.nodeCount()+" - "+old1_vP.nodeCount()+" = "+new_vP.nodeCount());
            old1_vP.free();
            
            {
                // Rule 2
                BDD t3 = S.relprod(new_vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S: relprod "+S.nodeCount()+" -> "+t3.nodeCount());
                if (!FILTER_NULL) t3.andWith(NNfilter.id());
                if (!FILTER_NULL && TRACE_SOLVER) out.println("Outer #"+outer+" S: and NNfilter "+NNfilter.nodeCount()+" -> "+t3.nodeCount());
                BDD t4 = vP.replace(V1H1toV2H2); // V2xH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S: replace "+vP.nodeCount()+" -> "+t4.nodeCount());
                BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S: relprod -> "+t5.nodeCount());
                t3.free(); t4.free();
                if (FILTER_HP) t5.andWith(hPfilter.id());
                if (FILTER_HP && TRACE_SOLVER) out.println("Outer #"+outer+" S: and hPfilter "+hPfilter.nodeCount()+" -> "+t5.nodeCount());
                if (TRACE_SOLVER) out.print("Outer #"+outer+" S: or hP "+hP.nodeCount()+" -> ");
                hP.orWith(t5);
                if (TRACE_SOLVER) out.println(hP.nodeCount());
            }
            {
                // Rule 2
                BDD t3 = S.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S': relprod "+S.nodeCount()+", "+vP.nodeCount()+" -> "+t3.nodeCount());
                if (!FILTER_NULL) t3.andWith(NNfilter.id());
                if (!FILTER_NULL && TRACE_SOLVER) out.println("Outer #"+outer+" S': and NNfilter "+NNfilter.nodeCount()+" -> "+t3.nodeCount());
                BDD t4 = new_vP.replace(V1H1toV2H2); // V2xH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S': replace "+new_vP.nodeCount()+" -> "+t4.nodeCount());
                BDD t5 = t3.relprod(t4, V2set); // H1xFxV2 x V2xH2 = H1xFxH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" S': relprod -> "+t5.nodeCount());
                t3.free(); t4.free();
                if (FILTER_HP) t5.andWith(hPfilter.id());
                if (FILTER_HP && TRACE_SOLVER) out.println("Outer #"+outer+" S': and hPfilter "+hPfilter.nodeCount()+" -> "+t5.nodeCount());
                if (TRACE_SOLVER) out.print("Outer #"+outer+" S': or hP "+hP.nodeCount()+" -> ");
                hP.orWith(t5);
                if (TRACE_SOLVER) out.println(hP.nodeCount());
            }

            old1_vP = vP.id();
            
            BDD new_hP = hP.apply(old1_hP, BDDFactory.diff);
            if (TRACE_SOLVER) out.println("Outer #"+outer+": diff hP "+hP.nodeCount()+" - "+old1_hP.nodeCount()+" = "+new_hP.nodeCount());
            if (new_hP.isZero() && new_vP.isZero() && inner < 256) break;
            old1_hP = hP.id();
            
            {
                // Rule 3
                BDD t6 = L.relprod(new_vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L: relprod "+L.nodeCount()+", "+new_vP.nodeCount()+" -> "+t6.nodeCount());
                BDD t7 = t6.relprod(hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L: relprod "+hP.nodeCount()+" -> "+t7.nodeCount());
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L: replace "+t7.nodeCount());
                if (FILTER_VP) t7.andWith(vPfilter.id());
                if (FILTER_VP && TRACE_SOLVER) out.println("Outer #"+outer+" L: and vPfilter "+vPfilter.nodeCount()+" -> "+t7.nodeCount());
                if (TRACE_SOLVER) out.print("Outer #"+outer+" L: or vP "+vP.nodeCount()+" -> ");
                vP.orWith(t7);
                if (TRACE_SOLVER) out.println(vP.nodeCount());
            }
            {
                // Rule 3
                BDD t6 = L.relprod(vP, V1set); // V1xFxV2 x V1xH1 = H1xFxV2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L': relprod "+L.nodeCount()+", "+vP.nodeCount()+" -> "+t6.nodeCount());
                BDD t7 = t6.relprod(new_hP, H1Fset); // H1xFxV2 x H1xFxH2 = V2xH2
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L': relprod "+new_hP.nodeCount()+" -> "+t7.nodeCount());
                t6.free();
                t7.replaceWith(V2H2toV1H1); // V1xH1
                if (TRACE_SOLVER) out.println("Outer #"+outer+" L': replace "+t7.nodeCount());
                if (FILTER_VP) t7.andWith(vPfilter.id());
                if (FILTER_VP && TRACE_SOLVER) out.println("Outer #"+outer+" L': and vPfilter "+vPfilter.nodeCount()+" -> "+t7.nodeCount());
                if (TRACE_SOLVER) out.print("Outer #"+outer+" L': or vP "+vP.nodeCount()+" -> ");
                vP.orWith(t7);
                if (TRACE_SOLVER) out.println(vP.nodeCount());
            }
        }
    }
    
    public void dumpWithV1c(BDD z, BDD set) {
        BDD a = z.exist(V1cset);
        for (Iterator i = a.iterator(set); i.hasNext(); ) {
            BDD b = (BDD) i.next();
            System.out.println(b.toStringWithDomains(TS));
            b.andWith(z.id());
            BDD c = b.exist(set);
            if (c.isOne()) {
                System.out.println("    under all contexts");
            } else {
                System.out.print("    under context ");
                Assert._assert(!c.isZero());
                for (int j = 0; j < V1csets.length; ++j) {
                    BDD d = c.id();
                    for (int k = 0; k < V1csets.length; ++k) {
                        if (k == j) continue;
                        BDD e = d.exist(V1csets[k]);
                        d.free();
                        d = e;
                    }
                    if (d.isOne()) System.out.print("*");
                    else if (d.isZero()) System.out.print("0");
                    else if (d.satCount(V1csets[j]) > 10) System.out.print("many");
                    else for (Iterator k = d.iterator(V1csets[j]); k.hasNext(); ) {
                        BDD e = (BDD) k.next();
                        long v = e.scanVar(V1c[j]);
                        long mask = (1L << (H_BITS)) - 1;
                        long val = v & mask;
                        if (val != 0) System.out.print(TS.elementName(H1.getIndex(), val));
                        else System.out.print('_');
                        if (k.hasNext()) System.out.print(',');
                    }
                    if (j < MAX_PARAMS-1) System.out.print("|");
                }
                System.out.println();
            }
        }
    }
    
    public void dumpIEcs() {
        for (Iterator i = IEcs.iterator(Iset); i.hasNext(); ) {
            BDD q = (BDD) i.next(); // V2cxIxV1cxM
            long I_i = q.scanVar(I);
            System.out.println("Invocation site "+TS.elementName(I.getIndex(), I_i));
            BDD a = q.exist(IMset.and(V1cset)); // V2c
            Iterator k = null;
            boolean bool1;
            if (a.isOne()) {
                System.out.println("    under all contexts");
                bool1 = true;
            } else if (a.satCount(V2cset) > 16) {
                System.out.println("    under many contexts");
                bool1 = true;
            } else {
                k = q.iterator(V2cset);
                bool1 = false;
            }
            if (bool1) k = Collections.singleton(q).iterator();

            for ( ; k.hasNext(); ) {
                BDD s = (BDD) k.next(); // V2cxIxV1cxM
                if (!bool1) {
                    System.out.println("    under context "+s.exist(IMset.and(V1cset)).toStringWithDomains(TS));
                }
                for (Iterator j = s.iterator(Mset); j.hasNext(); ) {
                    BDD r = (BDD) j.next(); // V2cxIxV1cxM
                    long M_i = r.scanVar(M);
                    System.out.println(" calls "+TS.elementName(M.getIndex(), M_i));
                    BDD b = r.exist(IMset.and(V2cset));
                    if (b.isOne()) {
                        System.out.println("        all contexts");
                    } else if (b.satCount(V1cset) > 16) {
                        System.out.println("        many contexts");
                    } else {
                        for (Iterator m = r.iterator(V1cset); m.hasNext(); ) {
                            BDD t = (BDD) m.next();
                            System.out.println("        context "+s.exist(IMset.and(V2cset)).toStringWithDomains(TS));
                        }
                    }
                }
            }
        }
    }
    
    public void dumpVP(BDD my_vP) {
        for (Iterator i = my_vP.iterator(V1.set()); i.hasNext(); ) {
            BDD q = (BDD) i.next();
            long V_i = q.scanVar(V1);
            System.out.println("Variable "+TS.elementName(V1.getIndex(), V_i)+" points to:");
            for (Iterator j = q.iterator(H1.set()); j.hasNext(); ) {
                BDD r = (BDD) j.next();
                long H_i = r.scanVar(H1);
                System.out.println("  "+TS.elementName(H1.getIndex(), H_i));
                if (USE_VCONTEXT) {
                    BDD a = r.exist(V1.set().and(H1set));
                    if (a.isOne()) {
                        System.out.println("    under all contexts");
                    } else {
                        System.out.print("    under context ");
                        for (int m = 0; m < MAX_PARAMS; ++m) {
                            if (m > 0) System.out.print("|");
                            BDD b = a.id();
                            for (int k = 0; k < V1csets.length; ++k) {
                                if (k == m) continue;
                                BDD c = b.exist(V1csets[k]);
                                b.free();
                                b = c;
                            }
                            if (b.isOne()) {
                                System.out.print("*");
                            } else if (b.satCount(V1csets[m]) > 100) {
                                System.out.print("many");
                            } else for (Iterator k = b.iterator(V1csets[m]); k.hasNext(); ) {
                                BDD s = (BDD) k.next();
                                long foo = s.scanVar(V1c[m]);
                                System.out.print(TS.elementName(H1.getIndex(), foo));
                                if (k.hasNext()) System.out.print(",");
                            }
                        }
                        System.out.println();
                    }
                }
            }
        }
    }
    
    // t1 = actual x (z=0)
    // t3 = t1 x mI
    // t4 = t3 x vP
    // t5 = t4 x hT
    // t6 = t5 x cha
    // IE |= t6
    /** Uses points-to information to bind virtual call sites.  (Adds to IE/IEcs.) */
    public void bindInvocations() {
        if (INCREMENTAL3 && !OBJECT_SENSITIVE) {
            bindInvocations_incremental();
            return;
        }
        BDD t1 = actual.restrict(Z.ithVar(0)); // IxV2
        if (USE_VCONTEXT) t1.andWith(V2cdomain.id()); // IxV2cxV2
        t1.replaceWith(V2toV1);
        BDD t3 = t1.relprod(mI, Mset); // IxV1cxV1 & MxIxN = IxV1cxV1xN
        t1.free();
        BDD t4;
        if (CS_CALLGRAPH) {
            // We keep track of where a call goes under different contexts.
            t4 = t3.relprod(vP, V1.set()); // IxV1cxV1xN x V1cxV1xH1cxH1 = V1cxIxH1cxH1xN
        } else {
            // By quantifying out V1c, we merge all contexts.
            t4 = t3.relprod(vP, V1set); // IxV1cxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
        }
// 9%
        BDD t5 = t4.relprod(hT, H1set); // (V1cx)IxH1cxH1xN x H1xT2 = (V1cx)IxT2xN
        t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // (V1cx)IxT2xN x T2xNxM = (V1cx)IxM
        t5.free();
        
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        if (CS_CALLGRAPH) IE.orWith(t6.exist(V1cset));
        else IE.orWith(t6.id());
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        if (CONTEXT_SENSITIVE || THREAD_SENSITIVE) {
            // Add the context for the new call graph edges.
            if (CS_CALLGRAPH) t6.replaceWith(V1ctoV2c); // V2cxIxM
            t6.andWith(IEfilter.id()); // V2cxIxV1cxM
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
        } else if (CARTESIAN_PRODUCT) {
            // t6 |= statics
            // for (i=0..k)
            //     t8 = actual x (z=i)
            //     t9 = t8 x vP
            //     context &= t9
            //     t9 &= V1cH1equals[i]
            //     tb = t9 x t6
            //     tc = tb x formal_i
            //     newPt |= tc
            // newPt2 = newPt x context
            // newPt2 &= vPfilter
            // vP |= newPt2
            // context &= t6
            // IEcs |= context
            
            // Add all statically-bound calls to t6.
            // They are true under all contexts.
            BDD statics = staticCalls.exist(V1.set()); // IxM
            if (CS_CALLGRAPH) statics.andWith(V1cdomain.id()); // V1cxIxM
            t6.orWith(statics); // V1cxIxM | V1cxIxM = V1cxIxM
            
            // Edges in the call graph.  Invocation I under context V2c has target method M.
            if (CS_CALLGRAPH) t6.replaceWith(V1ctoV2c); // V2cxIxM
            // The context for the new cg edges are based on the points-to set of every parameter.
            BDD context = bdd.one();
            // We need to add points-to relations for each of the actual parameters.
            BDD newPt = bdd.zero();
            for (int i = MAX_PARAMS - 1; i >= 0; --i) {
                if (TRACE_BIND) System.out.println("Param "+i+":");
                BDD t8 = actual.restrict(Z.ithVar(i)).and(V2cdomain); // IxV2
                t8.replaceWith(V2toV1); // IxV1
                if (TRACE_BIND) System.out.println("t8 = "+t8.toStringWithDomains());
                BDD t9 = t8.relprod(vP, V1.set()); // IxV1 x V1cxV1xH1cxH1 = V1cxIxH1cxH1
                if (TRACE_BIND) {
                    System.out.println("t9 =");
                    dumpWithV1c(t9, Iset.and(H1set));
                }
                t8.free();
                
                t9.replaceWith(V1ctoV2c); // V2cxIxH1cxH1
                BDD ta = t9.replace(H1toV1c[i]); // V2cxIxV1c[i]
                // Invocation I under context V2c leads to context V1c
// 20%
                context.andWith(ta); // V2cxIxV1c[i]
                
                // Calculate new points-to relations for this actual parameter.
                t9.andWith(V1cH1equals[i].id()); // V2cxIxV1c[i]xH1cxH1
                BDD tb = t9.relprod(t6, V2cset); // V2cxIxV1c[i]xH1cxH1 x (V2cx)IxM = IxV1c[i]xMxH1cxH1
                t9.free();
                BDD formal_i = formal.restrict(Z.ithVar(i)); // MxV1
                BDD tc = tb.relprod(formal_i, Mset); // IxV1c[i]xMxH1cxH1 x MxV1 = IxV1c[i]xV1xH1cxH1
                formal_i.free(); tb.free();
                for (int j = 0; j < MAX_PARAMS; ++j) {
                    if (i == j) continue;
                    tc.andWith(V1c[j].domain());
                }
                if (TRACE_BIND) dumpVP(tc.exist(Iset));
                newPt.orWith(tc);
            }
            
            // Now, filter out unrealizables.
            // IxV1c[i]xV1xH1cxH1 x V2cxIxV1c[i] = V1cxV1xH1cxH1
// 13%
            BDD newPt2 = newPt.relprod(context, V2cset.and(Iset));
            newPt.free();
            if (TRACE_BIND) dumpVP(newPt2);
            
            if (FILTER_VP) newPt2.andWith(vPfilter.id());
            vP.orWith(newPt2);
            
            context.andWith(t6.id()); // V2cxIxV1c[k]xM
            if (TRACE_BIND) System.out.println("context = "+context.toStringWithDomains());
            IEcs.orWith(context);
        }
        t3.free();
        t6.free();
    }
    
    BDD old3_t3;
    BDD old3_vP;
    BDD old3_t4;
    BDD old3_hT;
    BDD old3_t6;
    BDD old3_t9[];
    
    // t1 = actual x (z=0)
    // t3 = t1 x mI
    // new_t3 = t3 - old_t3
    // new_vP = vP - old_vP
    // t4 = t3 x new_vP
    // old_t3 = t3
    // t4 |= new_t3 x vP
    // new_t4 = t4 - old_t4
    // new_hT = hT - old_hT
    // t5 = t4 x new_hT
    // old_t4 = t4
    // t5 |= new_t4 x hT
    // t6 = t5 x cha
    // IE |= t6
    // old_vP = vP
    // old_hT = hT
    
    public void bindInvocations_incremental() {
        BDD t1 = actual.restrict(Z.ithVar(0)); // IxV2
        if (USE_VCONTEXT) t1.andWith(V2cdomain.id()); // IxV2cxV2
        t1.replaceWith(V2toV1); // IxV1cxV1
        BDD t3 = t1.relprod(mI, Mset); // IxV1cxV1 & MxIxN = IxV1cxV1xN
        t1.free();
        BDD new_t3 = t3.apply(old3_t3, BDDFactory.diff);
        old3_t3.free();
        if (false) out.println("New invokes: "+new_t3.toStringWithDomains());
        BDD new_vP = vP.apply(old3_vP, BDDFactory.diff);
        old3_vP.free();
        if (false) out.println("New vP: "+new_vP.toStringWithDomains());
        BDD t4, new_t4;
        if (CS_CALLGRAPH) {
            // We keep track of where a call goes under different contexts.
            t4 = t3.relprod(new_vP, V1.set()); // IxV1cxV1xN x V1cxV1xH1cxH1 = V1cxIxH1cxH1xN
            old3_t3 = t3;
            t4.orWith(new_t3.relprod(vP, V1.set())); // IxV1cxV1xN x V1cxV1xH1cxH1 = V1cxIxH1cxH1xN
            new_t3.free();
            new_t4 = t4.apply(old3_t4, BDDFactory.diff);
            old3_t4.free();
        } else {
            // By quantifying out V1c, we merge all contexts.
            t4 = t3.relprod(new_vP, V1set); // IxV1cxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
            old3_t3 = t3;
            t4.orWith(new_t3.relprod(vP, V1set)); // IxV1cxV1xN x V1cxV1xH1cxH1 = IxH1cxH1xN
            new_t3.free();
            new_t4 = t4.apply(old3_t4, BDDFactory.diff);
            old3_t4.free();
        }
        if (false) out.println("New 'this' objects: "+new_t4.toStringWithDomains());
        BDD new_hT = hT.apply(old3_hT, BDDFactory.diff);
        old3_hT.free();
        BDD t5 = t4.relprod(new_hT, H1set); // (V1cx)IxH1cxH1xN x H1xT2 = (V1cx)IxT2xN
        new_hT.free();
        old3_t4 = t4;
        t5.orWith(new_t4.relprod(hT, H1set)); // (V1cx)IxH1cxH1xN x H1xT2 = (V1cx)IxT2xN
        new_t4.free();
        BDD t6 = t5.relprod(cha, T2Nset); // (V1cx)IxT2xN x T2xNxM = (V1cx)IxM
        t5.free();
        
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        if (CS_CALLGRAPH) IE.orWith(t6.exist(V1cset));
        else IE.orWith(t6.id());
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        old3_vP = vP.id();
        old3_hT = hT.id();
        
        if (CONTEXT_SENSITIVE) {
            if (CS_CALLGRAPH) t6.replaceWith(V1ctoV2c); // V2cxIxM
            t6.andWith(IEfilter.id()); // V2cxIxV1cxM
            IEcs.orWith(t6);
        } else if (OBJECT_SENSITIVE) {
            throw new Error();
        } else if (CARTESIAN_PRODUCT) {
            // t6 |= statics
            // new_t6 = t6 - old_t6
            // for (i=0..k)
            //     t8[i] = actual x (z=i)
            //     t9[i] = t8[i] x vP
            //     new_t9[i] = t9[i] - old_t9[i]
            //     new_context &= new_t9[i]
            //     new_t9[i] &= V1cH1equals[i]
            //     tb[i] = new_t9[i] x t6
            //     tb[i] |= t9[i] x new_t6
            //     old_t9[i] = t9[i]
            //     tc[i] = tb[i] x formal_i
            //     newPt |= tc[i]
            // newPt2 = newPt x new_context
            // newPt2 &= vPfilter
            // vP |= newPt2
            // new_context &= t6
            // IEcs |= new_context
            
            // Add all statically-bound calls to t6.
            // They are true under all contexts.
            BDD statics = staticCalls.exist(V1.set()); // IxM
            if (CS_CALLGRAPH) statics.andWith(V1cdomain.id()); // V1cxIxM
            t6.orWith(statics); // V1cxIxM | V1cxIxM = V1cxIxM
            
            // Edges in the call graph.  Invocation I under context V2c has target method M.
            if (CS_CALLGRAPH) t6.replaceWith(V1ctoV2c); // V2cxIxM
            
            BDD new_t6 = t6.apply(old3_t6, BDDFactory.diff);
            
            // The context for the new cg edges are based on the points-to set of every parameter.
            BDD newContext = bdd.one();
            // We need to add points-to relations for each of the actual parameters.
            BDD newPt = bdd.zero();
            for (int i = MAX_PARAMS - 1; i >= 0; --i) {
                if (TRACE_BIND) System.out.println("Param "+i+":");
                BDD t8_i = actual.restrict(Z.ithVar(i)).and(V2cdomain); // IxV2
                t8_i.replaceWith(V2toV1); // IxV1
                if (TRACE_BIND) System.out.println("t8 = "+t8_i.toStringWithDomains());
                BDD t9_i = t8_i.relprod(vP, V1.set()); // IxV1 x V1cxV1xH1cxH1 = V1cxIxH1cxH1
                if (TRACE_BIND) {
                    System.out.println("t9 =");
                    dumpWithV1c(t9_i, Iset.and(H1set));
                }
                t8_i.free();
                
                t9_i.replaceWith(V1ctoV2c); // V2cxIxH1cxH1
                BDD new_t9_i = t9_i.apply(old3_t9[i], BDDFactory.diff);
                old3_t9[i] = t9_i.id();
                // Invocation I under context V2c leads to context V1c
// 20%
                newContext.andWith(new_t9_i.replace(H1toV1c[i])); // V2cxIxV1c[i]
                
                // Calculate new points-to relations for this actual parameter.
                new_t9_i.andWith(V1cH1equals[i].id()); // V2cxIxV1c[i]xH1cxH1
                t9_i.andWith(V1cH1equals[i].id());
                BDD tb_i = new_t9_i.relprod(t6, V2cset); // V2cxIxV1c[i]xH1cxH1 x (V2cx)IxM = IxV1c[i]xMxH1cxH1
                tb_i.orWith(t9_i.relprod(new_t6, V2cset));
                BDD formal_i = formal.restrict(Z.ithVar(i)); // MxV1
                BDD tc_i = tb_i.relprod(formal_i, Mset); // IxV1c[i]xMxH1cxH1 x MxV1 = IxV1c[i]xV1xH1cxH1
                formal_i.free(); tb_i.free();
                for (int j = 0; j < MAX_PARAMS; ++j) {
                    if (i == j) continue;
                    tc_i.andWith(V1c[j].domain());
                }
                if (TRACE_BIND) dumpVP(tc_i.exist(Iset));
                newPt.orWith(tc_i);
            }
            
            // Now, filter out unrealizables.
            // IxV1c[i]xV1xH1cxH1 x V2cxIxV1c[i] = V1cxV1xH1cxH1
// 13%
            BDD newPt2 = newPt.relprod(newContext, V2cset.and(Iset));
            newPt.free();
            if (TRACE_BIND) dumpVP(newPt2);
            
            if (FILTER_VP) newPt2.andWith(vPfilter.id());
            vP.orWith(newPt2);
            
            newContext.andWith(t6.id()); // V2cxIxV1c[k]xM
            old3_t6 = t6;
            if (TRACE_BIND) System.out.println("context = "+newContext.toStringWithDomains());
            IEcs.orWith(newContext);
        }
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
        
        BDD my_IE = USE_VCONTEXT ? IEcs : IE;
        
        if (TRACE_SOLVER) out.println("Call graph edges: "+my_IE.nodeCount());
        
        BDD my_formal = CARTESIAN_PRODUCT ? formal.and(Z.varRange(0, MAX_PARAMS-1).not()) : formal;
        BDD my_actual = CARTESIAN_PRODUCT ? actual.and(Z.varRange(0, MAX_PARAMS-1).not()) : actual;
        
        BDD t1 = my_IE.relprod(my_actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(my_formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("A before param bind: "+A.nodeCount());
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("A after param bind: "+A.nodeCount());
        
        if (TRACE_SOLVER) out.println("Binding return values...");
        BDD my_IEr = USE_VCONTEXT ? IEcs.replace(V1cV2ctoV2cV1c) : IE;
        BDD t3 = my_IEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("A before return bind: "+A.nodeCount());
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("A after return bind: "+A.nodeCount());
        
        if (TRACE_SOLVER) out.println("Binding exceptions...");
        BDD t5 = my_IEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (USE_VCONTEXT) my_IEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("A before exception bind: "+A.nodeCount());
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("A after exception bind: "+A.nodeCount());
        
    }
    
    BDD old2_myIE;
    BDD old2_visited;
    
    public void bindParameters_incremental() {

        BDD my_IE = USE_VCONTEXT ? IEcs : IE;
        BDD new_myIE = my_IE.apply(old2_myIE, BDDFactory.diff);
        
        BDD new_visited = visited.apply(old2_visited, BDDFactory.diff);
        // add in any old edges targetting newly-visited methods, because the
        // argument/retval binding doesn't occur until the method has been visited.
        new_myIE.orWith(old2_myIE.and(new_visited));
        old2_myIE.free();
        old2_visited.free();
        new_visited.free();
        
        if (TRACE_SOLVER) out.println("New call graph edges: "+new_myIE.nodeCount());
        
        BDD my_formal = CARTESIAN_PRODUCT ? formal.and(Z.varRange(0, MAX_PARAMS-1).not()) : formal;
        BDD my_actual = CARTESIAN_PRODUCT ? actual.and(Z.varRange(0, MAX_PARAMS-1).not()) : actual;
        
        if (TRACE_SOLVER) out.println("Binding parameters...");
        
        BDD t1 = new_myIE.relprod(my_actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(my_formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("A before param bind: "+A.nodeCount());
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("A after param bind: "+A.nodeCount());
        
        if (TRACE_SOLVER) out.println("Binding return values...");
        BDD new_myIEr = USE_VCONTEXT ? new_myIE.replace(V1cV2ctoV2cV1c) : new_myIE;
        BDD t3 = new_myIEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("A before return bind: "+A.nodeCount());
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("A after return bind: "+A.nodeCount());
        
        if (TRACE_SOLVER) out.println("Binding exceptions...");
        BDD t5 = new_myIEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        if (USE_VCONTEXT) new_myIEr.free();
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("A before exception bind: "+A.nodeCount());
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("A after exception bind: "+A.nodeCount());
        
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
            oCnumbering = new SCCPathNumbering(objectPathSelector);
            BigInteger paths = (BigInteger) oCnumbering.countPaths(ocg);
            if (updateBits) {
                HC_BITS = VC_BITS = paths.bitLength();
                System.out.print("Object paths="+paths+" ("+VC_BITS+" bits), ");
            }
        }
        if ((CONTEXT_SENSITIVE && MAX_HC_BITS > 1) || THREAD_SENSITIVE) {
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
        // First, print something because the set of objects reachable via System.out changes
        // depending on whether something has been printed or not!
        System.out.println("Adding default static variables.");
        
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
        
        if (CARTESIAN_PRODUCT) {
            VC_BITS = (HC_BITS + H_BITS) * MAX_PARAMS;
            System.out.println("Variable context bits = ("+HC_BITS+"+"+H_BITS+")*"+MAX_PARAMS+"="+VC_BITS);
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
                if (USE_VCONTEXT) {
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
        
        if (DUMP_INITIAL) {
            buildTypes();
            dumpBDDRelations();
            return;
        }
        
        // Start timing solver.
        time = System.currentTimeMillis();
        
        if (DISCOVER_CALL_GRAPH || OBJECT_SENSITIVE || CARTESIAN_PRODUCT) {
            iterate();
        } else {
            assumeKnownCallGraph();
        }
        
        System.out.println("Time spent solving: "+(System.currentTimeMillis()-time)/1000.);

        printSizes();
        
        System.out.println("Writing call graph...");
        time = System.currentTimeMillis();
        dumpCallGraph();
        System.out.println("Time spent writing: "+(System.currentTimeMillis()-time)/1000.);
        
        if (DUMP_RESULTS) {
            System.out.println("Writing results...");
            time = System.currentTimeMillis();
            dumpResults(resultsFileName);
            System.out.println("Time spent writing: "+(System.currentTimeMillis()-time)/1000.);
        }
    }
   
    static Collection readClassesFromFile(String fname) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader(fname));
        Collection rootMethods = new ArrayList();
        String s = null;
        while ((s = r.readLine()) != null) {
            jq_Class c = (jq_Class) jq_Type.parseType(s);
            c.prepare();
            rootMethods.addAll(Arrays.asList(c.getDeclaredStaticMethods()));
        }
        return rootMethods;
    }

    public static void main(String[] args) throws IOException {
        if (USE_JOEQ_CLASSLIBS) {
            System.setProperty("joeq.classlibinterface", "joeq.ClassLib.pa.Interface");
            joeq.ClassLib.ClassLibInterface.useJoeqClasslib(true);
        }
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        Collection rootMethods = null;
        if (args[0].startsWith("@")) {
            rootMethods = readClassesFromFile(args[0].substring(1));
        } else {
            jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
            c.prepare();
        
            rootMethods = Arrays.asList(c.getDeclaredStaticMethods());
        }

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
                if (dis.CONTEXT_SENSITIVE || dis.OBJECT_SENSITIVE || dis.THREAD_SENSITIVE) {
                    System.out.println("Discovering call graph first...");
                    dis.CONTEXT_SENSITIVE = false;
                    dis.OBJECT_SENSITIVE = false;
                    dis.THREAD_SENSITIVE = false;
                    dis.DISCOVER_CALL_GRAPH = true;
                    dis.CS_CALLGRAPH = false;
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

        if (WRITE_PARESULTS_BATCHFILE)
            writePAResultsBatchFile("runparesults");
    }

    /**
     * write a file that when executed by shell runs PAResults in proper environment.
     */
    static void writePAResultsBatchFile(String batchfilename) throws IOException {
        PrintWriter w = new PrintWriter(new FileOutputStream(batchfilename));
        Properties p = System.getProperties();
        w.print(p.getProperty("java.home") + File.separatorChar + "bin" + File.separatorChar + "java");
        w.print(" -Xmx512M");
        w.print(" -classpath \"" + System.getProperty("java.class.path")+"\"");
        w.print(" -Djava.library.path=\"" + System.getProperty("java.library.path")+"\"");
        for (Iterator i = p.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            String key = (String)e.getKey();
            String val = (String)e.getValue();
            if (key.startsWith("ms.") || key.startsWith("pa.")) {
                w.print(" -D" + key + "=" + val);
            }
        }
        w.println(" joeq.Compiler.Analysis.IPA.PAResults");
        w.close();
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
    
    // XXX should we use an interface here for long location printing?
    public String longForm(Object o) {
	if (o == null || !LONG_LOCATIONS)
	    return "";

	// Node is a ProgramLocation
	if (o instanceof ProgramLocation) {
	    return " in "+((ProgramLocation)o).toStringLong();
	} else {
	    try {
		Class c = o.getClass();
		try {
		    // Node has getLocation() 
		    Method m = c.getMethod("getLocation", new Class[] {});
		    ProgramLocation pl = (ProgramLocation)m.invoke(o, null);
		    if (pl == null)
			throw new NoSuchMethodException();
		    return " in "+pl.toStringLong();
		} catch (NoSuchMethodException _1) {
		    try {
			// Node has at least a getMethod() 
			Method m = c.getMethod("getMethod", new Class[] {});
			return " " + m.invoke(o, null);
		    } catch (NoSuchMethodException _2) {
			try {
			    // or getDefiningMethod() 
			    Method m = c.getMethod("getDefiningMethod", new Class[] {});
			    return " " + m.invoke(o, null);
			} catch (NoSuchMethodException _3) {
			}
		    }
		}
	    } catch (InvocationTargetException _) {
	    } catch (IllegalAccessException _) { 
	    }
	}
	return "";
    }

    String findInMap(IndexedMap map, int j) {
        String jp = "("+j+")";
        if (j < map.size() && j >= 0) {
            Object o = map.get(j);
	    jp += o + longForm(o);
	    return jp;
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
                default: return "("+j+")"+"??";
            }
        }
        public String elementNames(int i, long j, long k) {
            // TODO: don't bother printing out long form of big sets.
            return super.elementNames(i, j, k);
        }
    }
   
    private void dumpCallGraphAsDot(CallGraph cg, String dotFileName) throws IOException {
	DataOutputStream dos = new DataOutputStream(new FileOutputStream(dotFileName));
	countCallGraph(cg, null, false).dotGraph(dos, cg.getRoots(), cg.getCallSiteNavigator());
	dos.close();
    }

    public void dumpCallGraph() throws IOException {
        //CallGraph callgraph = CallGraph.makeCallGraph(roots, new PACallTargetMap());
        CallGraph callgraph = new CachedCallGraph(new PACallGraph(this));
        //CallGraph callgraph = callGraph;
        DataOutputStream dos;
        dos = new DataOutputStream(new FileOutputStream(callgraphFileName));
        LoadedCallGraph.write(callgraph, dos);
        dos.close();
        
        if (DUMP_DOTGRAPH)
            dumpCallGraphAsDot(callgraph, callgraphFileName + ".dot");
        
    }
    
    public void dumpResults(String dumpfilename) throws IOException {

        System.out.println("A: "+(long) A.satCount(V1V2set)+" relations, "+A.nodeCount()+" nodes");
        bdd.save(dumpfilename+".A", A);
        System.out.println("vP: "+(long) vP.satCount(V1H1set)+" relations, "+vP.nodeCount()+" nodes");
        bdd.save(dumpfilename+".vP", vP);
        //BuildBDDIR.dumpTuples(bdd, dumpfilename+".vP.tuples", vP);
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
        System.out.println("formal: "+(long) formal.satCount(MZset.and(V1.set()))+" relations, "+formal.nodeCount()+" nodes");
        bdd.save(dumpfilename+".formal", formal);
        System.out.println("Iret: "+(long) Iret.satCount(Iset.and(V1.set()))+" relations, "+Iret.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Iret", Iret);
        System.out.println("Mret: "+(long) Mret.satCount(Mset.and(V2.set()))+" relations, "+Mret.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Mret", Mret);
        System.out.println("Ithr: "+(long) Ithr.satCount(Iset.and(V1.set()))+" relations, "+Ithr.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Ithr", Ithr);
        System.out.println("Mthr: "+(long) Mthr.satCount(Mset.and(V2.set()))+" relations, "+Mthr.nodeCount()+" nodes");
        bdd.save(dumpfilename+".Mthr", Mthr);
        System.out.println("mI: "+(long) mI.satCount(INset.and(Mset))+" relations, "+mI.nodeCount()+" nodes");
        bdd.save(dumpfilename+".mI", mI);
        System.out.println("mV: "+(long) mV.satCount(Mset.and(V1.set()))+" relations, "+mV.nodeCount()+" nodes");
        bdd.save(dumpfilename+".mV", mV);
        System.out.println("sync: "+(long) sync.satCount(V1.set())+" relations, "+sync.nodeCount()+" nodes");
        bdd.save(dumpfilename+".sync", sync);
        
        System.out.println("hP: "+(long) hP.satCount(H1FH2set)+" relations, "+hP.nodeCount()+" nodes");
        bdd.save(dumpfilename+".hP", hP);
        //BuildBDDIR.dumpTuples(bdd, dumpfilename+".hP.tuples", hP);
        System.out.println("IE: "+(long) IE.satCount(IMset)+" relations, "+IE.nodeCount()+" nodes");
        bdd.save(dumpfilename+".IE", IE);
        BuildBDDIR.dumpTuples(bdd, dumpfilename+".IE.tuples", IE);
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
        if (NNfilter != null) {
            System.out.println("NNfilter: "+NNfilter.nodeCount()+" nodes");
            bdd.save(dumpfilename+".NNfilter", NNfilter);
        }
        System.out.println("visited: "+(long) visited.satCount(Mset)+" relations, "+visited.nodeCount()+" nodes");
        bdd.save(dumpfilename+".visited", visited);
        
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
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
        if (pa.A instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.V1); set.add(pa.V2);
            set.addAll(Arrays.asList(pa.V1c)); set.addAll(Arrays.asList(pa.V2c));
            ((TypedBDD) pa.A).setDomains(set);
        }
        if (pa.vP instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.V1); set.add(pa.H1);
            set.addAll(Arrays.asList(pa.V1c)); set.addAll(Arrays.asList(pa.H1c));
            ((TypedBDD) pa.vP).setDomains(set);
        }
        if (pa.S instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.V1); set.add(pa.V2); set.add(pa.F);
            set.addAll(Arrays.asList(pa.V1c)); set.addAll(Arrays.asList(pa.V2c));
            ((TypedBDD) pa.S).setDomains(set);
        }
        if (pa.L instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.V1); set.add(pa.V2); set.add(pa.F);
            set.addAll(Arrays.asList(pa.V1c)); set.addAll(Arrays.asList(pa.V2c));
            ((TypedBDD) pa.L).setDomains(set);
        }
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

        if (pa.hP instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.H1); set.add(pa.H2); set.add(pa.F);
            set.addAll(Arrays.asList(pa.H1c)); set.addAll(Arrays.asList(pa.H2c));
            ((TypedBDD) pa.hP).setDomains(set);
        }
        if (pa.IE instanceof TypedBDD)
            ((TypedBDD) pa.IE).setDomains(pa.I, pa.M);
        if (pa.IEcs instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.I); set.add(pa.M);
            set.addAll(Arrays.asList(pa.V2c)); set.addAll(Arrays.asList(pa.V1c));
            ((TypedBDD) pa.IEcs).setDomains(set);
        }
        if (pa.vPfilter instanceof TypedBDD)
            ((TypedBDD) pa.vPfilter).setDomains(pa.V1, pa.H1);
        if (pa.hPfilter instanceof TypedBDD)
            ((TypedBDD) pa.hPfilter).setDomains(pa.H1, pa.F, pa.H2);
        if (pa.IEfilter instanceof TypedBDD) {
            Set set = TypedBDDFactory.makeSet();
            set.add(pa.I); set.add(pa.M);
            set.addAll(Arrays.asList(pa.V2c)); set.addAll(Arrays.asList(pa.V1c));
            ((TypedBDD) pa.IEfilter).setDomains(set);
        }
        if (pa.NNfilter instanceof TypedBDD)
            ((TypedBDD) pa.NNfilter).setDomains(pa.H1);
        
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
        out.writeBytes("TS="+(THREAD_SENSITIVE?"yes":"no")+"\n");
        out.writeBytes("CP="+(CARTESIAN_PRODUCT?"yes":"no")+"\n");
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
            } else if (s1.equals("TS")) {
                THREAD_SENSITIVE = s2.equals("yes");
            } else if (s1.equals("CP")) {
                CARTESIAN_PRODUCT = s2.equals("yes");
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
            HashMap m = new HashMap();
            for (Iterator i = map.keySet().iterator(); i.hasNext(); ) {
                Object o = i.next();
                m.put(o, get(o));
            }
            return m.entrySet();
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
        System.out.println("Vars="+vars+" Heaps="+heaps+" Classes="+classes.size()+" Fields="+fields.size());
        PathNumbering pn = null;
        if (THREAD_SENSITIVE)
            pn = new RootPathNumbering();
        else if (CONTEXT_SENSITIVE)
            pn = new SCCPathNumbering(varPathSelector);
        else
            pn = null;
        if (updateBits) {
            V_BITS = BigInteger.valueOf(vars+256).bitLength();
            I_BITS = BigInteger.valueOf(calls).bitLength();
            H_BITS = BigInteger.valueOf(heaps+256).bitLength();
            F_BITS = BigInteger.valueOf(fields.size()+64).bitLength();
            T_BITS = BigInteger.valueOf(classes.size()+64).bitLength();
            N_BITS = I_BITS;
            M_BITS = BigInteger.valueOf(methods).bitLength() + 1;
            if (CONTEXT_SENSITIVE || THREAD_SENSITIVE) {
                System.out.println("Thread runs="+thread_runs);
                Map initialCounts = new ThreadRootMap(thread_runs);
                BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
                VC_BITS = paths.bitLength();
                if (VC_BITS > MAX_VC_BITS)
                    System.out.println("Trimming var context bits from "+VC_BITS);
                VC_BITS = Math.min(MAX_VC_BITS, VC_BITS);
                System.out.println("Paths="+paths+", Var context bits="+VC_BITS);
            }
            System.out.println(" V="+V_BITS+" I="+I_BITS+" H="+H_BITS+
                               " F="+F_BITS+" T="+T_BITS+" N="+N_BITS+
                               " M="+M_BITS+" VC="+VC_BITS);
        }
        return pn;
    }

    public final VarPathSelector varPathSelector = new VarPathSelector(MAX_VC_BITS);
    
    public static boolean THREADS_ONLY = false;
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
            if (THREADS_ONLY) {
                Set s = scc2.nodeSet();
                Iterator i = s.iterator();
                Object o = i.next();
                if (i.hasNext()) return false;
                if (o instanceof ProgramLocation) return true;
                jq_Method m = (jq_Method) o;
                if (m.getNameAndDesc() == main_method) return true;
                if (m.getNameAndDesc() == run_method) return true;
                return false;
            }
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
    
    public static boolean MATCH_FACTORY = false;
    
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
            if (m.getNameAndDesc() == main_method) return true;
            if (m.getNameAndDesc() == run_method) return true;
            if (m.getBytecode() == null) return false;
            if (MATCH_FACTORY) {
                if (!m.getReturnType().isReferenceType()) return false;
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
        PathNumbering pn;
	if (THREAD_SENSITIVE) pn = new RootPathNumbering();
        else pn = new SCCPathNumbering(heapPathSelector);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        System.out.println("Number of paths for heap context sensitivity: "+paths);
        if (updateBits) {
            HC_BITS = paths.bitLength();
            if (HC_BITS > MAX_HC_BITS)
                System.out.println("Trimming heap context bits from "+HC_BITS);
            HC_BITS = Math.min(HC_BITS, MAX_HC_BITS);
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
                    context = buildContextMap(V2c[0],
                                              PathNumbering.toBigInt(r_caller.low),
                                              PathNumbering.toBigInt(r_caller.high),
                                              V1c[0],
                                              PathNumbering.toBigInt(r_edge.low),
                                              PathNumbering.toBigInt(r_edge.high));
                } else {
                    if (USE_VCONTEXT) context = V1cdomain.and(V2cdomain);
                    else context = bdd.one();
                }
                context.andWith(I.ithVar(I_i));
                context.andWith(M.ithVar(M_i));
                IEfilter.orWith(context);
            }
        }
    }
    
    public BDD getV1H1Context(jq_Method m) {
        if (CONTEXT_SENSITIVE || THREAD_SENSITIVE) {
            if (V1H1correspondence != null)
                return (BDD) V1H1correspondence.get(m);
            Range r1 = vCnumbering.getRange(m);
            BDD b = V1c[0].varRange(r1.low.longValue(), r1.high.longValue());
            if (USE_HCONTEXT)
                b.andWith(H1c[0].ithVar(0));
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
        } else if (CARTESIAN_PRODUCT) {
            // todo! heap context sensitivity for cartesian product.
            BDD context;
            if (USE_HCONTEXT) context = V1cdomain.and(H1cdomain);
            else context = V1cdomain.id();
            return context;
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
                cm = buildContextMap(V1c[0],
                                     PathNumbering.toBigInt(r1.low),
                                     PathNumbering.toBigInt(r1.high),
                                     H1c[0],
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
                cm = buildContextMap(V1c[0],
                                     PathNumbering.toBigInt(r1.low),
                                     PathNumbering.toBigInt(r1.high),
                                     H1c[0],
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
        V2cH2ctoV1cH1c.set(V2c, V1c);
        V2cH2ctoV1cH1c.set(H2c, H1c);
        
        V1H1correspondence = new HashMap();
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            Range r1 = vCnumbering.getRange(m);
            Range r2 = hCnumbering.getRange(m);
            BDD relation;
            if (r1.equals(r2)) {
                relation = V1c[0].buildAdd(H1c[0], BigInteger.valueOf(r1.high.longValue()).bitLength(), 0);
                relation.andWith(V1c[0].varRange(r1.low.longValue(), r1.high.longValue()));
            } else {
                long v_val = r1.high.longValue()+1;
                long h_val = r2.high.longValue()+1;
                
                if (h_val == 1L) {
                    relation = V1c[0].varRange(r1.low.longValue(), r1.high.longValue());
                    relation.andWith(H1c[0].ithVar(0));
                } else {
                    int v_bits = BigInteger.valueOf(v_val).bitLength();
                    int h_bits = BigInteger.valueOf(h_val).bitLength();
                    // make it faster.
                    h_val = 1 << h_bits;
                    
                    int[] v = new int[v_bits];
                    for (int j = 0; j < v_bits; ++j) {
                        v[j] = V1c[0].vars()[j];
                    }
                    BDDBitVector v_vec = bdd.buildVector(v);
                    BDDBitVector z = v_vec.divmod(h_val, false);
                    
                    //int h_bits = BigInteger.valueOf(h_val).bitLength();
                    //int[] h = new int[h_bits];
                    //for (int j = 0; j < h_bits; ++j) {
                    //    h[j] = H1c.vars()[j];
                    //}
                    //BDDBitVector h_vec = bdd.buildVector(h);
                    BDDBitVector h_vec = bdd.buildVector(H1c[0]);
                    
                    relation = bdd.one();
                    int n;
                    for (n = 0; n < h_vec.size() || n < v_vec.size(); n++) {
                        BDD a = (n < v_vec.size()) ? z.getBit(n) : bdd.zero();
                        BDD b = (n < h_vec.size()) ? h_vec.getBit(n) : bdd.zero();
                        relation.andWith(a.biimp(b));
                    }
                    for ( ; n < V1c[0].varNum() || n < H1c[0].varNum(); n++) {
                        if (n < V1c[0].varNum())
                            relation.andWith(bdd.nithVar(V1c[0].vars()[n]));
                        if (n < H1c[0].varNum())
                            relation.andWith(bdd.nithVar(H1c[0].vars()[n]));
                    }
                    relation.andWith(V1c[0].varRange(r1.low.longValue(), r1.high.longValue()));
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
        V2cH2ctoV1cH1c.set(V2c, V1c);
        V2cH2ctoV1cH1c.set(H2c, H1c);
        BDDPairing V2ctoV1c = bdd.makePair();
        V2ctoV1c.set(V2c, V1c);
        BDDPairing H2ctoH1c = bdd.makePair();
        H2ctoH1c.set(H2c, H1c);
        
        V1H1correspondence = new HashMap();
        for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
            jq_Method root = (jq_Method) i.next();
            Range r1 = vCnumbering.getRange(root);
            Range r2 = hCnumbering.getRange(root);
            BDD relation;
            if (r1.equals(r2)) {
                relation = V1c[0].buildAdd(H1c[0], BigInteger.valueOf(r1.high.longValue()).bitLength(), 0);
                relation.andWith(V1c[0].varRange(r1.low.longValue(), r1.high.longValue()));
                System.out.println("Root "+root+" numbering: "+relation.toStringWithDomains());
            } else {
                System.out.println("Root numbering doesn't match: "+root);
                // just intermix them all, because we don't know the mapping.
                relation = V1c[0].varRange(r1.low.longValue(), r1.high.longValue());
                relation.andWith(H1c[0].varRange(r2.low.longValue(), r2.high.longValue()));
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
                    cm1 = buildContextMap(V1c[0],
                                          PathNumbering.toBigInt(r1_caller.low),
                                          PathNumbering.toBigInt(r1_caller.high),
                                          V2c[0],
                                          PathNumbering.toBigInt(r1_edge.low),
                                          PathNumbering.toBigInt(r1_edge.high));
                    tmpRel = callerRelation.relprod(cm1, V1cset);
                    cm1.free();
                } else {
                    tmpRel = callerRelation.id();
                }
                BDD tmpRel2;
                if (!r2_same) {
                    cm1 = buildContextMap(H1c[0],
                                          PathNumbering.toBigInt(r2_caller.low),
                                          PathNumbering.toBigInt(r2_caller.high),
                                          H2c[0],
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
    
    public PAResults getResults() {
        PAResults r = new PAResults(this);
        r.cg = this.cg;
        return r;
    }
    
    public void dumpBDDRelations() throws IOException {
        
        // difference in compatibility
        BDD S0 = S.exist(V1cV2cset);
        BDD L0 = L.exist(V1cV2cset);
        BDD IE0 = IE.exist(V1cV2cset);
        BDD vP0 = vP.exist(V1cH1cset);
        
        String dumpPath = "";
        bdd.save(dumpPath+"vP0.bdd", vP0);
        bdd.save(dumpPath+"hP0.bdd", hP);
        bdd.save(dumpPath+"L.bdd", L0);
        bdd.save(dumpPath+"S.bdd", S0);
        bdd.save(dumpPath+"A.bdd", A);
        bdd.save(dumpPath+"vT.bdd", vT);
        bdd.save(dumpPath+"hT.bdd", hT);
        bdd.save(dumpPath+"aT.bdd", aT);
        bdd.save(dumpPath+"cha.bdd", cha);
        bdd.save(dumpPath+"actual.bdd", actual);
        bdd.save(dumpPath+"formal.bdd", formal);
        bdd.save(dumpPath+"mI.bdd", mI);
        bdd.save(dumpPath+"Mret.bdd", Mret);
        bdd.save(dumpPath+"Mthr.bdd", Mthr);
        bdd.save(dumpPath+"Iret.bdd", Iret);
        bdd.save(dumpPath+"Ithr.bdd", Ithr);
        bdd.save(dumpPath+"IE0.bdd", IE0);
        if (IEfilter != null) bdd.save(dumpPath+"IEfilter.bdd", IEfilter);
        
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(dumpPath+"bddinfo"));
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            BDDDomain d = bdd.getDomain(i);
            if (d == V1 || d == V2) dos.writeBytes("V\n");
            else if (d == H1 || d == H2) dos.writeBytes("H\n");
            else if (d == T1 || d == T2) dos.writeBytes("T\n");
            else if (d == F) dos.writeBytes("F\n");
            else if (d == I) dos.writeBytes("I\n");
            else if (d == Z) dos.writeBytes("Z\n");
            else if (d == N) dos.writeBytes("N\n");
            else if (d == M) dos.writeBytes("M\n");
            else if (Arrays.asList(V1c).contains(d) || Arrays.asList(V2c).contains(d)) dos.writeBytes("VC\n");
            else if (Arrays.asList(H1c).contains(d) || Arrays.asList(H2c).contains(d)) dos.writeBytes("HC\n");
            else dos.writeBytes(d.toString()+"\n");
        }
        dos.close();
        
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"fielddomains.pa"));
        dos.writeBytes("V "+(1L<<V_BITS)+" var.map\n");
        dos.writeBytes("H "+(1L<<H_BITS)+" heap.map\n");
        dos.writeBytes("T "+(1L<<T_BITS)+" type.map\n");
        dos.writeBytes("F "+(1L<<F_BITS)+" field.map\n");
        dos.writeBytes("I "+(1L<<I_BITS)+" invoke.map\n");
        dos.writeBytes("Z "+(1L<<Z_BITS)+"\n");
        dos.writeBytes("N "+(1L<<N_BITS)+" name.map\n");
        dos.writeBytes("M "+(1L<<M_BITS)+" method.map\n");
        dos.writeBytes("VC "+(1L<<VC_BITS)+"\n");
        dos.writeBytes("HC "+(1L<<HC_BITS)+"\n");
        dos.close();
        
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"var.map"));
        Vmap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"heap.map"));
        Hmap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"type.map"));
        Tmap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"field.map"));
        Fmap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"invoke.map"));
        Imap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"name.map"));
        Nmap.dumpStrings(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpPath+"method.map"));
        Mmap.dumpStrings(dos);
        dos.close();
    }
    
    
}

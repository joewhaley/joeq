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
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Member;
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
import Util.Graphs.PathNumbering;
import Util.Graphs.SCComponent;
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
    boolean FILTER_TYPE = true;
    boolean INCREMENTAL1 = true;
    boolean INCREMENTAL2 = true;
    boolean INCREMENTAL3 = true;
    boolean CONTEXT_SENSITIVE = false;
    
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "2500000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "150000"));
    static String resultsFileName = System.getProperty("bddresults", "pa");
    static String callgraphfilename = System.getProperty("callgraph", "callgraph");
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    BDDDomain V1c, V2c, H1c, H2c;
    
    int V_BITS=16, I_BITS=15, H_BITS=15, Z_BITS=5, F_BITS=12, T_BITS=12, N_BITS=11, M_BITS=14;
    int VC_BITS=1, HC_BITS=1;
    
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
    BDD actual; // IxZxV2, actual parameters            (+context)
    BDD formal; // MxZxV1, formal parameters            (+context)
    BDD Iret;   // IxV1, invocation return value        (+context)
    BDD Mret;   // MxV2, method return value            (+context)
    BDD Ithr;   // IxV1, invocation thrown value        (+context)
    BDD Mthr;   // MxV2, method thrown value            (+context)
    BDD mI;     // MxIxN, method invocations            (+context)
    BDD mV;     // MxV, method variables                (+context)
    
    BDD hP;     // H1xFxH2, heap points-to              (+context)
    BDD IE;     // IxM, invocation edges                (+context)
    BDD filter; // V1xH1, type filter                   (+context)
    
    BDD visited; // M, visited methods
    
    String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2c_H2_T2_H1c_H1");
    //String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_H2_T2_H1");
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDDPairing V1cV2ctoV2cV1c;
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1H1set, IMset, H1Fset, H1FH2set, T2Nset, MZset;
    
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
        
        // IxH1xN x H1xT2
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
            V2H2toV1H1.set(new BDDDomain[] {V1c,V2c},
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
        if (TRACE_RELATIONS) out.println("Adding to visited: (M:"+M_bdd.scanVar(M)+")");
        visited.orWith(M_bdd.id());
    }
    
    void addToFormal(BDD M_bdd, int z, Node v) {
        BDD bdd1 = Z.ithVar(z);
        int V_i = Vmap.get(v);
        bdd1.andWith(V1.ithVar(V_i));
        bdd1.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to formal: (M:"+M_bdd.scanVar(M)+",Z:"+z+",V1:"+V_i+")");
        formal.orWith(bdd1);
    }
    
    void addToIE(BDD V1V2context, BDD I_bdd, jq_Method target) {
        int M2_i = Mmap.get(target);
        BDD bdd1 = M.ithVar(M2_i);
        bdd1.andWith(I_bdd.id());
        if (CONTEXT_SENSITIVE) bdd1.andWith(V1V2context.id());
        if (TRACE_RELATIONS) out.println("Adding to IE: (I:"+I_bdd.scanVar(I)+",M:"+M2_i+")");
        IE.orWith(bdd1);
    }
    
    void addToMI(BDD M_bdd, BDD I_bdd, jq_Method target) {
        int N_i = Nmap.get(target);
        BDD bdd1 = N.ithVar(N_i);
        bdd1.andWith(M_bdd.id());
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to mI: (M:"+M_bdd.scanVar(M)+",I:"+I_bdd.scanVar(I)+",N:"+N_i+")");
        mI.orWith(bdd1);
    }
    
    void addToActual(BDD I_bdd, int z, Set s) {
        BDD bdd1 = bdd.zero();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            int V_i = Vmap.get(j.next());
            if (TRACE_RELATIONS) out.println("Adding to actual: (I:"+I_bdd.scanVar(I)+",Z:"+z+",V2:"+V_i+")");
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
        if (TRACE_RELATIONS) out.println("Adding to Iret: (I:"+I_bdd.scanVar(I)+",V1:"+V_i+")");
        Iret.orWith(bdd1);
    }
    
    void addToIthr(BDD I_bdd, Node v) {
        int V_i = Vmap.get(v);
        BDD bdd1 = V1.ithVar(V_i);
        bdd1.andWith(I_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Ithr: (I:"+I_bdd.scanVar(I)+",V1:"+V_i+")");
        Ithr.orWith(bdd1);
    }
    
    void addToMV(BDD M_bdd, BDD V_bdd) {
        BDD bdd = M_bdd.id();
        bdd.andWith(V_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to mV: (M:"+M_bdd.scanVar(M)+",V1:"+V_bdd.scanVar(V1)+")");
        mV.orWith(bdd);
    }
    
    void addToMret(BDD M_bdd, Node v) {
        addToMret(M_bdd, Vmap.get(v));
    }
    
    void addToMret(BDD M_bdd, int V_i) {
        BDD bdd = V2.ithVar(V_i);
        bdd.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Mret: (M:"+M_bdd.scanVar(M)+",V2:"+V_i+")");
        Mret.orWith(bdd);
    }
    
    void addToMthr(BDD M_bdd, int V_i) {
        BDD bdd = V2.ithVar(V_i);
        bdd.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to Mthr: (M:"+M_bdd.scanVar(M)+",V2:"+V_i+")");
        Mthr.orWith(bdd);
    }
    
    void addToVP(Node p, int H_i) {
        int V1_i = Vmap.get(p);
        BDD bdd1 = V1.ithVar(V1_i);
        bdd1.andWith(H1.ithVar(H_i));
        vP.orWith(bdd1);
    }
    
    void addToVP(BDD V_bdd, Node h) {
        int H_i = Hmap.get(h);
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(V_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: (V1:"+V_bdd.scanVar(V1)+",H1:"+H_i+")");
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
        if (TRACE_RELATIONS) out.println("Adding to A: (V1:"+V_bdd.scanVar(V1)+",V2:"+V2_i+")");
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
            if (TRACE_RELATIONS) out.println("Adding to S: (V1:"+V_bdd.scanVar(V1)+",F:"+F_i+",V2:"+V2_i+")");
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
            if (TRACE_RELATIONS) out.println("Adding to L: (V1:"+V_bdd.scanVar(V1)+",F:"+F_i+",V2:"+V2_i+")");
            L.orWith(bdd1);
        }
        F_bdd.free();
    }
    
    public void visitMethod(jq_Method m) {
        if (visitedMethods.contains(m)) return;
        visitedMethods.add(m);
        
        if (TRACE) out.println("Visiting method "+m);
        m.getDeclaringClass().prepare();
        
        int M_i = Mmap.get(m);
        BDD M_bdd = M.ithVar(M_i);
        addToVisited(M_bdd);
        
        BDD V1V2context = null;
        if (CONTEXT_SENSITIVE) {
            Number n1 = vCnumbering.numberOfPathsTo(m);
            int bits = BigInteger.valueOf(n1.longValue()).bitLength();
            V1V2context = V1c.buildAdd(V2c, bits, 0L);
            V1V2context.andWith(V1c.varRange(0, n1.longValue()-1));
        }
        
        if (m.getBytecode() == null) {
            // todo: parameters passed into native methods.
            // build up 'Mret'
            jq_Type retType = m.getReturnType();
            if (retType instanceof jq_Reference) {
                Node node = UnknownTypeNode.get((jq_Reference) retType);
                addToMret(M_bdd, node);
                visitNode(V1V2context, node);
            }
            M_bdd.free();
            if (CONTEXT_SENSITIVE) {
                V1V2context.free();
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
            int I_i = Imap.get(mc);
            BDD I_bdd = I.ithVar(I_i);
            jq_Method target = mc.getTargetMethod();
            if (mc.isSingleTarget()) {
                BDD context = null;
                if (CONTEXT_SENSITIVE) {
                    context = bdd.one(); //TODO.
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
            
            visitNode(V1V2context, node);
        }
        if (CONTEXT_SENSITIVE) {
            V1V2context.free();
        }
    }
    
    public void visitNode(BDD V1V2context, Node node) {
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
        
        if (node instanceof ConcreteTypeNode ||
            node instanceof ConcreteObjectNode ||
            node instanceof UnknownTypeNode ||
            node instanceof GlobalNode) {
            addToVP(V_bdd, node);
        }
        
        if (node instanceof GlobalNode) {
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
        if (TRACE_RELATIONS) out.println("Adding to vT: (V1:"+V_i+",T1:"+T_i+")");
        vT.orWith(bdd1);
    }
    
    void addToHT(int H_i, jq_Reference type) {
        BDD bdd1 = H1.ithVar(H_i);
        int T_i = Tmap.get(type);
        bdd1.andWith(T2.ithVar(T_i));
        if (TRACE_RELATIONS) out.println("Adding to hT: (H1:"+H_i+",T2:"+T_i+")");
        hT.orWith(bdd1);
    }
    
    void addToAT(BDD T1_bdd, int T2_i) {
        BDD bdd1 = T2.ithVar(T2_i);
        bdd1.andWith(T1_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to aT: (T1:"+T1_bdd.scanVar(T1)+",T2:"+T2_i+")");
        aT.orWith(bdd1);
    }
    
    void addToCHA(BDD T_bdd, int N_i, jq_Method m) {
        BDD bdd1 = N.ithVar(N_i);
        int M_i = Mmap.get(m);
        bdd1.andWith(M.ithVar(M_i));
        bdd1.andWith(T_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to cha: (T:"+T_bdd.scanVar(T2)+",N:"+N_i+",M:"+M_i+")");
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
        
        // build up 'hT', and identify clinit and thread run methods.
        for (int H_i = last_H; H_i < Hmap.size(); ++H_i) {
            Node n = (Node) Hmap.get(H_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) type.prepare();
            addToHT(H_i, type);
            
            if (type != null) {
                if (n instanceof ConcreteTypeNode && type instanceof jq_Class)
                    addClassInitializer((jq_Class) type);
                if (ADD_THREADS &&
                    (type.isSubtypeOf(PrimordialClassLoader.getJavaLangThread()) ||
                     type.isSubtypeOf(PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;")))) {
                    addThreadRun(H_i, (jq_Class) type);
                }
            }
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
    
    jq_NameAndDesc run_method = new jq_NameAndDesc("run", "()V");
    public void addThreadRun(int H_i, jq_Class c) {
        if (!ADD_THREADS) return;
        jq_Method m = c.getVirtualMethod(run_method);
        if (m != null && m.getBytecode() != null) {
            visitMethod(m);
            roots.add(m);
            Node p = MethodSummary.getSummary(CodeCache.getCode(m)).getParamNode(0);
            addToVP(p, H_i);
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
        old_vP.free();
        old_hP.free();
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
        BDD t6 = t5.relprod(cha, T2Nset); // IxT2xN x T2xNxM = IxM
        t5.free();
        if (TRACE_SOLVER) out.println("Call graph edges before: "+IE.satCount(IMset));
        IE.orWith(t6);
        if (TRACE_SOLVER) out.println("Call graph edges after: "+IE.satCount(IMset));
        
        old3_vP = vP.id();
        old3_hT = hT.id();
    }
    
    public boolean handleNewTargets() {
        BDD targets = IE.exist(Iset);
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
        
        BDD t1 = IE.relprod(actual, Iset); // V2cxIxV1cxM x IxZxV2 = V1cxMxZxV2cxV2
        BDD t2 = t1.relprod(formal, MZset); // V1cxMxZxV2cxV2 x MxZxV1 = V1cxV1xV2cxV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD IEr = IE.replace(V1cV2ctoV2cV1c);
        BDD t3 = IEr.relprod(Iret, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t4 = t3.relprod(Mret, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = IEr.relprod(Ithr, Iset); // V1cxIxV2cxM x IxV1 = V1cxV1xV2cxM
        BDD t6 = t5.relprod(Mthr, Mset); // V1cxV1xV2cxM x MxV2 = V1cxV1xV2cxV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
    }
    
    public void bindParameters_incremental() {
        BDD new_IE = IE.apply(old2_IE, BDDFactory.diff);
        BDD new_visited = visited.apply(old2_visited, BDDFactory.diff);
        new_IE.orWith(old2_IE.and(new_visited));
        old2_IE.free();
        old2_visited.free();
        new_visited.free();
        
        BDD t1 = new_IE.relprod(actual, Iset); // IxM x IxZxV2 = ZxV2xM
        BDD t2 = t1.relprod(formal, MZset); // ZxV2xM x ZxMxV1 = V1xV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD t3 = new_IE.relprod(Iret, Iset); // IxM x IxV1 = V1xM
        BDD t4 = t3.relprod(Mret, Mset); // V1xM x MxV2 = V1xV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = new_IE.relprod(Ithr, Iset); // IxM x IxV1 = V1xM
        BDD t6 = t5.relprod(Mthr, Mset); // V1xM x MxV2 = V1xV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
        new_IE.free();
        old2_IE = IE.id();
        old2_visited = visited.id();
    }
    
    public void iterate() {
        BDD IE_old = IE.id();
        boolean change;
        for (int major = 1; ; ++major) {
            change = false;
            
            if (TRACE_SOLVER) out.println("Major iteration "+major);
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
        if (new File(callgraphfilename).exists()) {
            try {
                System.out.print("Loading initial call graph...");
                long time = System.currentTimeMillis();
                CallGraph cg = new LoadedCallGraph("callgraph");
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
    
    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        PA dis = new PA();
        
        if (dis.CONTEXT_SENSITIVE) {
            CallGraph cg = loadCallGraph(roots);
            dis.numberPaths(cg);
        }
        
        dis.initialize();
        dis.roots.addAll(roots);
        
        long time = System.currentTimeMillis();
        
        GlobalNode.GLOBAL.addDefaultStatics();
        BDD V1V2context = null;
        if (dis.CONTEXT_SENSITIVE) {
            V1V2context = dis.bdd.one();
        }
        dis.visitNode(V1V2context, GlobalNode.GLOBAL);
        for (Iterator i = ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            dis.visitNode(V1V2context, (ConcreteObjectNode) i.next());
        }
        if (dis.CONTEXT_SENSITIVE) {
            V1V2context.free();
        }
        
        for (Iterator i = roots.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            dis.visitMethod(m);
        }
        
        dis.iterate();
        
        System.out.println("Time spent solving: "+(System.currentTimeMillis()-time)/1000.);

        dis.printSizes();
        
        System.out.println("Writing results...");
        time = System.currentTimeMillis();
        dis.dumpResults(resultsFileName);
        System.out.println("Time spent writing: "+(System.currentTimeMillis()-time)/1000.);
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
    
    public void dumpResults(String dumpfilename) throws IOException {
        
        //CallGraph callgraph = CallGraph.makeCallGraph(roots, new PACallTargetMap());
        CallGraph callgraph = new CachedCallGraph(new PACallGraph(this));
        //CallGraph callgraph = callGraph;
        DataOutputStream dos;
        dos = new DataOutputStream(new FileOutputStream("callgraph"));
        LoadedCallGraph.write(callgraph, dos);
        dos.close();
        
        bdd.save(dumpfilename+".vP", vP);
        bdd.save(dumpfilename+".hP", hP);
        bdd.save(dumpfilename+".S", S);
        bdd.save(dumpfilename+".L", L);
        
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
        dumpConfig(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Vmap"));
        dumpMap(dos, Vmap);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Hmap"));
        dumpMap(dos, Hmap);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Fmap"));
        dumpMap(dos, Fmap);
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
    
    private void dumpMap(DataOutput out, IndexMap m) throws IOException {
        int n = m.size();
        out.writeBytes(n+"\n");
        int j = 0;
        while (j < m.size()) {
            Object o = m.get(j);
            if (o == null) {
                out.writeBytes("null");
            } else if (o instanceof Node) {
                ((Node) o).write(m, out);
            } else if (o instanceof jq_Member) {
                ((jq_Member) o).writeDesc(out);
            } else {
                throw new InternalError(o.toString());
            }
            out.writeByte('\n');
            //System.out.println(j+": "+o);
            ++j;
        }
    }

    static class ThreadRootMap extends AbstractMap {
        Set roots;
        ThreadRootMap(Set s) {
            roots = s;
        }
        public Object get(Object o) {
            if (roots.contains(o)) return new Integer(1);
            return new Integer(0);
        }
        /* (non-Javadoc)
         * @see java.util.AbstractMap#entrySet()
         */
        public Set entrySet() {
            throw new UnsupportedOperationException();
        }
    }
    
    static Set thread_runs = new HashSet();
    
    public PathNumbering countCallGraph(CallGraph cg) {
        Set fields = new HashSet();
        Set classes = new HashSet();
        int vars = 0, heaps = 0, bcodes = 0, methods = 0, calls = 0;
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            ++methods;
            if (m.getBytecode() == null) continue;
            bcodes += m.getBytecode().length;
            if (m.getNameAndDesc().equals(run_method)) {
                jq_Class k = m.getDeclaringClass();
                k.prepare();
                PrimordialClassLoader.getJavaLangThread().prepare();
                jq_Class jlr = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;");
                jlr.prepare();
                if (k.isSubtypeOf(PrimordialClassLoader.getJavaLangThread()) ||
                    k.isSubtypeOf(jlr)) {
                    if (TRACE) System.out.println("Thread run method found: "+m);
                    thread_runs.add(m);
                }
            }
            MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
            for (Iterator j = ms.nodeIterator(); j.hasNext(); ) {
                Node n = (Node) j.next();
                ++vars;
                if (n instanceof ConcreteTypeNode ||
                    n instanceof UnknownTypeNode ||
                    n instanceof ConcreteObjectNode)
                    ++heaps;
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
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        System.out.println("Vars="+vars+" Heaps="+heaps+" Classes="+classes.size()+" Fields="+fields.size()+" Paths="+paths);
        double log2 = Math.log(2);
        V_BITS = (int) (Math.log(vars+256)/log2 + 1.0);
        H_BITS = (int) (Math.log(heaps+256)/log2 + 1.0);
        F_BITS = (int) (Math.log(fields.size()+64)/log2 + 2.0);
        T_BITS = (int) (Math.log(classes.size()+64)/log2 + 2.0);
        VC_BITS = paths.bitLength();
        VC_BITS = Math.min(60, VC_BITS);
        System.out.println("Var bits="+V_BITS+" Heap bits="+H_BITS+" Class bits="+T_BITS+" Field bits="+F_BITS);
        System.out.println("Var context bits="+VC_BITS);
        return pn;
    }

    public final HeapPathSelector heapPathSelector = new HeapPathSelector();
    
    public class HeapPathSelector implements Selector {

        /* (non-Javadoc)
         * @see Util.Graphs.PathNumbering.Selector#isImportant(Util.Graphs.SCComponent, Util.Graphs.SCComponent)
         */
        public boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num) {
            if (num.bitLength() > HC_BITS) return false;
            Set s = scc2.nodeSet();
            Iterator i = s.iterator();
            Object o = i.next();
            if (i.hasNext()) return false;
            if (o instanceof ProgramLocation) return true;
            jq_Method m = (jq_Method) o;
            if (!m.getReturnType().isReferenceType()) return false;
            return true;
        }
    }

    public PathNumbering countHeapNumbering(CallGraph cg) {
        PathNumbering pn = new PathNumbering(heapPathSelector);
        Map initialCounts = new ThreadRootMap(thread_runs);
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), initialCounts);
        HC_BITS = paths.bitLength();
        HC_BITS = Math.min(10, HC_BITS);
        System.out.println("Heap context bits="+HC_BITS);
        return pn;
    }
    
}

// PA.java, created Oct 16, 2003 3:39:34 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.PrintStream;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Compil3r.Quad.CodeCache;
import Main.HostedVM;
import Util.Collections.IndexMap;

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
    boolean TRACE_SOLVER = true;
    boolean TRACE_BIND = false;
    boolean TRACE_RELATIONS = false;
    PrintStream out = System.out;

    boolean ADD_CLINIT = true;
    boolean FILTER_TYPE = true;
    boolean INCREMENTAL1 = true;
    boolean INCREMENTAL2 = true;
    
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "5000000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "250000"));
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, H2, Z, F, T1, T2, N, M;
    
    int V_BITS=16, I_BITS=15, H_BITS=15, Z_BITS=5, F_BITS=12, T_BITS=12, N_BITS=11, M_BITS=14;
    
    IndexMap/*Node*/ Vmap;
    IndexMap/*ProgramLocation*/ Imap;
    IndexMap/*Node*/ Hmap;
    IndexMap/*jq_Field*/ Fmap;
    IndexMap/*jq_Reference*/ Tmap;
    IndexMap/*jq_Method*/ Nmap;
    IndexMap/*jq_Method*/ Mmap;
    
    BDD A;      // V1xV2, arguments and return values
    BDD vP;     // V1xH1, variable points-to
    BDD S;      // (V1xF)xV2, stores
    BDD L;      // (V1xF)xV2, loads
    BDD vT;     // V1xT1, variable type
    BDD hT;     // H1xT2, heap type
    BDD aT;     // T1xT2, assignable types
    BDD cha;    // T2xNxM, class hierarchy information
    BDD actual; // IxZxV2, actual parameters
    BDD formal; // MxZxV1, formal parameters
    BDD Iret;   // IxV1, invocation return value
    BDD Mret;   // MxV2, method return value
    BDD Ithr;   // IxV1, invocation thrown value
    BDD Mthr;   // MxV2, method thrown value
    BDD mI;     // MxIxN, method invocations
    BDD mV;     // MxV, method variables
    
    BDD hP;     // H1xFxH2, heap points-to
    BDD IE;     // IxM, invocation edges
    BDD filter; // V1xH1, type filter
    
    BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    BDD V1set, V2set, H1set, H2set, T1set, T2set, Fset, Mset, Nset, Iset, Zset;
    BDD V1V2set, V1H1set, IMset, H1Fset, H1FH2set, T2Nset, MZset;
    
    Set visitedMethods = new HashSet();
    BDD visited; // M
    
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
        
        boolean reverseLocal = System.getProperty("bddreverse", "true") != null;
        String varorder = System.getProperty("bddordering", "M_N_F_Z_I_T2_T1_H2_V2xV1_H1");
        int[] ordering = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(ordering);
        
        Vmap = makeMap("Vars", V_BITS);
        Imap = makeMap("Invokes", I_BITS);
        Hmap = makeMap("Heaps", H_BITS);
        Fmap = makeMap("Fields", F_BITS);
        Tmap = makeMap("Types", T_BITS);
        Nmap = makeMap("Names", N_BITS);
        Mmap = makeMap("Methods", M_BITS);
        
        V1toV2 = bdd.makePair(V1, V2);
        V2toV1 = bdd.makePair(V2, V1);
        V1H1toV2H2 = bdd.makePair();
        V1H1toV2H2.set(new BDDDomain[] {V1,H1},
                       new BDDDomain[] {V2,H2});
        V2H2toV1H1 = bdd.makePair();
        V2H2toV1H1.set(new BDDDomain[] {V2,H2},
                       new BDDDomain[] {V1,H1});
        
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
            old_A = bdd.zero();
            old_S = bdd.zero();
            old_L = bdd.zero();
            old_vP = bdd.zero();
            old_hP = bdd.zero();
        }
        if (INCREMENTAL2) {
            old_IE = bdd.zero();
            old_visited = bdd.zero();
        }

    }
    
    public void visitMethod(jq_Method m) {
        if (m == null) return;
        if (visitedMethods.contains(m)) return;
        visitedMethods.add(m);
        
        if (TRACE) out.println("Visiting method "+m);
        m.getDeclaringClass().prepare();
        
        int M_i = Mmap.get(m);
        BDD M_bdd = M.ithVar(M_i);
        if (TRACE_RELATIONS) out.println("Adding to visited: (M:"+M_i+")");
        visited.orWith(M_bdd.id());
        
        if (m.getBytecode() == null) return;
        
        MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
        if (TRACE) out.println("Visiting method summary "+ms);
        
        if (ADD_CLINIT)
            visitMethod(ms.getMethod().getDeclaringClass().getClassInitializer());
        
        // build up 'formal'
        int nParams = ms.getNumOfParams();
        int offset = ms.getMethod().isStatic()?1:0;
        for (int i = 0; i < nParams; ++i) {
            Node node = ms.getParamNode(i);
            if (node == null) continue;
            int Z_i = i + offset;
            BDD Z_bdd = Z.ithVar(Z_i);
            int V_i = Vmap.get(node);
            BDD V_bdd = V1.ithVar(V_i);
            if (TRACE_RELATIONS) out.println("Adding to formal: (M:"+M_i+",Z:"+Z_i+",V1:"+V_i+")");
            formal.orWith(M_bdd.and(Z_bdd).and(V_bdd));
        }
        
        // build up 'mI', 'actual', 'Iret', 'Ithr'
        for (Iterator i = ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            if (TRACE) out.println("Visiting call site "+mc);
            int I_i = Imap.get(mc);
            BDD I_bdd = I.ithVar(I_i);
            jq_Method target = mc.getTargetMethod();
            if (mc.isSingleTarget()) {
                int M2_i = Mmap.get(target);
                BDD M2_bdd = M.ithVar(M2_i);
                if (TRACE_RELATIONS) out.println("Adding to IE: (I:"+I_i+",M:"+M2_i+")");
                IE.orWith(I_bdd.and(M2_bdd));
            } else {
                int N_i = Nmap.get(target);
                BDD N_bdd = N.ithVar(N_i);
                if (TRACE_RELATIONS) out.println("Adding to mI: (M:"+M_i+",I:"+I_i+",N:"+N_i+")");
                mI.orWith(M_bdd.and(I_bdd).and(N_bdd));
            }
            
            if (ADD_CLINIT && target.isStatic())
                visitMethod(target.getDeclaringClass().getClassInitializer());
            
            if (target.isStatic()) {
                int Z_i = 0;
                BDD Z_bdd = Z.ithVar(Z_i);
                int V_i = Vmap.get(GlobalNode.GLOBAL);
                BDD V_bdd = V2.ithVar(V_i);
                if (TRACE_RELATIONS) out.println("Adding to actual: (I:"+I_i+",Z:"+Z_i+",V2:"+V_i+")");
                actual.orWith(I_bdd.and(Z_bdd).and(V_bdd));
                offset = 1;
            } else {
                offset = 0;
            }
            jq_Type[] params = mc.getParamTypes();
            for (int k = 0; k<params.length; ++k) {
                if (!params[k].isReferenceType()) continue;
                int Z_i = k + offset;
                BDD Z_bdd = Z.ithVar(Z_i);
                Set s = ms.getNodesThatCall(mc, k);
                BDD V_bdd = bdd.zero();
                for (Iterator j = s.iterator(); j.hasNext(); ) {
                    int V_i = Vmap.get(j.next());
                    if (TRACE_RELATIONS) out.println("Adding to actual: (I:"+I_i+",Z:"+Z_i+",V2:"+V_i+")");
                    V_bdd.orWith(V2.ithVar(V_i));
                }
                actual.orWith(I_bdd.and(Z_bdd).and(V_bdd));
            }
            Node node = ms.getRVN(mc);
            if (node != null) {
                int V_i = Vmap.get(node);
                BDD V_bdd = V1.ithVar(V_i);
                if (TRACE_RELATIONS) out.println("Adding to Iret: (I:"+I_i+",V1:"+V_i+")");
                Iret.orWith(I_bdd.and(V_bdd));
            }
            node = ms.getTEN(mc);
            if (node != null) {
                int V_i = Vmap.get(node);
                BDD V_bdd = V1.ithVar(V_i);
                if (TRACE_RELATIONS) out.println("Adding to Iret: (I:"+I_i+",V1:"+V_i+")");
                Ithr.orWith(I_bdd.and(V_bdd));
            }
        }
        // build up 'mV', 'vP', 'S', 'L', 'Mret', 'Mthr'
        for (Iterator i = ms.nodeIterator(); i.hasNext(); ) {
            Node node = (Node) i.next();
            int V_i = Vmap.get(node);
            BDD V_bdd = V1.ithVar(V_i);
            if (TRACE_RELATIONS) out.println("Adding to mV: (M:"+M_i+",V1:"+V_i+")");
            mV.orWith(M_bdd.and(V_bdd));
            
            if (ms.getReturned().contains(node)) {
                BDD V2_bdd = V2.ithVar(V_i);
                if (TRACE_RELATIONS) out.println("Adding to Mret: (M:"+M_i+",V2:"+V_i+")");
                Mret.orWith(M_bdd.and(V2_bdd));
            }
            
            if (ms.getThrown().contains(node)) {
                BDD V2_bdd = V2.ithVar(V_i);
                if (TRACE_RELATIONS) out.println("Adding to Mthr: (M:"+M_i+",V2:"+V_i+")");
                Mthr.orWith(M_bdd.and(V2_bdd));
            }
            
            visitNode(node);
        }
    }
    
    public void visitNode(Node node) {
        if (TRACE) out.println("Visiting node "+node);
        
        int V_i = Vmap.get(node);
        BDD V_bdd = V1.ithVar(V_i);
        
        if (node instanceof ConcreteTypeNode ||
            node instanceof ConcreteObjectNode ||
            node instanceof UnknownTypeNode ||
            node instanceof GlobalNode) {
            int H_i = Hmap.get(node);
            BDD H_bdd = H1.ithVar(H_i);
            if (TRACE_RELATIONS) out.println("Adding to vP: (V1:"+V_i+",H1:"+H_i+")");
            vP.orWith(V_bdd.and(H_bdd));
        }
        
        if (node instanceof GlobalNode) {
            int V2_i = Vmap.get(GlobalNode.GLOBAL);
            BDD V2_bdd = V2.ithVar(V2_i);
            A.orWith(V_bdd.and(V2_bdd));
            
            BDD V1_bdd = V1.ithVar(V2_i);
            V2_bdd = V2.ithVar(V_i);
            A.orWith(V1_bdd.and(V2_bdd));
        }
        
        for (Iterator j = node.getEdgeFields().iterator(); j.hasNext(); ) {
            jq_Field f = (jq_Field) j.next();
            int F_i = Fmap.get(f);
            BDD F_bdd = F.ithVar(F_i);
            for (Iterator k = node.getEdges(f).iterator(); k.hasNext(); ) {
                Node node2 = (Node) k.next();
                int V2_i = Vmap.get(node2);
                BDD V2_bdd = V2.ithVar(V2_i);
                if (TRACE_RELATIONS) out.println("Adding to S: (V1:"+V_i+",F:"+F_i+",V2:"+V2_i+")");
                S.orWith(V_bdd.and(F_bdd).and(V2_bdd));
            }
        }
        
        for (Iterator j = node.getAccessPathEdgeFields().iterator(); j.hasNext(); ) {
            jq_Field f = (jq_Field) j.next();
            int F_i = Fmap.get(f);
            BDD F_bdd = F.ithVar(F_i);
            for (Iterator k = node.getAccessPathEdges(f).iterator(); k.hasNext(); ) {
                Node node2 = (Node) k.next();
                int V2_i = Vmap.get(node2);
                BDD V2_bdd = V2.ithVar(V2_i);
                if (TRACE_RELATIONS) out.println("Adding to L: (V1:"+V_i+",F:"+F_i+",V2:"+V2_i+")");
                L.orWith(V_bdd.and(F_bdd).and(V2_bdd));
            }
            if (ADD_CLINIT && node instanceof GlobalNode)
                visitMethod(f.getDeclaringClass().getClassInitializer());
        }
    }
    
    int last_V = 0;
    int last_H = 0;
    int last_T = 0;
    int last_N = 0;
    
    public void buildTypes() {
        // build up 'vT'
        for (int V_i = last_V; V_i < Vmap.size(); ++V_i) {
            Node n = (Node) Vmap.get(V_i);
            BDD V_bdd = V1.ithVar(V_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) type.prepare();
            int T_i = Tmap.get(type);
            BDD T_bdd = T1.ithVar(T_i);
            if (TRACE_RELATIONS) out.println("Adding to vT: (V1:"+V_i+",T1:"+T_i+")");
            vT.orWith(V_bdd.and(T_bdd));
        }
        
        // build up 'hT'
        for (int H_i = last_H; H_i < Hmap.size(); ++H_i) {
            Node n = (Node) Hmap.get(H_i);
            BDD H_bdd = H1.ithVar(H_i);
            jq_Reference type = n.getDeclaredType();
            if (type != null) type.prepare();
            int T_i = Tmap.get(type);
            BDD T_bdd = T2.ithVar(T_i);
            if (TRACE_RELATIONS) out.println("Adding to hT: (H1:"+H_i+",T2:"+T_i+")");
            hT.orWith(H_bdd.and(T_bdd));
        }
        
        // build up 'aT'
        for (int T1_i = 0; T1_i < Tmap.size(); ++T1_i) {
            jq_Reference t1 = (jq_Reference) Tmap.get(T1_i);
            BDD T1_bdd = T1.ithVar(T1_i);
            int start = (T1_i < last_T)?last_T:0;
            for (int T2_i = start; T2_i < Tmap.size(); ++T2_i) {
                jq_Reference t2 = (jq_Reference) Tmap.get(T2_i);
                BDD T2_bdd = T2.ithVar(T2_i);
                if (t2 == null || (t1 != null && t2.isSubtypeOf(t1))) {
                    if (TRACE_RELATIONS) out.println("Adding to aT: (T1:"+T1_i+",T2:"+T2_i+")");
                    aT.orWith(T1_bdd.and(T2_bdd));
                }
            }
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
                BDD N_bdd = N.ithVar(N_i);
                int M_i = Mmap.get(m);
                BDD M_bdd = M.ithVar(M_i);
                if (TRACE_RELATIONS) out.println("Adding to cha: (T:"+T_i+",N:"+N_i+",M:"+M_i+")");
                cha.orWith(T_bdd.and(N_bdd).and(M_bdd));
            }
        }
        last_V = Vmap.size();
        last_H = Hmap.size();
        last_T = Tmap.size();
        last_N = Nmap.size();
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
    
    BDD old_A;
    BDD old_S;
    BDD old_L;
    BDD old_vP;
    BDD old_hP;
    
    public void solvePointsTo_incremental() {
        // handle new A
        BDD new_A = A.apply(old_A, BDDFactory.diff);
        old_A.free();
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
        old_A = A.id();
        
        // handle new S
        BDD new_S = S.apply(old_S, BDDFactory.diff);
        old_S.free();
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
        old_S = S.id();
        
        // handle new L
        BDD new_L = L.apply(old_L, BDDFactory.diff);
        old_L.free();
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
        old_L = S.id();
        
        for (int outer = 1; ; ++outer) {
            BDD new_vP_inner = vP.apply(old_vP, BDDFactory.diff);
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
            
            BDD new_vP = vP.apply(old_vP, BDDFactory.diff);
            old_vP.free();
            
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
                out.println(", hP "+old_hP.satCount(H1FH2set)+" -> "+hP.satCount(H1FH2set));
            
            old_vP = vP.id();
            
            BDD new_hP = hP.apply(old_hP, BDDFactory.diff);
            if (new_hP.isZero()) break;
            old_hP = hP.id();
            
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
                out.println(", vP "+old_vP.satCount(V1H1set)+
                            " -> "+vP.satCount(V1H1set));
        }
    }
    
    public void bindInvocations() {
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
    
    BDD old_IE;
    BDD old_visited;
    
    public void bindParameters() {
        if (INCREMENTAL2) {
            bindParameters_incremental();
            return;
        }
        
        BDD t1 = IE.relprod(actual, Iset); // IxM x IxZxV2 = ZxV2xM
        BDD t2 = t1.relprod(formal, MZset); // ZxV2xM x ZxMxV1 = V1xV2
        t1.free();
        if (TRACE_SOLVER) out.println("Edges before param bind: "+A.satCount(V1V2set));
        A.orWith(t2);
        if (TRACE_SOLVER) out.println("Edges after param bind: "+A.satCount(V1V2set));
        
        BDD t3 = IE.relprod(Iret, Iset); // IxM x IxV1 = V1xM
        BDD t4 = t3.relprod(Mret, Mset); // V1xM x MxV2 = V1xV2
        t3.free();
        if (TRACE_SOLVER) out.println("Edges before return bind: "+A.satCount(V1V2set));
        A.orWith(t4);
        if (TRACE_SOLVER) out.println("Edges after return bind: "+A.satCount(V1V2set));
        
        BDD t5 = IE.relprod(Ithr, Iset); // IxM x IxV1 = V1xM
        BDD t6 = t5.relprod(Mthr, Mset); // V1xM x MxV2 = V1xV2
        t5.free();
        if (TRACE_SOLVER) out.println("Edges before exception bind: "+A.satCount(V1V2set));
        A.orWith(t6);
        if (TRACE_SOLVER) out.println("Edges after exception bind: "+A.satCount(V1V2set));
        
    }
    
    public void bindParameters_incremental() {
        BDD new_IE = IE.apply(old_IE, BDDFactory.diff);
        BDD new_visited = visited.apply(old_visited, BDDFactory.diff);
        new_IE.orWith(old_IE.and(new_visited));
        old_IE.free();
        old_visited.free();
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
        old_IE = IE.id();
        old_visited = visited.id();
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
    
    public static void main(String[] args) {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        PA dis = new PA();
        dis.initialize();
        
        long time = System.currentTimeMillis();
        
        GlobalNode.GLOBAL.addDefaultStatics();
        dis.visitNode(GlobalNode.GLOBAL);
        for (Iterator i = ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            dis.visitNode((ConcreteObjectNode) i.next());
        }
        
        for (Iterator i = roots.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            dis.visitMethod(m);
        }
        
        dis.iterate();
        
        System.out.println("Total time spent: "+(System.currentTimeMillis()-time)/1000.);

        dis.printSizes();
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
                case 0: 
                case 1: return Vmap.get((int)j).toString();
                case 2: return Imap.get((int)j).toString();
                case 3: 
                case 4: return Hmap.get((int)j).toString();
                case 5: return Long.toString(j);
                case 6: return ""+Fmap.get((int)j);
                case 7: 
                case 8: return ""+Tmap.get((int)j);
                case 9: return Nmap.get((int)j).toString();
                case 10: return Mmap.get((int)j).toString();
                default: return "??";
            }
        }
    }
}

// BDDPointerAnalysis.java, created Sun Feb  2  2:22:10 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_Field;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.CallSite;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.FieldNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.PassedParameter;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnValueNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ThrownExceptionNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.CallTargets;
import Main.HostedVM;
import Run_Time.TypeCheck;
import Util.Assert;
import Util.Collections.HashCodeComparator;
import Util.Collections.IndexMap;
import Util.Collections.SortedArraySet;

/**
 * This is an implementation of the "Points-to Analysis using BDDs" algorithm
 * described in the PLDI 2003 paper by Berndl, Lhotak, Qian, Hendren and Umanee.
 * This code is based on their original implementation available at:
 * http://www.sable.mcgill.ca/bdd/.  This version has been rewritten in Java and
 * requires the open-source JavaBDD library, available at http://javabdd.sf.net.
 * 
 * This implementation extends Berndl et al.'s algorithm to support on-the-fly
 * computation of the call graph using BDDs.  See the handleVirtualCalls() method.
 * 
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class BDDPointerAnalysis {

    public static final boolean TRACE = false;
    public static final boolean TRACE_BDD = false;
    public static final boolean TRACE_INST = false;
    public static final boolean TRACE_VIRTUAL = false;

    /**
     * The default initial node count.  Smaller values save memory for
     * smaller problems, larger values save the time to grow the node tables
     * on larger problems.
     */
    public static final int DEFAULT_NODE_COUNT = Integer.parseInt(System.getProperty("bddnodes", "1000000"));

    /**
     * The absolute maximum number of variables that we will ever use
     * in the BDD.  Smaller numbers will be more efficient, larger
     * numbers will allow larger programs to be analyzed.
     */
    public static final int DEFAULT_CACHE_SIZE = 100000;

    /**
     * Singleton BDD object that provides access to BDD functions.
     */
    private BDDFactory bdd;

    public int V_BITS = 18;
    public int F_BITS = 13;
    public int H_BITS = 14;
    public int T_BITS = 1;
    public int I_BITS = 1;
    public int Z_BITS = 1;
    public int N_BITS = 1;
    public int M_BITS = 1;
    public int VC_BITS = 1;
    public int HC_BITS = 1;
    
    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1, V2, FD, H1, H2;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2, T3, T4; 
    // context domains: unused
    BDDDomain V1c, V2c, H1c, H2c;

    // domain pairs for bdd_replace
    BDDPairing V1ToV2;
    BDDPairing V2ToV1;
    BDDPairing H1ToH2;
    BDDPairing H2ToH1;
    BDDPairing T2ToT1;

    // relations
    BDD pointsTo;     // V1 x H1
    BDD edgeSet;      // V1 x V2
    BDD typeFilter;   // V1 x H1
    BDD stores;       // V1 x (V2 x FD) 
    BDD loads;        // (V1 x FD) x V2

    // cached temporary relations
    BDD storePt;      // (V1 x FD) x H2
    BDD fieldPt;      // (H1 x FD) x H2
    BDD loadAss;      // (H1 x FD) x V2
    BDD loadPt;       // V2 x H2

    BDD V1set;
    BDD V1andH1set;
    BDD V1andV2andFDset;
    BDD T1set;
    BDD T2set;
    BDD H1andFDset;
    BDD H1andT3set;

    public BDDPointerAnalysis() {
        this(System.getProperty("bdd", "buddy"), DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
    
    BDDDomain makeDomain(String name, int bits) {
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }
    
    String varorder = System.getProperty("bddordering", "N_F_Z_I_M_T1_V2xV1_V2cxV1c_H2c_H2_T2_H1c_H1");
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
    
    public BDDPointerAnalysis(String bddpackage, int nodeCount, int cacheSize) {
        bdd = org.sf.javabdd.BDDFactory.init(bddpackage, nodeCount, cacheSize);
        
        bdd.setCacheRatio(8);
        bdd.setMaxIncrease(Math.min(2500000, nodeCount / 4));
        bdd.setMinFreeNodes(10);
        
        V1 = makeDomain("V1", V_BITS);
        V2 = makeDomain("V2", V_BITS);
        makeDomain("I", I_BITS);
        H1 = makeDomain("H1", H_BITS);
        H2 = makeDomain("H2", H_BITS);
        makeDomain("Z", Z_BITS);
        FD = makeDomain("F", F_BITS);
        makeDomain("T1", T_BITS);
        makeDomain("T2", T_BITS);
        makeDomain("N", N_BITS);
        makeDomain("M", M_BITS);
        
        V1c = makeDomain("V1c", VC_BITS);
        V2c = makeDomain("V2c", VC_BITS);
        H1c = makeDomain("H1c", HC_BITS);
        H2c = makeDomain("H2c", HC_BITS);
        
        // IxH1xN x H1xT2
        int[] ordering = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(ordering);
        
        Vmap = makeMap("Vars", V_BITS);
        Hmap = makeMap("Heaps", H_BITS);
        Fmap = makeMap("Fields", F_BITS);
        Tmap = makeMap("Types", T_BITS);
        
        T1 = V2;
        T2 = V1;
        T3 = H2;
        T4 = V2;
        
        V1ToV2 = bdd.makePair(V1, V2);
        V2ToV1 = bdd.makePair(V2, V1);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
        T2ToT1 = bdd.makePair(T2, T1);
        
        V1set = V1.set();
        T1set = T1.set();
        T2set = T2.set();
        H1andFDset = H1.set(); H1andFDset.andWith(FD.set());
        H1andT3set = H1.set(); H1andT3set.andWith(T3.set());
        V1andH1set = V1.set(); V1andH1set.andWith(H1.set());
        V1andV2andFDset = V1.set();
        V1andV2andFDset.andWith(V2.set());
        V1andV2andFDset.andWith(FD.set());
        
        reset();
    }

    void reset() {
        // initialize relations to zero.
        pointsTo = bdd.zero();
        edgeSet = bdd.zero();
        typeFilter = bdd.zero();
        stores = bdd.zero();
        loads = bdd.zero();
        storePt = bdd.zero();
        fieldPt = bdd.zero();
        loadAss = bdd.zero();
        loadPt = bdd.zero();
        
        aC = bdd.zero(); vC = bdd.zero(); cC = bdd.zero();
        vtable_bdd = bdd.zero();
    }
    
    public void done() {
        aC.free();
        vC.free();
        cC.free();
        if (cTypes != null) cTypes.free();
        pointsTo.free();
        edgeSet.free();
        typeFilter.free();
        stores.free();
        loads.free();
        storePt.free();
        fieldPt.free();
        loadAss.free();
        loadPt.free();
        vtable_bdd.free();
        bdd.done();
        bdd = null;
        System.gc();
    }

    public static boolean INCREMENTAL_POINTSTO = true;
    public static boolean INCREMENTAL_ITERATION = false;
    public static boolean FORCE_GC = false;

    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        
        CodeCache.AlwaysMap = true;
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        BDDPointerAnalysis dis = new BDDPointerAnalysis();
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        CallGraph cg = INCREMENTAL_ITERATION ?
                       dis.goIncremental(roots) :
                       dis.goNonincremental(roots);
        
        cg = new CachedCallGraph(cg);
        
        Collection[] depths = cg.findDepths();
        for (int i=0; i<depths.length; ++i) {
            System.out.println(">>>>> Depth "+i+": size "+depths[i].size());
        }
        
        RootedCHACallGraph.test(cg);
        
        System.out.println("Variables: "+dis.Vmap.size());
        System.out.println("Heap objects: "+dis.Hmap.size());
        System.out.println("Fields: "+dis.Fmap.size());
        System.out.println("Types: "+dis.Tmap.size());
        System.out.println("Virtual Method Names: "+dis.methodIndexMap.size());
        System.out.println("Virtual Method Targets: "+dis.targetIndexMap.size());

        int bc = 0;
        for (Iterator i=dis.visitedMethods.iterator(); i.hasNext(); ) {
            MethodSummary ms = (MethodSummary) i.next();
            jq_Method m = (jq_Method) ms.getMethod();
            bc += m.getBytecode().length;
        }
        System.out.println("Bytecodes: "+bc);
        
        try {
            FileOutputStream o = new FileOutputStream("callgraph");
            DataOutputStream d = new DataOutputStream(o);
            LoadedCallGraph.write(cg, d);
            d.close(); o.close();
        } catch (java.io.IOException x) {
            x.printStackTrace();
        }
        
        if (DUMP)
            dis.dumpResults(cg);
        dis.dumpResults("pa");
            
    }
    
    public CallGraph goNonincremental(Collection roots) {
        long time = System.currentTimeMillis();
        
        GlobalNode.GLOBAL.addDefaultStatics();
        this.addObjectAllocation(GlobalNode.GLOBAL, null);
        this.addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        this.addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        this.handleNode(GlobalNode.GLOBAL, false);
        for (Iterator i=ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            this.handleNode((ConcreteObjectNode) i.next(), false);
        }
        
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod) i.next();
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                this.handleMethodSummary(ms);
            }
        }
        boolean first = true;
        
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        int iteration = 1;
        do {
            long time2 = System.currentTimeMillis();

            System.out.println("--> Iteration "+iteration+" Methods: "+this.visitedMethods.size()+" Call graph edges: "+this.callGraphEdges.size());
            this.change = first;
            first = false;
            this.calculateTypeFilter();

            long time3 = System.currentTimeMillis();
            System.out.println("Calculate type filter:\t"+(time3-time2)/1000.+" seconds.");

            oldPointsTo = bdd.zero();
            if (INCREMENTAL_POINTSTO) this.solveIncremental();
            else this.solveNonincremental();
            oldPointsTo.free();

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Solve pointers:\t\t"+time3/1000.+" seconds.");

            time3 = System.currentTimeMillis();

            this.calculateVTables();

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Calculate vtables:\t"+time3/1000.+" seconds.");

            time3 = System.currentTimeMillis();

            this.handleVirtualCalls(pointsTo);

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Handle virtual calls:\t"+time3/1000.+" seconds.");

            if (FORCE_GC) {
                time3 = System.currentTimeMillis();
                System.gc();
                time3 = System.currentTimeMillis() - time3;
                System.out.println("Garbage collection:\t"+time3/1000.+" seconds.");
            }

            time2 = System.currentTimeMillis() - time2;
            System.out.println("Iteration completed:\t"+time2/1000.+" seconds.");
            ++iteration;
        } while (this.change);
        
        time = System.currentTimeMillis() - time;

        System.out.println("Total time: "+time/1000.+" seconds.");
        
        if (!IGNORE_CLINIT || !IGNORE_THREADS) {
            roots = new HashSet(roots);
            if (!IGNORE_CLINIT)
                roots.addAll(class_initializers);
            if (!IGNORE_THREADS)
                roots.addAll(thread_runs);
        }
        
        CallGraph cg = CallGraph.makeCallGraph(roots, callSiteToTargets);
        return cg;
    }
    
    public CallGraph goIncremental(Collection roots) {
        long time = System.currentTimeMillis();
        
        GlobalNode.GLOBAL.addDefaultStatics();
        this.addObjectAllocation(GlobalNode.GLOBAL, null);
        this.addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        this.addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        this.handleNode(GlobalNode.GLOBAL, false);
        for (Iterator i=ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            this.handleNode((ConcreteObjectNode) i.next(), false);
        }
        
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod) i.next();
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                this.handleMethodSummary(ms);
            }
        }
        boolean first = true;
        
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        int iteration = 1;
        oldPointsTo = this.bdd.zero();
        do {
            long time2 = System.currentTimeMillis();

            System.out.println("--> Iteration "+iteration+" Methods: "+this.visitedMethods.size()+" Call graph edges: "+this.callGraphEdges.size());
            this.change = first;
            first = false;
            this.calculateTypeFilter();

            long time3 = System.currentTimeMillis();
            System.out.println("Calculate type filter:\t"+(time3-time2)/1000.+" seconds.");

            BDD myOldPointsTo = oldPointsTo.id();
            if (TRACE)
                System.out.println("Old points-to relations: "+(long)myOldPointsTo.satCount(V1andH1set));
            if (INCREMENTAL_POINTSTO) this.solveIncremental();
            else this.solveNonincremental();
            oldPointsTo.free();
            if (TRACE)
                System.out.println("Current points-to relations: "+(long)this.pointsTo.satCount(V1andH1set));
            BDD newPointsTo = this.pointsTo.apply(myOldPointsTo, BDDFactory.diff);
            myOldPointsTo.free();
            if (TRACE)
                System.out.println("New points-to relations: "+(long)newPointsTo.satCount(V1andH1set));

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Solve pointers:\t\t"+time3/1000.+" seconds.");

            time3 = System.currentTimeMillis();

            this.calculateVTables();

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Calculate vtables:\t"+time3/1000.+" seconds.");

            time3 = System.currentTimeMillis();

            this.handleVirtualCalls(newPointsTo);
            newPointsTo.free();

            time3 = System.currentTimeMillis() - time3;
            System.out.println("Handle virtual calls:\t"+time3/1000.+" seconds.");

            if (FORCE_GC) {
                time3 = System.currentTimeMillis();
                System.gc();
                time3 = System.currentTimeMillis() - time3;
                System.out.println("Garbage collection:\t"+time3/1000.+" seconds.");
            }

            time2 = System.currentTimeMillis() - time2;
            System.out.println("Iteration completed:\t"+time2/1000.+" seconds.");
            ++iteration;
            oldPointsTo = this.pointsTo.id();
        } while (this.change);
        
        time = System.currentTimeMillis() - time;

        System.out.println("Total time: "+time/1000.+" seconds.");
        
        if (!IGNORE_CLINIT || !IGNORE_THREADS) {
            roots = new HashSet(roots);
            if (!IGNORE_CLINIT)
                roots.addAll(class_initializers);
            if (!IGNORE_THREADS)
                roots.addAll(thread_runs);
        }
        
        CallGraph cg = CallGraph.makeCallGraph(roots, callSiteToTargets);
        return cg;
    }

    boolean change;

    IndexMap getIndexMap(BDDDomain d) {
        if (d == V1 || d == V2) return Vmap;
        if (d == FD) return Fmap;
        if (d == H1 || d == H2) return Hmap;
        return null;
    }

    void printSet(String desc, BDD b) {
        System.out.print(desc+": ");
        System.out.flush();
        //if (desc.startsWith(" "))
            b.printSetWithDomains();
        System.out.println();
    }

    BDD cTypes; // H1 x T1

    public void dumpResults(CallGraph cg) {
        System.out.println(visitedMethods.size()+" methods");
        
        System.out.println(cg);
        
        // (V1xH1) * (H1xT1) => (V1xT1)
        //printSet("Points to", pointsTo, "V1xH1");
        //BDD varTypes = pointsTo.relprod(cTypes, H1.set());
        //printSet("Var types", varTypes, "V1xT1");
        if (TRACE) {
            for (int i=0, n=Vmap.size(); i<n; ++i) {
                Node node = (Node) Vmap.get(i);
                System.out.print(i+": "+node.toString());
                BDD var = V1.ithVar(i);
                BDD p = pointsTo.restrict(var);
                printSet(" can point to", p);
            }
        }
    }

    public Set getPointsTo(Node n) {
        int i = Vmap.get(n);
        BDD var = V1.ithVar(i);
        BDD p = pointsTo.restrict(var);
        HashSet set = new HashSet();
        for (;;) {
            int a = p.scanVar(H1);
            if (a < 0) break;
            set.add(Hmap.get(a));
            p.applyWith(H1.ithVar(a), BDDFactory.diff);
        }
        p.free();
        return set;
    }

    public void dumpNode(Node n) {
        int x = getVariableIndex(n);
        printSet(x+": "+n.toString(), pointsTo.restrict(V1.ithVar(x)));
        for (Iterator i=n.getAccessPathEdges().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_Field f = (jq_Field)e.getKey();
            Object o = e.getValue();
            System.out.println("Field "+f);
            // o = n.f
            Set s;
            if (o instanceof Set) {
                s = (Set) o;
            } else {
                s = Collections.singleton(o);
            }
            for (Iterator j=s.iterator(); j.hasNext(); ) {
                Node n2 = (Node) j.next();
                int x2 = getVariableIndex(n2);
                printSet("==> "+x2+": "+n2.toString(), V1.ithVar(x2));
            }
        }
        System.out.println("---");
    }

    public static final boolean IGNORE_CLINIT = false;
    Set class_initializers = new HashSet();

    public void addClassInit(jq_Type t) {
        if (IGNORE_CLINIT) return;
        if (t instanceof jq_Class) {
            jq_Class c = (jq_Class) t;
            c.prepare();
            jq_ClassInitializer i = c.getClassInitializer();
            if (i != null && i.getBytecode() != null) {
                class_initializers.add(i);
                ControlFlowGraph cfg = CodeCache.getCode(i);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                handleMethodSummary(ms);
            }
        }
    }

    public static final boolean IGNORE_THREADS = false;
    Set thread_runs = new HashSet();
    public void addThreadRun(Node n, jq_Class c) {
        if (IGNORE_THREADS) return;
        c.prepare();
        jq_NameAndDesc nd = new jq_NameAndDesc("run", "()V");
        jq_InstanceMethod i = c.getVirtualMethod(nd);
        if (i != null && i.getBytecode() != null) {
            thread_runs.add(i);
            ControlFlowGraph cfg = CodeCache.getCode(i);
            MethodSummary ms = MethodSummary.getSummary(cfg);
            handleMethodSummary(ms);
            ParamNode p = ms.getParamNode(0);
            addDirectAssignment(p, n);
            this.change = true;
        }
    }
    
    HashSet visitedMethods = new HashSet();

    public static boolean NO_HEAP = System.getProperty("noheap") != null;

    public void handleNode(Node n, boolean add_clinit) {
        if (TRACE) System.out.println("Handling node: "+n);
        
        if (NO_HEAP) {
            addObjectAllocation(n, n);
        }
        
        Iterator j;
        j = n.getAllEdges().iterator();
        while (j.hasNext()) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Object o = e.getValue();
            // n.f = o
            if (o instanceof Set) {
                addFieldStore(n, f, (Set) o);
            } else {
                addFieldStore(n, f, (Node) o);
            }
            if (add_clinit && n instanceof GlobalNode)
                addClassInit(f.getDeclaringClass());
        }
        j = n.getAccessPathEdges().iterator();
        while (j.hasNext()) {
            Map.Entry e = (Map.Entry)j.next();
            jq_Field f = (jq_Field)e.getKey();
            Object o = e.getValue();
            // o = n.f
            if (o instanceof Set) {
                addLoadField((Set) o, n, f);
            } else {
                addLoadField((FieldNode) o, n, f);
            }
            if (add_clinit && n instanceof GlobalNode)
                addClassInit(f.getDeclaringClass());
        }
        if (n instanceof ConcreteTypeNode) {
            ConcreteTypeNode ctn = (ConcreteTypeNode) n;
            addObjectAllocation(ctn, ctn);
            addAllocType(ctn, (jq_Reference) ctn.getDeclaredType());
            if (add_clinit)
                addClassInit((jq_Reference) ctn.getDeclaredType());
        } else if (n instanceof UnknownTypeNode) {
            UnknownTypeNode utn = (UnknownTypeNode) n;
            addObjectAllocation(utn, utn);
            addAllocType(utn, (jq_Reference) utn.getDeclaredType());
        } else if (n instanceof ConcreteObjectNode) {
            addObjectAllocation(n, n);
            addAllocType(n, (jq_Reference) n.getDeclaredType());
        }
        if (n instanceof GlobalNode) {
            addDirectAssignment(GlobalNode.GLOBAL, n);
            addDirectAssignment(n, GlobalNode.GLOBAL);
            addVarType(n, PrimordialClassLoader.getJavaLangObject());
        } else {
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }
    }

    public static HashSet visitedClasses = new HashSet();

    public void handleMethodSummary(MethodSummary ms) {
        if (visitedMethods.contains(ms)) return;
        visitedMethods.add(ms);
        this.change = true;
        jq_Class klass = ms.getMethod().getDeclaringClass();
        if (!visitedClasses.contains(klass)) {
            if (TRACE) System.out.println("Discovered class "+klass);
            visitedClasses.add(klass);
        }
        if (TRACE) System.out.println("Handling method summary: "+ms);
        for (Iterator i = ms.nodeIterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            handleNode(n, true);
        }
        
        addClassInit(((jq_Method) ms.getMethod()).getDeclaringClass());
        
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            Assert._assert(!callSiteToTargets.containsKey(mc));
            if (mc.isSingleTarget()) {
                jq_Method target = (jq_Method) mc.getTargetMethod();
                addClassInit(target.getDeclaringClass());
                Set definite_targets = Collections.singleton(target);
                callSiteToTargets.put(mc, definite_targets);
                if (target.getBytecode() != null) {
                    ControlFlowGraph cfg2 = CodeCache.getCode(target);
                    MethodSummary ms2 = MethodSummary.getSummary(cfg2);
                    handleMethodSummary(ms2);
                    bindParameters(ms, mc, ms2);
                } else {
                    bindParameters_native(ms, mc);
                }
            } else {
                Set definite_targets = SortedArraySet.FACTORY.makeSet(HashCodeComparator.INSTANCE);
                callSiteToTargets.put(mc, definite_targets);
                jq_InstanceMethod method = (jq_InstanceMethod) mc.getTargetMethod();
                addClassInit(method.getDeclaringClass());
                getMethodIndex(method); // allocate index in the method map.
                PassedParameter pp = new PassedParameter(mc, 0);
                Set receiverObjects = ms.getNodesThatCall(pp);
                BDD receiverBDD = bdd.zero();
                for (Iterator j=receiverObjects.iterator(); j.hasNext(); ) {
                    Node receiverNode = (Node) j.next();
                    int receiverIndex = getVariableIndex(receiverNode);
                    BDD tempBDD = V1.ithVar(receiverIndex);
                    receiverBDD.orWith(tempBDD);
                }
                if (TRACE) {
                    System.out.println("Virtual call "+method+" receiver vars "+receiverObjects);
                }
                if (TRACE_BDD) {
                    printSet("receiverBDD", receiverBDD);
                }
                CallSite cs = new CallSite(ms, mc);
                virtualCallSites.add(cs);
                virtualCallReceivers.add(receiverBDD);
                virtualCallMethods.add(method);
            }
        }
    }
    
    HashMap callSiteToTargets = new HashMap();

    List virtualCallSites = new LinkedList();
    List virtualCallReceivers = new LinkedList();
    List virtualCallMethods = new LinkedList();

    int last_vcalls;
    
    public void handleVirtualCalls(BDD newPointsTo) {
        int n = virtualCallSites.size();
        System.out.println(n-last_vcalls+" new virtual call sites");
        Iterator h=new LinkedList(virtualCallSites).iterator();
        Iterator i=new LinkedList(virtualCallReceivers).iterator();
        Iterator j=new LinkedList(virtualCallMethods).iterator();
        if (TRACE_BDD) System.out.println("new points-to: "+newPointsTo.toStringWithDomains());
        for (int index=0; h.hasNext(); ++index) {
            CallSite cs = (CallSite) h.next();
            MethodSummary caller = cs.getCaller();
            ProgramLocation mc = cs.getLocation();
            BDD receiverVars = (BDD) i.next();
            if (TRACE_VIRTUAL) {
                System.out.println("Caller: "+caller.getMethod());
                System.out.println("Call: "+mc);
                printSet(" receiverVars", receiverVars);
            }
            BDD pt;
            if (false || index < last_vcalls) pt = newPointsTo;
            else pt = pointsTo;
            BDD receiverObjects;
            if (false || receiverVars.satCount(V1set) == 1.0) {
                receiverObjects = pt.restrict(receiverVars); // time-consuming!
            } else {
                receiverObjects = pt.relprod(receiverVars, V1set); // time-consuming!
            }
            if (TRACE_VIRTUAL) {
                printSet(" receiverObjects", receiverObjects);
            }
            jq_InstanceMethod method = (jq_InstanceMethod) j.next();
            if (receiverObjects.isZero()) {
                continue;
            }
            int methodIndex = getMethodIndex(method);
            BDD methodBDD = T3.ithVar(methodIndex);
            if (TRACE_VIRTUAL) {
                printSet("Method "+method+" index "+methodIndex, methodBDD);
            }
            receiverObjects.andWith(methodBDD);
            if (TRACE_VIRTUAL) {
                printSet(" receiverObjects", receiverObjects);
            }
            // (H1 x T3) * (H1 x T3 x T4) 
            BDD targets = receiverObjects.relprod(vtable_bdd, H1andT3set);
            receiverObjects.free();
            if (TRACE_VIRTUAL) {
                printSet(" targets", targets);
            }
            if (TRACE_VIRTUAL) {
                System.out.println("# of targets: "+targets.satCount(T4.set()));
            }
            Set definite_targets = (Set) callSiteToTargets.get(mc);
            for (;;) {
                int p = targets.scanVar(T4);
                if (p < 0) break;
                jq_InstanceMethod target = getTarget(p);
                if (TRACE_VIRTUAL) {
                    System.out.println("Target "+p+": "+target);
                }
                definite_targets.add(target);
                if (target.getBytecode() != null) {
                    long time = System.currentTimeMillis();
                    ControlFlowGraph cfg = CodeCache.getCode(target);
                    MethodSummary ms2 = MethodSummary.getSummary(cfg);
                    method_summary_time += time - System.currentTimeMillis();
                    handleMethodSummary(ms2);
                    bindParameters(caller, mc, ms2);
                } else {
                    bindParameters_native(caller, mc);
                }
                targets.applyWith(T4.ithVar(p), BDDFactory.diff);
            }
            targets.free();
        }
        last_vcalls = n;
    }
    
    long method_summary_time = 0L;
    
    HashSet callGraphEdges = new HashSet();
    
    public void bindParameters(MethodSummary caller, ProgramLocation mc, MethodSummary callee) {
        Object key = new CallSite(callee, mc);
        if (callGraphEdges.contains(key)) return;
        if (true && !this.change) {
            System.out.println("Adding call graph edge "+caller.getMethod()+"->"+callee.getMethod());
        }
        callGraphEdges.add(key);
        this.change = true;
        jq_Type[] paramTypes = mc.getParamTypes();
        for (int i=0; i<paramTypes.length; ++i) {
            if (i >= callee.getNumOfParams()) break;
            ParamNode pn = callee.getParamNode(i);
            if (pn == null) continue;
            PassedParameter pp = new PassedParameter(mc, i);
            Set s = caller.getNodesThatCall(pp);
            addDirectAssignment(pn, s);
        }
        ReturnValueNode rvn = caller.getRVN(mc);
        if (rvn != null) {
            Set s = callee.getReturned();
            addDirectAssignment(rvn, s);
        }
        ThrownExceptionNode ten = caller.getTEN(mc);
        if (ten != null) {
            Set s = callee.getThrown();
            addDirectAssignment(ten, s);
        }
    }

    public void bindParameters_native(MethodSummary caller, ProgramLocation mc) {
        // only handle return value for now.
        jq_Type t = mc.getReturnType();
        if (t instanceof jq_Reference) {
            ReturnValueNode rvn = caller.getRVN(mc);
            UnknownTypeNode utn = UnknownTypeNode.get((jq_Reference) t);
            addObjectAllocation(utn, utn);
            addAllocType(utn, (jq_Reference) t);
            addVarType(utn, (jq_Reference) t);
            if (rvn != null) {
                addDirectAssignment(rvn, utn);
            }
        }
    }
        
    IndexMap/* Node->index */ Vmap = new IndexMap("Variable");
    IndexMap/* Node->index */ Hmap = new IndexMap("HeapObj");
    IndexMap/* jq_Field->index */ Fmap = new IndexMap("Field");
    IndexMap/* jq_Reference->index */ Tmap = new IndexMap("Class");
    IndexMap/* jq_InstanceMethod->index */ methodIndexMap = new IndexMap("MethodCall");
    IndexMap/* jq_InstanceMethod->index */ targetIndexMap = new IndexMap("MethodTarget");

    int getVariableIndex(Node dest) {
        return Vmap.get(dest);
    }
    int getHeapobjIndex(Node site) {
        return Hmap.get(site);
    }
    int getFieldIndex(jq_Field f) {
        return Fmap.get(f);
    }
    int getTypeIndex(jq_Reference f) {
        return Tmap.get(f);
    }
    int getMethodIndex(jq_InstanceMethod f) {
        return methodIndexMap.get(f);
    }
    int getTargetIndex(jq_InstanceMethod f) {
        return targetIndexMap.get(f);
    }
    Node getVariable(int index) {
        return (Node) Vmap.get(index);
    }
    Node getHeapobj(int index) {
        return (Node) Hmap.get(index);
    }
    jq_Field getField(int index) {
        return (jq_Field) Fmap.get(index);
    }
    jq_Reference getType(int index) {
        return (jq_Reference) Tmap.get(index);
    }
    jq_InstanceMethod getMethod(int index) {
        return (jq_InstanceMethod) methodIndexMap.get(index);
    }
    jq_InstanceMethod getTarget(int index) {
        return (jq_InstanceMethod) targetIndexMap.get(index);
    }

    public void addObjectAllocation(Node dest, Node site) {
        int dest_i = getVariableIndex(dest);
        int site_i = getHeapobjIndex(site);
        BDD dest_bdd = V1.ithVar(dest_i);
        BDD site_bdd = H1.ithVar(site_i);
        dest_bdd.andWith(site_bdd);
        if (TRACE_INST) {
            System.out.println("Adding object allocation site="+site_i+" dest="+dest_i);
        }
        pointsTo.orWith(dest_bdd);
        if (TRACE_BDD) {
            printSet("Points-to is now", pointsTo);
        }
        
        if (!IGNORE_THREADS && site != null) {
            jq_Reference type = (jq_Reference) site.getDeclaredType();
            if (type instanceof jq_Class) {
                type.prepare();
                PrimordialClassLoader.getJavaLangThread().prepare();
                PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;").prepare();
                if (type.isSubtypeOf(PrimordialClassLoader.getJavaLangThread()) ||
                    type.isSubtypeOf(PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;"))) {
                    System.out.println("Thread creation site found: "+site);
                    addThreadRun(site, (jq_Class) type);
                }
            }
        }
    }

    public void addDirectAssignment(Node dest, Set srcs) {
        int dest_i = getVariableIndex(dest);
        BDD dest_bdd = V2.ithVar(dest_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Node src = (Node) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1.ithVar(src_i);
            src_bdd.andWith(dest_bdd.id());
            if (TRACE_INST) {
                System.out.println("Adding direct assignment dest="+dest_i+" src="+src_i);
            }
            edgeSet.orWith(src_bdd);
            if (TRACE_BDD) {
                printSet("Edge-set is now", edgeSet);
            }
        }
        dest_bdd.free();
    }
    
    public void addDirectAssignment(Node dest, Node src) {
        int dest_i = getVariableIndex(dest);
        int src_i = getVariableIndex(src);
        BDD dest_bdd = V2.ithVar(dest_i);
        BDD src_bdd = V1.ithVar(src_i);
        dest_bdd.andWith(src_bdd);
        if (TRACE_INST) {
            System.out.println("Adding direct assignment dest="+dest_i+" src="+src_i);
        }
        edgeSet.orWith(dest_bdd);
        if (TRACE_BDD) {
            printSet("Edge-set is now", edgeSet);
        }
    }

    public void addLoadField(Node dest, Node base, jq_Field f) {
        int dest_i = getVariableIndex(dest);
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD dest_bdd = V2.ithVar(dest_i);
        BDD base_bdd = V1.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        base_bdd.andWith(f_bdd);
        dest_bdd.andWith(base_bdd);
        if (TRACE_INST) {
            System.out.println("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i);
        }
        loads.orWith(dest_bdd);
        if (TRACE_BDD) {
            printSet("Loads-set is now", loads);
        }
    }

    public void addLoadField(Set dests, Node base, jq_Field f) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V1.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=dests.iterator(); i.hasNext(); ) {
            FieldNode dest = (FieldNode) i.next();
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V2.ithVar(dest_i);
            dest_bdd.andWith(f_bdd.id());
            dest_bdd.andWith(base_bdd.id());
            if (TRACE_INST) {
                System.out.println("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i);
            }
            loads.orWith(dest_bdd);
            if (TRACE_BDD) {
                printSet("Loads-set is now", loads);
            }
        }
        base_bdd.free(); f_bdd.free();
    }
    
    public void addFieldStore(Node base, jq_Field f, Node src) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        int src_i = getVariableIndex(src);
        BDD base_bdd = V2.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        BDD src_bdd = V1.ithVar(src_i);
        f_bdd.andWith(src_bdd);
        base_bdd.andWith(f_bdd);
        if (TRACE_INST) {
            System.out.println("Adding store field base="+base_i+" f="+f_i+" src="+src_i);
        }
        stores.orWith(base_bdd);
        if (TRACE_BDD) {
            printSet("Stores-set is now", stores);
        }
    }

    public void addFieldStore(Node base, jq_Field f, Set srcs) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V2.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Node src = (Node) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1.ithVar(src_i);
            src_bdd.andWith(f_bdd.id());
            src_bdd.andWith(base_bdd.id());
            if (TRACE_INST) {
                System.out.println("Adding store field base="+base_i+" f="+f_i+" src="+src_i);
                //printSet("Adding store field base="+base_i+" f="+f_i+" src="+src_i, base_bdd, "V1xV2xFD");
            }
            stores.orWith(src_bdd);
            if (TRACE_BDD) {
                printSet("Stores-set is now", stores);
            }
        }
        base_bdd.free(); f_bdd.free();
    }
    
    public void addClassType(jq_Reference type) {
        if (type == null) return;
        if (Tmap.contains(type)) return;
        if (type instanceof jq_Class) {
            jq_Class k = (jq_Class) type;
            k.prepare();
            jq_Class[] interfaces = k.getInterfaces();
            for (int i=0; i<interfaces.length; ++i) {
                addClassType(interfaces[i]);
            }
            addClassType(k.getSuperclass());
        }
    }

    BDD aC; // H1 x T2
    BDD vC; // V1 x T1
    BDD cC; // T1 x T2

    public void addAllocType(Node site, jq_Reference type) {
        addClassType(type);
        int site_i = getHeapobjIndex(site);
        int type_i = getTypeIndex(type);
        BDD site_bdd = H1.ithVar(site_i);
        BDD type_bdd = T2.ithVar(type_i);
        type_bdd.andWith(site_bdd);
        if (TRACE) {
            printSet("Adding alloc type site="+site_i+" type="+type_i, type_bdd);
        }
        aC.orWith(type_bdd);
        if (TRACE_BDD) {
            printSet("AllocClass is now", aC);
        }
    }

    public void addVarType(Node var, jq_Reference type) {
        addClassType(type);
        int var_i = getVariableIndex(var);
        int type_i = getTypeIndex(type);
        BDD var_bdd = V1.ithVar(var_i);
        BDD type_bdd = T1.ithVar(type_i);
        type_bdd.andWith(var_bdd);
        if (TRACE) {
            printSet("Adding var type var="+var_i+" type="+type_i, type_bdd);
        }
        vC.orWith(type_bdd);
        if (TRACE_BDD) {
            printSet("VarClass is now", vC);
        }
    }
    
    int last_typeIndex;
    
    void calculateTypeHierarchy() {
        int n1=Tmap.size();
        if (TRACE) System.out.println(n1-last_typeIndex + " new types");
        for (int i1=0; i1<n1; ++i1) {
            jq_Type t1 = (jq_Type) Tmap.get(i1);
            if (t1 == null) {
                BDD type1_bdd = T1.ithVar(i1);
                BDD type2_bdd = T2.domain();
                type1_bdd.andWith(type2_bdd);
                cC.orWith(type1_bdd);
                continue;
            }
            t1.prepare();
            int i2 = (i1 < last_typeIndex) ? last_typeIndex : 0;
            for ( ; i2<n1; ++i2) {
                jq_Type t2 = (jq_Type) Tmap.get(i2);
                if (t2 == null) {
                    BDD type1_bdd = T1.domain();
                    BDD type2_bdd = T2.ithVar(i2);
                    type1_bdd.andWith(type2_bdd);
                    cC.orWith(type1_bdd);
                    continue;
                }
                t2.prepare();
                if (TypeCheck.isAssignable(t2, t1)) {
                    BDD type1_bdd = T1.ithVar(i1);
                    BDD type2_bdd = T2.ithVar(i2);
                    type1_bdd.andWith(type2_bdd);
                    cC.orWith(type1_bdd);
                }
            }
        }
        last_typeIndex = n1;
    }
    
    public static boolean NO_TYPE_FILTERING = NO_HEAP || System.getProperty("notypefilter") != null;
    
    public void calculateTypeFilter() {
        if (NO_TYPE_FILTERING) {
            typeFilter = bdd.one();
        } else {
            calculateTypeHierarchy();
        
            // (T1 x T2) * (H1 x T2) => (T1 x H1)
            BDD assignableTypes = cC.relprod(aC, T2set);
            // (T1 x H1) * (V1 x T1) => (V1 x H1)
            typeFilter = assignableTypes.relprod(vC, T1set);
            assignableTypes.free();
            cTypes = aC.replace(T2ToT1);
            //cC.free(); vC.free(); aC.free();
        }
    }
    
    BDD vtable_bdd; // H1 x T3 x T4
    
    int last_methodIndex;
    int last_heapobjIndex;
    
    public static final boolean ALL_CONCRETE = true;

    public void calculateVTables() {
        int n1 = methodIndexMap.size();
        int n2 = Hmap.size();
        if (TRACE) {
            System.out.println(n1-last_methodIndex + " new distinct virtual methods, total="+n1);
            System.out.println(n2-last_heapobjIndex + " new heap objects, total="+n2);
        }
        for (int i1=0; i1<n1; ++i1) {
            jq_InstanceMethod m = (jq_InstanceMethod) methodIndexMap.get(i1);
            BDD method_bdd = T3.ithVar(i1);
            int i2 = (i1 < last_methodIndex) ? last_heapobjIndex : 0;
            for ( ; i2<n2; ++i2) {
                Node c = (Node) Hmap.get(i2);
                if (c == null) continue;
                if (c instanceof GlobalNode) continue;
                jq_Reference r2 = (jq_Reference) c.getDeclaredType();
                if (r2 == null) continue;
                r2.prepare(); m.getDeclaringClass().prepare();
                if (TRACE) {
                    System.out.println("Comparing "+r2+" and "+m);
                }
                if (!r2.isSubtypeOf(m.getDeclaringClass())) continue;
                BDD heapobj_bdd = H1.ithVar(i2);
                if (c instanceof ConcreteTypeNode || c instanceof ConcreteObjectNode || ALL_CONCRETE) {
                    jq_InstanceMethod target = r2.getVirtualMethod(m.getNameAndDesc());
                    if (TRACE) {
                        System.out.println("Target = "+target);
                    }
                    if (target != null) {
                        int i3 = getTargetIndex(target);
                        BDD target_bdd = T4.ithVar(i3);
                        target_bdd.andWith(heapobj_bdd.id());
                        target_bdd.andWith(method_bdd.id());
                        vtable_bdd.orWith(target_bdd);
                    }
                } else {
                    CallTargets ct = CallTargets.getTargets(m.getDeclaringClass(), m, BytecodeVisitor.INVOKE_VIRTUAL, r2, false, true);
                    if (TRACE) {
                        System.out.println("Targets = "+ct);
                    }
                    for (Iterator i=ct.iterator(); i.hasNext(); ) {
                        jq_InstanceMethod target = (jq_InstanceMethod) i.next();
                        int i3 = getTargetIndex(target);
                        BDD target_bdd = T4.ithVar(i3);
                        target_bdd.andWith(heapobj_bdd.id());
                        target_bdd.andWith(method_bdd.id());
                        vtable_bdd.orWith(target_bdd);
                    }
                }
                heapobj_bdd.free();
            }
            method_bdd.free();
        }
        last_methodIndex = n1;
        last_heapobjIndex = n2;
        if (TRACE_BDD) {
            printSet("vtable", vtable_bdd);
        }
    }
    
    public void solveNonincremental() {
        BDD oldPt1;

        // start solving 
        do {
            oldPt1 = pointsTo;
            // repeat rule (1) in the inner loop
            BDD oldPt2 = bdd.zero();
            do {
                oldPt2 = pointsTo;
                /* --- rule (1) --- */
                // 
                //   l1 -> l2    o \in pt(l1)
                // --------------------------
                //          o \in pt(l2)

                // (V1 x V2) * (V1 x H1) => (V2 x H1)
                BDD newPt1 = edgeSet.relprod(pointsTo, V1set);
                // (V2 x H1) => (V1 x H1)
                BDD newPt2 = newPt1.replace(V2ToV1);

                /* --- apply type filtering and merge into pointsTo relation --- */
                // (V1 x H1)
                BDD newPt3 = newPt2.and(typeFilter);
                if (TRACE_BDD) {
                    BDD temp = newPt2.apply(typeFilter, BDDFactory.diff);
                    printSet("removed by type filter", temp);
                    temp.free();
                }
                // (V1 x H1)
                pointsTo = pointsTo.or(newPt3);
                
            } while (!oldPt2.equals(pointsTo));

            // propagate points-to set over field loads and stores
            /* --- rule (2) --- */
            //
            //   o2 \in pt(l)   l -> q.f   o1 \in pt(q)
            // -----------------------------------------
            //                  o2 \in pt(o1.f) 
            // (V1 x (V2 x FD)) * (V1 x H1) => ((V2 x FD) x H1)
            BDD tmpRel1 = stores.relprod(pointsTo, V1set);
            // ((V2 x FD) x H1) => ((V1 x FD) x H2)
            BDD tmpRel2 = tmpRel1.replace(V2ToV1).replace(H1ToH2);
            // ((V1 x FD) x H2) * (V1 x H1) => ((H1 x FD) x H2)
            fieldPt = tmpRel2.relprod(pointsTo, V1set);
            System.out.println("fieldPt = "+fieldPt.satCount(H1andFDset.and(H2.set())));

            /* --- rule (3) --- */
            //
            //   p.f -> l   o1 \in pt(p)   o2 \in pt(o1)
            // -----------------------------------------
            //                 o2 \in pt(l)
            // ((V1 x FD) x V2) * (V1 x H1) => ((H1 x FD) x V2)
            BDD tmpRel3 = loads.relprod(pointsTo, V1set);
            // ((H1 x FD) x V2) * ((H1 x FD) x H2) => (V2 x H2)
            BDD newPt4 = tmpRel3.relprod(fieldPt, H1andFDset);
            // (V2 x H2) => (V1 x H1)
            BDD newPt5 = newPt4.replace(V2ToV1).replace(H2ToH1);

            /* --- apply type filtering and merge into pointsTo relation --- */
            if (TRACE_BDD) {
                printSet("before type filter", newPt5);
            }
            BDD newPt6 = newPt5.and(typeFilter);
            if (TRACE_BDD) {
                printSet("after type filter", newPt6);
            }
            pointsTo = pointsTo.or(newPt6);

        }
        while (!oldPt1.equals(pointsTo));

    }
    
    BDD oldPointsTo;
    
    public void solveIncremental() {

        BDD newPointsTo = pointsTo.id();

        // start solving 
        for (;;) {

            // repeat rule (1) in the inner loop
            for (;;) {
                BDD newPt1 = edgeSet.relprod(newPointsTo, V1set);
                newPointsTo.free();
                BDD newPt2 = newPt1.replace(V2ToV1);
                newPt1.free();
                newPt2.applyWith(pointsTo.id(), BDDFactory.diff);
                newPt2.andWith(typeFilter.id());
                newPointsTo = newPt2;
                if (newPointsTo.isZero()) break;
                pointsTo.orWith(newPointsTo.id());
            }
            newPointsTo.free();
            newPointsTo = pointsTo.apply(oldPointsTo, BDDFactory.diff);

            // apply rule (2)
            BDD tmpRel1 = stores.relprod(newPointsTo, V1set); // time-consuming!
            // (V2xFD)xH1
            BDD tmpRel2 = tmpRel1.replace(V2ToV1);
            tmpRel1.free();
            BDD tmpRel3 = tmpRel2.replace(H1ToH2);
            tmpRel2.free();
            // (V1xFD)xH2
            tmpRel3.applyWith(storePt.id(), BDDFactory.diff);
            BDD newStorePt = tmpRel3;
            // cache storePt
            storePt.orWith(newStorePt.id()); // (V1xFD)xH2

            BDD newFieldPt = storePt.relprod(newPointsTo, V1set); // time-consuming!
            // (H1xFD)xH2
            newFieldPt.orWith(newStorePt.relprod(oldPointsTo, V1set));
            newStorePt.free();
            oldPointsTo.free();
            // (H1xFD)xH2
            newFieldPt.applyWith(fieldPt.id(), BDDFactory.diff);
            // cache fieldPt
            fieldPt.orWith(newFieldPt.id()); // (H1xFD)xH2

            // apply rule (3)
            BDD tmpRel4 = loads.relprod(newPointsTo, V1set); // time-consuming!
            newPointsTo.free();
            // (H1xFD)xV2
            BDD newLoadAss = tmpRel4.apply(loadAss, BDDFactory.diff);
            tmpRel4.free();
            BDD newLoadPt = loadAss.relprod(newFieldPt, H1andFDset);
            newFieldPt.free();
            // V2xH2
            newLoadPt.orWith(newLoadAss.relprod(fieldPt, H1andFDset));
            // V2xH2
            // cache loadAss
            loadAss.orWith(newLoadAss);

            // update oldPointsTo
            oldPointsTo = pointsTo.id();

            // convert new points-to relation to normal type
            BDD tmpRel5 = newLoadPt.replace(V2ToV1);
            newPointsTo = tmpRel5.replace(H2ToH1);
            tmpRel5.free();
            newPointsTo.applyWith(pointsTo.id(), BDDFactory.diff);

            // apply typeFilter
            newPointsTo.andWith(typeFilter.id());
            if (newPointsTo.isZero()) break;
            pointsTo.orWith(newPointsTo.id());
        }
        
        newPointsTo.free();
    }

    /**
     * @param dumpfilename
     */
    void dumpResults(String dumpfilename) throws IOException {
        
        BDD cs_pointsTo = pointsTo.and(V1c.ithVar(0));
        cs_pointsTo.andWith(H1c.ithVar(0));
        System.out.println("vP = "+(long)cs_pointsTo.satCount(V1andH1set)+" relations, "+cs_pointsTo.nodeCount()+" nodes");
        bdd.save(dumpfilename+".vP", cs_pointsTo);
        
        BDD cs_fieldPt = fieldPt.and(H1c.ithVar(0));
        cs_fieldPt.andWith(H2c.ithVar(0));
        System.out.println("hP = "+(long)cs_fieldPt.satCount(H1andFDset.and(H2.set()))+" relations, "+cs_fieldPt.nodeCount()+" nodes");
        bdd.save(dumpfilename+".hP", cs_fieldPt);
        
        BDD cs_stores = stores.and(V1c.ithVar(0));
        cs_stores.andWith(V2c.ithVar(0));
        System.out.println("S = "+(long)cs_stores.satCount(V1andV2andFDset)+" relations, "+cs_stores.nodeCount()+" nodes");
        bdd.save(dumpfilename+".S", cs_stores);
        
        BDD cs_loads = loads.and(V1c.ithVar(0));
        cs_loads.andWith(V2c.ithVar(0));
        System.out.println("L = "+(long)cs_loads.satCount(V1andV2andFDset)+" relations, "+cs_loads.nodeCount()+" nodes");
        bdd.save(dumpfilename+".L", cs_loads);
        
        DataOutputStream dos;
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
        dumpConfig(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Vmap"));
        dumpVarIndexMap(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Hmap"));
        dumpHeapIndexMap(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".Fmap"));
        dumpFieldIndexMap(dos);
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
    
    private void dumpVarIndexMap(DataOutput out) throws IOException {
        int n = Vmap.size();
        out.writeBytes(n+"\n");
        int j = 0;
        while (j < Vmap.size()) {
            Node node = getVariable(j);
            node.write(Vmap, out);
            out.writeByte('\n');
            ++j;
        }
    }

    private void dumpHeapIndexMap(DataOutput out) throws IOException {
        int n = Hmap.size();
        out.writeBytes(n+"\n");
        int j = 0;
        while (j < Hmap.size()) {
            // UnknownTypeNode
            Node node = getHeapobj(j);
            if (node == null) out.writeBytes("null");
            else node.write(Hmap, out);
            out.writeByte('\n');
            ++j;
        }
    }
    
    private void dumpFieldIndexMap(DataOutput out) throws IOException {
        int n = Fmap.size();
        out.writeBytes(n+"\n");
        int j = 0;
        while (j < Fmap.size()) {
            jq_Field f = this.getField(j);
            if (f == null) out.writeBytes("null");
            else f.writeDesc(out);
            out.writeByte('\n');
            ++j;
        }
    }
}

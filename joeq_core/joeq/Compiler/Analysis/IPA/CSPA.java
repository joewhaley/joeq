// CSPA.java, created Jun 15, 2003 10:08:38 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.math.BigInteger;
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
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDBitVector;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Compil3r.Quad.BDDPointerAnalysis;
import Compil3r.Quad.CachedCallGraph;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.LoadedCallGraph;
import Compil3r.Quad.MethodInline;
import Compil3r.Quad.Operator;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;
import Compil3r.Quad.RootedCHACallGraph;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.PassedParameter;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnValueNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnedNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ThrownExceptionNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Main.HostedVM;
import Run_Time.TypeCheck;
import Util.Assert;
import Util.Strings;
import Util.Collections.Pair;
import Util.Graphs.Navigator;
import Util.Graphs.PathNumbering;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;
import Util.Graphs.PathNumbering.Range;

/**
 * CSPA
 * 
 * @author John Whaley
 * @version $Id$
 */
public class CSPA {

    /***** FLAGS *****/

    /** Various trace flags. */
    public static final boolean TRACE_ALL = false;
    
    public static final boolean TRACE_MATCHING  = false || TRACE_ALL;
    public static final boolean TRACE_TYPES     = false || TRACE_ALL;
    public static final boolean TRACE_MAPS      = false || TRACE_ALL;
    public static final boolean TRACE_SIZES     = false || TRACE_ALL;
    public static final boolean TRACE_CALLGRAPH = false || TRACE_ALL;
    public static final boolean TRACE_EDGES     = false || TRACE_ALL;
    public static final boolean TRACE_TIMES     = false || TRACE_ALL;
    public static final boolean TRACE_VARORDER  = false || TRACE_ALL;
    public static final boolean TRACE_RELATIONS = false || TRACE_ALL;
    public static final boolean TRACE_BDD = false;
    
    public static final boolean USE_CHA     = false;
    public static final boolean DO_INLINING = false;

    public static boolean LOADED_CALLGRAPH = false;
    public static final boolean TEST_CALLGRAPH = false;
    public static final boolean DUMP_DOTGRAPH = true;
    
    public static boolean BREAK_RECURSION = false;
    
    public static final boolean CONTEXT_SENSITIVE = true;
    public static final boolean CONTEXT_SENSITIVE_HEAP = true;
    
    public static boolean NUKE_OLD_FILES = false; // if true, will ignore
                                                  // existing files and always create new ones.
    
    public static void main(String[] args) throws IOException {
        runAnalysis(args, null);
    }
    
    public static void runAnalysis(String[] args, String addToClasspath) throws IOException {
        // We use bytecode maps.
        CodeCache.AlwaysMap = true;
        HostedVM.initialize();
        
        if (addToClasspath != null)
            PrimordialClassLoader.loader.addToClasspath(addToClasspath);
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        if (args.length > 1) {
            for (Iterator i=roots.iterator(); i.hasNext(); ) {
                jq_StaticMethod sm = (jq_StaticMethod) i.next();
                if (args[1].equals(sm.getName().toString())) {
                    roots = Collections.singleton(sm);
                    break;
                }
            }
            if (args[1].equals("--nukefiles")) {
                NUKE_OLD_FILES = true;
            }
        }
        if (args.length > 2 && args[2].equals("--nukefiles")) {
            NUKE_OLD_FILES = true;
        }
        
        String callgraphfilename = System.getProperty("callgraph", "callgraph");
        
        CallGraph cg = null;
        
        if (NUKE_OLD_FILES) {
            (new File(callgraphfilename)).delete();
        }
        
        if (new File(callgraphfilename).exists()) {
            try {
                System.out.print("Loading initial call graph...");
                long time = System.currentTimeMillis();
                cg = new LoadedCallGraph("callgraph");
                time = System.currentTimeMillis() - time;
                System.out.println("done. ("+time/1000.+" seconds)");
                roots = cg.getRoots();
                LOADED_CALLGRAPH = true;
            } catch (IOException x) {
                x.printStackTrace();
            }
        }
        if (cg == null) {
            System.out.print("Setting up initial call graph...");
            long time = System.currentTimeMillis();
            BDDPointerAnalysis dis = null;
            if (USE_CHA) {
                cg = new RootedCHACallGraph();
                cg = new CachedCallGraph(cg);
                cg.setRoots(roots);
            } else {
                dis = new BDDPointerAnalysis("java", 1000000, 100000);
                cg = dis.goIncremental(roots);
                cg = new CachedCallGraph(cg);
                // BDD pointer analysis changes the root set by adding class initializers,
                // thread entry points, etc.
                roots = cg.getRoots();
            }
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
        
            System.out.print("Calculating reachable methods...");
            time = System.currentTimeMillis();
            /* Calculate the reachable methods once to touch each method,
               so that the set of types are stable. */
            Set methods = cg.calculateReachableMethods(roots);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+methods.size()+" methods, "+time/1000.+" seconds)");
            Assert._assert(roots.equals(cg.getRoots()));
            
            try {
                java.io.FileWriter fw = new java.io.FileWriter("callgraph");
                java.io.PrintWriter pw = new java.io.PrintWriter(fw);
                LoadedCallGraph.write(cg, pw);
                pw.close();
            } catch (java.io.IOException x) {
                x.printStackTrace();
            }
            
            // reload the stored call graph because we like our program locations w.r.t. bytecode indices.
            System.out.print("Loading initial call graph...");
            time = System.currentTimeMillis();
            cg = new LoadedCallGraph("callgraph");
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
            roots = cg.getRoots();
            LOADED_CALLGRAPH = true;
            if (dis != null) dis.done();
            System.gc();
            long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.out.println("Used memory: "+usedMemory);
        }
        
        if (DO_INLINING) {
            System.out.print("Doing inlining on call graph...");
            CachedCallGraph ccg;
            if (cg instanceof CachedCallGraph)
                ccg = (CachedCallGraph) cg;
            else
                ccg = new CachedCallGraph(cg);
            long time = System.currentTimeMillis();
            // pre-initialize all classes so that we can inline more.
            for (Iterator i=ccg.getAllMethods().iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method) i.next();
                m.getDeclaringClass().cls_initialize();
            }
            MethodInline mi = new MethodInline(ccg);
            Navigator navigator = ccg.getNavigator();
            for (Iterator i=Traversals.postOrder(navigator, roots).iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method) i.next();
                if (m.getBytecode() == null) continue;
                ControlFlowGraph cfg = CodeCache.getCode(m);
                //MethodSummary ms = MethodSummary.getSummary(cfg);
                mi.visitCFG(cfg);
            }
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
            
            System.out.print("Rebuilding call graph...");
            time = System.currentTimeMillis();
            ccg.invalidateCache();
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
            cg = ccg;
        }

        long time;
        
        if (TEST_CALLGRAPH) {
            RootedCHACallGraph.test(cg);
        }
        
        System.out.print("Counting size of call graph...");
        time = System.currentTimeMillis();
        PathNumbering pn = countCallGraph(cg);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        if (DUMP_DOTGRAPH) {
            try {
                DataOutputStream dos = new DataOutputStream(new FileOutputStream("callgraph.dot"));
                pn.dotGraph(dos);
            } catch (IOException x) {
                x.printStackTrace();
            }
        }
        
        // Allocate CSPA object.
        CSPA dis = new CSPA(cg);
        dis.roots = roots;
        dis.pn = pn;
        
        // Initialize BDD package.
        dis.initializeBDD(DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
        
        // Add edges for existing globals.
        dis.addGlobals();
        
        System.out.print("Generating BDD summaries without context...");
        time = System.currentTimeMillis();
        dis.generateBDDSummaries();
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        System.out.print("Initializing relations and adding call graph edges...");
        time = System.currentTimeMillis();
        dis.goForIt();
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        System.out.print("Solving pointers...");
        time = System.currentTimeMillis();
        dis.solveIncremental();
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        dis.printHistogram();
        dis.escapeAnalysis();
        
        String dumpfilename = System.getProperty("cspa.dumpfile", "cspa");
        dis.dumpResults(dumpfilename);
    }
    
    /**
     * @param dumpfilename
     */
    void dumpResults(String dumpfilename) throws IOException {
        bdd.save(dumpfilename+".bdd", g_pointsTo);
        
        DataOutputStream dos;
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".config"));
        dumpConfig(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".vars"));
        dumpVarIndexMap(dos);
        dos.close();
        dos = new DataOutputStream(new FileOutputStream(dumpfilename+".heap"));
        dumpHeapIndexMap(dos);
        dos.close();
    }

    private void dumpConfig(DataOutput out) throws IOException {
        out.writeBytes(VARBITS+" "+HEAPBITS+" "+FIELDBITS+" "+CLASSBITS+" "+CONTEXTBITS+"\n");
        String ordering = System.getProperty("bddordering", "FD_H2cxH2o_V2cxV1cxV2oxV1o_H1cxH1o");
        out.writeBytes(ordering+"\n");
    }

    public static interface Variable {
        void write(DataOutput out) throws IOException;
    }
    public static interface HeapObject {
        void write(DataOutput out) throws IOException;
    }

    private void dumpVarIndexMap(DataOutput out) throws IOException {
        int n = variableIndexMap.size();
        out.writeBytes(n+"\n");
        int j;
        for (j=0; j<=globalVarHighIndex; ++j) {
            Variable node = getVariable(j); 
            node.write(out);
            out.writeByte('\n');
        }
        for (Iterator i=bddSummaryList.iterator(); i.hasNext(); ) {
            BDDMethodSummary s = (BDDMethodSummary) i.next();
            Assert._assert(s.lowVarIndex == j);
            for ( ; j<=s.highVarIndex; ++j) {
                Variable node = getVariable(j);
                node.write(out);
                out.writeByte('\n');
            }
        }
        while (j < variableIndexMap.size()) {
            // UnknownTypeNode
            Variable node = getVariable(j);
            node.write(out);
            out.writeByte('\n');
            ++j;
        }
    }

    private void dumpHeapIndexMap(DataOutput out) throws IOException {
        int n = heapobjIndexMap.size();
        out.writeBytes(n+"\n");
        int j;
        for (j=0; j<=globalHeapHighIndex; ++j) {
            // ConcreteObjectNode
            HeapObject node = getHeapObject(j);
            if (node == null) out.writeBytes("null");
            else node.write(out);
            out.writeByte('\n');
        }
        for (Iterator i=bddSummaryList.iterator(); i.hasNext(); ) {
            BDDMethodSummary s = (BDDMethodSummary) i.next();
            Assert._assert(s.lowHeapIndex == j);
            for ( ; j<=s.highHeapIndex; ++j) {
                HeapObject node = getHeapObject(j);
                node.write(out);
                out.writeByte('\n');
            }
        }
        while (j < heapobjIndexMap.size()) {
            // UnknownTypeNode
            HeapObject node = getHeapObject(j);
            node.write(out);
            out.writeByte('\n');
            ++j;
        }
    }
    
    void printHistogram() {
        BDD pointsTo = g_pointsTo.exist(V1c.set());
        int[] histogram = new int[64];
        for (int i=0; i<variableIndexMap.size(); i++) {
            BDD a = pointsTo.restrict(V1o.ithVar(i));
            BDD b = a.exist(H1c.set());
            long size = (long) b.satCount(H1o.set());
            int index;
            if (size >= histogram.length) index = histogram.length - 1;
            else index = (int) size;
            histogram[index]++;
            //System.out.println(variableIndexMap.get(i)+" points to "+size+" objects");
        }
        for (int i=0; i<histogram.length; ++i) {
            if (histogram[i] != 0) {
                if (i==histogram.length-1) System.out.print(">=");
                System.out.println(i+" = "+histogram[i]);
            }
        }
    }
    
    int globalVarLowIndex, globalVarHighIndex;
    int globalHeapLowIndex, globalHeapHighIndex;
    
    public void addGlobals() {
        Assert._assert(variableIndexMap.size() == 0);
        globalVarLowIndex = 0; globalHeapLowIndex = 0;
        GlobalNode.GLOBAL.addDefaultStatics();
        addGlobalObjectAllocation(GlobalNode.GLOBAL, null);
        addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        handleGlobalNode(GlobalNode.GLOBAL);
        for (Iterator i=ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            handleGlobalNode((ConcreteObjectNode) i.next());
        }
        globalVarHighIndex = variableIndexMap.size() - 1;
        globalHeapHighIndex = heapobjIndexMap.size() - 1;
    }
    
    public void addGlobalV1V2Context(BDD b) {
        if (CONTEXT_SENSITIVE) {
            BDD r = V1c.domain();
            r.andWith(V2c.domain());
            b.andWith(r);
        } else {
            BDD r = V1c.ithVar(0);
            r.andWith(V2c.ithVar(0));
            b.andWith(r);
        }
    }
    public void addGlobalV1H1Context(BDD b) {
        if (CONTEXT_SENSITIVE) {
            BDD r = V1c.domain();
            r.andWith(H1c.domain());
            b.andWith(r);
        } else {
            BDD r = V1c.ithVar(0);
            r.andWith(H1c.ithVar(0));
            b.andWith(r);
        }
    }
    public BDD getV1H1Context(BDD range) {
        if (CONTEXT_SENSITIVE_HEAP) {
            BDD r = V1c.buildEquals(H1c);
            if (TRACE_SIZES) {
                System.out.print("V1cH1c = ");
                report(r, V1c.set().and(H1c.set()));
            }
            r.andWith(range.id());
            if (TRACE_SIZES) {
                System.out.print("after mask = ");
                report(r, V1c.set().and(H1c.set()));
            }
            return r;
        } else {
            BDD r = V1c.ithVar(0);
            r.andWith(H1c.ithVar(0));
            return r;
        }
    }
    public BDD getV1V2Context(BDD range) {
        if (CONTEXT_SENSITIVE) {
            BDD r = V1c.buildEquals(V2c);
            if (TRACE_SIZES) {
                System.out.print("V1cH1c = ");
                report(r, V1c.set().and(H1c.set()));
            }
            r.andWith(range.id());
            if (TRACE_SIZES) {
                System.out.print("after mask = ");
                report(r, V1c.set().and(H1c.set()));
            }
            return r;
        } else {
            BDD r = V1c.ithVar(0);
            r.andWith(V2c.ithVar(0));
            return r;
        }
    }
    
    public void addGlobalObjectAllocation(Variable dest, HeapObject site) {
        int dest_i = getVariableIndex(dest);
        int site_i = getHeapObjectIndex(site);
        BDD dest_bdd = V1o.ithVar(dest_i);
        BDD site_bdd = H1o.ithVar(site_i);
        dest_bdd.andWith(site_bdd);
        addGlobalV1H1Context(dest_bdd);
        g_pointsTo.orWith(dest_bdd);
    }
    
    public void addGlobalLoad(Set dests, Variable base, jq_Field f) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V1o.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=dests.iterator(); i.hasNext(); ) {
            // FieldNode
            Variable dest = (Variable) i.next();
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V2o.ithVar(dest_i);
            dest_bdd.andWith(f_bdd.id());
            dest_bdd.andWith(base_bdd.id());
            addGlobalV1V2Context(dest_bdd);
            g_loads.orWith(dest_bdd);
        }
        base_bdd.free(); f_bdd.free();
    }
    
    public void addGlobalStore(Variable base, jq_Field f, Set srcs) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V2o.ithVar(base_i);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Variable src = (Variable) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            src_bdd.andWith(f_bdd.id());
            src_bdd.andWith(base_bdd.id());
            addGlobalV1V2Context(src_bdd);
            g_stores.orWith(src_bdd);
        }
        base_bdd.free(); f_bdd.free();
    }
    
    // v2 = v1;
    public void addGlobalEdge(Variable dest, Collection srcs) {
        int dest_i = getVariableIndex(dest);
        BDD dest_bdd = V2o.ithVar(dest_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Variable src = (Variable) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            src_bdd.andWith(dest_bdd.id());
            addGlobalV1V2Context(src_bdd);
            g_edgeSet.orWith(src_bdd);
        }
        dest_bdd.free();
    }
    
    public void handleGlobalNode(Node n) {
        
        Iterator j;
        j = n.getEdges().iterator();
        while (j.hasNext()) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Object o = e.getValue();
            // n.f = o
            if (o instanceof Set) {
                addGlobalStore(n, f, (Set) o);
            } else {
                addGlobalStore(n, f, Collections.singleton(o));
            }
        }
        j = n.getAccessPathEdges().iterator();
        while (j.hasNext()) {
            Map.Entry e = (Map.Entry)j.next();
            jq_Field f = (jq_Field)e.getKey();
            Object o = e.getValue();
            // o = n.f
            if (o instanceof Set) {
                addGlobalLoad((Set) o, n, f);
            } else {
                addGlobalLoad(Collections.singleton(o), n, f);
            }
        }
        if (n instanceof HeapObject) {
            HeapObject ho = (HeapObject) n;
            addGlobalObjectAllocation(n, ho);
            addAllocType(ho, (jq_Reference) n.getDeclaredType());
        }
        if (n instanceof GlobalNode) {
            addGlobalEdge(GlobalNode.GLOBAL, Collections.singleton(n));
            addGlobalEdge(n, Collections.singleton(GlobalNode.GLOBAL));
            addVarType(n, PrimordialClassLoader.getJavaLangObject());
        } else {
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }
    }
    
    /**
     * The default initial node count.  Smaller values save memory for
     * smaller problems, larger values save the time to grow the node tables
     * on larger problems.
     */
    public static final int DEFAULT_NODE_COUNT = Integer.parseInt(System.getProperty("bddnodes", "1000000"));

    /**
     * The size of the BDD operator cache.
     */
    public static final int DEFAULT_CACHE_SIZE = Integer.parseInt(System.getProperty("bddcache", "100000"));

    /**
     * Singleton BDD object that provides access to BDD functions.
     */
    private BDDFactory bdd;
    
    public static int VARBITS = 18;
    public static int HEAPBITS = 15;
    public static int FIELDBITS = 14;
    public static int CLASSBITS = 14;
    public static int CONTEXTBITS = 38;
    
    // the size of domains, can be changed to reflect the size of inputs
    int domainBits[];
    // to be computed in sysInit function
    int domainSpos[]; 
    
    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1o, V2o, H1o, H2o;
    BDDDomain V1c, V2c, H1c, H2c;
    BDDDomain FD;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2; 
    BDDDomain[] bdd_domains;

    // domain pairs for bdd_replace
    BDDPairing V1ToV2;
    BDDPairing V2ToV1;
    BDDPairing H1ToH2;
    BDDPairing H2ToH1;
    BDDPairing T2ToT1;
    
    // domain sets
    BDD V1set, V2set, V3set, FDset, H1set, H2set, T1set, T2set;
    BDD H1andFDset;

    // global BDDs
    BDD aC; // H1 x T2
    BDD vC; // V1 x T1
    BDD cC; // T1 x T2
    BDD typeFilter; // V1 x H1

    PathNumbering pn;
    
    public static PathNumbering countCallGraph(CallGraph cg) {
        Set fields = new HashSet();
        Set classes = new HashSet();
        int vars = 0, heaps = 0, bcodes = 0, methods = 0, calls = 0;
        for (Iterator i=cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            ++methods;
            if (m.getBytecode() == null) continue;
            bcodes += m.getBytecode().length;
            ControlFlowGraph cfg = CodeCache.getCode(m);
            MethodSummary ms = MethodSummary.getSummary(cfg);
            for (Iterator j=ms.nodeIterator(); j.hasNext(); ) {
                Node n = (Node) j.next();
                ++vars;
                if (n instanceof ConcreteTypeNode ||
                    n instanceof UnknownTypeNode ||
                    n instanceof ConcreteObjectNode)
                    ++heaps;
                fields.addAll(n.getAccessPathEdgeFields());
                fields.addAll(n.getEdgeFields());
                if (n instanceof GlobalNode) continue;
                jq_Reference r = (jq_Reference) n.getDeclaredType();
                classes.add(r);
            }
            calls += ms.getCalls().size();
        }
        System.out.println();
        System.out.println("Methods="+methods+" Bytecodes="+bcodes+" Call sites="+calls);
        PathNumbering pn = new PathNumbering();
        BigInteger paths = (BigInteger) pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator());
        System.out.println("Vars="+vars+" Heaps="+heaps+" Classes="+classes.size()+" Fields="+fields.size()+" Paths="+paths);
        double log2 = Math.log(2);
        VARBITS = (int) (Math.log(vars+256)/log2 + 1.0);
        HEAPBITS = (int) (Math.log(heaps+256)/log2 + 1.0);
        FIELDBITS = (int) (Math.log(fields.size()+64)/log2 + 2.0);
        CLASSBITS = (int) (Math.log(classes.size()+64)/log2 + 2.0);
        CONTEXTBITS = paths.bitLength();
        CONTEXTBITS = Math.min(60, CONTEXTBITS);
        System.out.println("Var bits="+VARBITS+" Heap bits="+HEAPBITS+" Class bits="+CLASSBITS+" Field bits="+FIELDBITS+" Context bits="+CONTEXTBITS);
        return pn;
    }

    public CSPA(CallGraph cg) {
        this.cg = cg;
    }
    
    CallGraph cg;
    Collection roots;
    
    public void initializeBDD(int nodeCount, int cacheSize) {
        bdd = BDDFactory.init(nodeCount, cacheSize);
        
        bdd.setCacheRatio(8);
        bdd.setMaxIncrease(Math.min(nodeCount/4, 2500000));
        bdd.setMaxNodeNum(0);
        
        variableIndexMap = new IndexMap("Variable", 1 << VARBITS);
        heapobjIndexMap = new IndexMap("HeapObj", 1 << HEAPBITS);
        fieldIndexMap = new IndexMap("Field", 1 << FIELDBITS);
        typeIndexMap = new IndexMap("Class", 1 << CLASSBITS);

        domainBits = new int[] {VARBITS, CONTEXTBITS,
                                VARBITS, CONTEXTBITS,
                                FIELDBITS,
                                HEAPBITS, CONTEXTBITS,
                                HEAPBITS, CONTEXTBITS};
        domainSpos = new int[domainBits.length];
        
        long[] domains = new long[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1L << domainBits[i]);
        }
        bdd_domains = bdd.extDomain(domains);
        V1o = bdd_domains[0];
        V1c = bdd_domains[1];
        V2o = bdd_domains[2];
        V2c = bdd_domains[3];
        FD = bdd_domains[4];
        H1o = bdd_domains[5];
        H1c = bdd_domains[6];
        H2o = bdd_domains[7];
        H2c = bdd_domains[8];
        T1 = V2o;
        T2 = V1o;
        for (int i=0; i<domainBits.length; ++i) {
            Assert._assert(bdd_domains[i].varNum() == domainBits[i]);
        }
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "FD_H2cxH2o_V2cxV1cxV2oxV1o_H1cxH1o");
        
        int[] varorder = makeVarOrdering(bdd, domainBits, domainSpos, reverseLocal, ordering);
        if (TRACE_VARORDER) {
            for (int i=0; i<varorder.length; ++i) {
                if (i != 0) 
                    System.out.print(",");
                System.out.print(varorder[i]);
            }
            System.out.println();
        }
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
        
        V1ToV2 = bdd.makePair();
        V1ToV2.set(new BDDDomain[] {V1o, V1c},
                   new BDDDomain[] {V2o, V2c});
        V2ToV1 = bdd.makePair();
        V2ToV1.set(new BDDDomain[] {V2o, V2c},
                   new BDDDomain[] {V1o, V1c});
        H1ToH2 = bdd.makePair();
        H1ToH2.set(new BDDDomain[] {H1o, H1c},
                   new BDDDomain[] {H2o, H2c});
        H2ToH1 = bdd.makePair();
        H2ToH1.set(new BDDDomain[] {H2o, H2c},
                   new BDDDomain[] {H1o, H1c});
        T2ToT1 = bdd.makePair(T2, T1);
        
        V1set = V1o.set();
        //if (CONTEXT_SENSITIVE)
            V1set.andWith(V1c.set());
        V2set = V2o.set();
        //if (CONTEXT_SENSITIVE)
            V2set.andWith(V2c.set());
        FDset = FD.set();
        H1set = H1o.set();
        //if (CONTEXT_SENSITIVE)
            H1set.andWith(H1c.set());
        H2set = H2o.set();
        //if (CONTEXT_SENSITIVE)
            H2set.andWith(H2c.set());
        T1set = T1.set();
        T2set = T2.set();
        H1andFDset = H1set.and(FDset);
        
        reset();
    }

    void reset() {
        aC = bdd.zero();
        vC = bdd.zero();
        cC = bdd.zero();
        typeFilter = bdd.zero();
        g_pointsTo = bdd.zero();
        g_loads = bdd.zero();
        g_stores = bdd.zero();
        g_edgeSet = bdd.zero();
    }

    static int[] makeVarOrdering(BDDFactory bdd, int[] domainBits, int[] domainSpos,
                                 boolean reverseLocal, String ordering) {
        
        int varnum = bdd.varNum();
        
        int[][] localOrders = new int[domainBits.length][];
        for (int i=0; i<localOrders.length; ++i) {
            localOrders[i] = new int[domainBits[i]];
        }
        
        for (int i=0, pos=0; i<domainBits.length; ++i) {
            domainSpos[i] = pos;
            pos += domainBits[i];
            for (int j=0; j<domainBits[i]; ++j) {
                if (reverseLocal) {
                    localOrders[i][j] = domainBits[i] - j - 1;
                } else {
                    localOrders[i][j] = j;
                }
            }
        }
        
        BDDDomain[] doms = new BDDDomain[domainBits.length];
        
        int[] varorder = new int[varnum];
        
        System.out.println("Ordering: "+ordering);
        StringTokenizer st = new StringTokenizer(ordering, "x_", true);
        int numberOfDomains = 0, bitIndex = 0;
        for (int i=0; ; ++i) {
            String s = st.nextToken();
            BDDDomain d;
            if (s.equals("V1o")) d = bdd.getDomain(0);
            else if (s.equals("V1c")) d = bdd.getDomain(1);
            else if (s.equals("V2o")) d = bdd.getDomain(2);
            else if (s.equals("V2c")) d = bdd.getDomain(3);
            else if (s.equals("FD")) d = bdd.getDomain(4);
            else if (s.equals("H1o")) d = bdd.getDomain(5);
            else if (s.equals("H1c")) d = bdd.getDomain(6);
            else if (s.equals("H2o")) d = bdd.getDomain(7);
            else if (s.equals("H2c")) d = bdd.getDomain(8);
            else {
                Assert.UNREACHABLE("bad domain: "+s);
                return null;
            }
            doms[i] = d;
            if (st.hasMoreTokens()) {
                s = st.nextToken();
                if (s.equals("x")) {
                    ++numberOfDomains;
                    continue;
                }
            }
            bitIndex = fillInVarIndices(domainBits, domainSpos,
                                        doms, i-numberOfDomains, numberOfDomains+1,
                                        localOrders, bitIndex, varorder);
            if (!st.hasMoreTokens()) {
                //Collection not_done = new ArrayList(Arrays.asList(bdd_domains));
                //not_done.removeAll(Arrays.asList(doms));
                //Assert._assert(not_done.isEmpty(), not_done.toString());
                break;
            }
            if (s.equals("_")) {
                numberOfDomains = 0;
            } else {
                Assert.UNREACHABLE("bad token: "+s);
                return null;
            }
        }
        
        for (int i=0; i<doms.length; ++i) {
            doms[i] = bdd.getDomain(i);
        }
        int[] outside2inside = new int[varnum];
        getVariableMap(outside2inside, doms);
        
        remapping(varorder, outside2inside);
        
        return varorder;
    }
    
    static int fillInVarIndices(int[] domainBits, int[] domainSpos,
                         BDDDomain[] doms, int domainIndex, int numDomains,
                         int[][] localOrders, int bitIndex, int[] varorder) {
        int maxBits = 0;
        for (int i=0; i<numDomains; ++i) {
            BDDDomain d = doms[domainIndex+i];
            int di = d.getIndex();
            maxBits = Math.max(maxBits, domainBits[di]);
        }
        for (int bitNumber=0; bitNumber<maxBits; ++bitNumber) {
            for (int i=0; i<numDomains; ++i) {
                BDDDomain d = doms[domainIndex+i];
                int di = d.getIndex();
                if (bitNumber < domainBits[di]) {
                    varorder[bitIndex++] = domainSpos[di] + localOrders[di][bitNumber];
                }
            }
        }
        return bitIndex;
    }
    
    static void getVariableMap(int[] map, BDDDomain[] doms) {
        int idx = 0;
        for (int var = 0; var < doms.length; var++) {
            int[] vars = doms[var].vars();
            for (int i = 0; i < vars.length; i++) {
                map[idx++] = vars[i];
            }
        }
    }
    
    /* remap according to a map */
    static void remapping(int[] varorder, int[] maps) {
        int[] varorder2 = new int[varorder.length];
        for (int i = 0; i < varorder.length; i++) {
            varorder2[i] = maps[varorder[i]];
        }
        System.arraycopy(varorder2, 0, varorder, 0, varorder.length);
    }
    
    IndexMap/* Variable->index */ variableIndexMap;
    IndexMap/* HeapObject->index */ heapobjIndexMap;
    IndexMap/* jq_Field->index */ fieldIndexMap;
    IndexMap/* jq_Reference->index */ typeIndexMap;

    int getVariableIndex(Variable dest) {
        return variableIndexMap.get(dest);
    }
    int getHeapObjectIndex(HeapObject site) {
        return heapobjIndexMap.get(site);
    }
    int getFieldIndex(jq_Field f) {
        return fieldIndexMap.get(f);
    }
    int getTypeIndex(jq_Reference f) {
        return typeIndexMap.get(f);
    }
    Variable getVariable(int index) {
        return (Variable) variableIndexMap.get(index);
    }
    HeapObject getHeapObject(int index) {
        return (HeapObject) heapobjIndexMap.get(index);
    }
    jq_Field getField(int index) {
        return (jq_Field) fieldIndexMap.get(index);
    }
    jq_Reference getType(int index) {
        return (jq_Reference) typeIndexMap.get(index);
    }

    public void addClassType(jq_Reference type) {
        if (type == null) return;
        if (typeIndexMap.contains(type)) return;
        int type_i = getTypeIndex(type);
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

    public void addAllocType(HeapObject site, jq_Reference type) {
        addClassType(type);
        int site_i = getHeapObjectIndex(site);
        int type_i = getTypeIndex(type);
        BDD site_bdd = H1o.ithVar(site_i);
        BDD type_bdd = T2.ithVar(type_i);
        type_bdd.andWith(site_bdd);
        if (TRACE_TYPES) System.out.println("Adding alloc type: "+type_bdd.toStringWithDomains());
        aC.orWith(type_bdd);
    }

    public void addVarType(Variable var, jq_Reference type) {
        addClassType(type);
        int var_i = getVariableIndex(var);
        int type_i = getTypeIndex(type);
        BDD var_bdd = V1o.ithVar(var_i);
        BDD type_bdd = T1.ithVar(type_i);
        type_bdd.andWith(var_bdd);
        if (TRACE_TYPES) System.out.println("Adding var type: "+type_bdd.toStringWithDomains());
        vC.orWith(type_bdd);
    }
    
    int last_typeIndex;
    
    void calculateTypeHierarchy() {
        int n1=typeIndexMap.size();
        if (TRACE_TYPES) System.out.println(n1-last_typeIndex + " new types");
        for (int i1=0; i1<n1; ++i1) {
            jq_Type t1 = (jq_Type) typeIndexMap.get(i1);
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
                jq_Type t2 = (jq_Type) typeIndexMap.get(i2);
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
    
    public void calculateTypeFilter() {
        calculateTypeHierarchy();
        
        // (T1 x T2) * (H1 x T2) => (T1 x H1)
        BDD assignableTypes = cC.relprod(aC, T2set);
        // (T1 x H1) * (V1 x T1) => (V1 x H1)
        typeFilter = assignableTypes.relprod(vC, T1set);
        assignableTypes.free();
        //cC.free(); vC.free(); aC.free();

        if (false) typeFilter = bdd.one();
    }

    BDDMethodSummary getOrCreateBDDSummary(jq_Method m) {
        if (m.getBytecode() == null) return null;
        ControlFlowGraph cfg = CodeCache.getCode(m);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        BDDMethodSummary bms = getBDDSummary(ms);
        if (bms == null) {
            bddSummaries.put(ms, bms = new BDDMethodSummary(ms));
            bddSummaryList.add(bms);
        }
        return bms;
    }
    
    Map bddSummaries = new HashMap();
    List bddSummaryList = new LinkedList();
    BDDMethodSummary getBDDSummary(MethodSummary ms) {
        BDDMethodSummary result = (BDDMethodSummary) bddSummaries.get(ms);
        return result;
    }
    
    BDDMethodSummary getBDDSummary(jq_Method m) {
        if (m.getBytecode() == null) return null;
        ControlFlowGraph cfg = CodeCache.getCode(m);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        return getBDDSummary(ms);
    }
    
    public ProgramLocation mapCall(ProgramLocation callSite) {
        if (LOADED_CALLGRAPH && callSite instanceof ProgramLocation.QuadProgramLocation) {
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
    
    public Collection getTargetMethods(ProgramLocation callSite) {
        return cg.getTargetMethods(mapCall(callSite));
    }

    BDD g_pointsTo;
    BDD g_edgeSet;
    BDD g_stores;
    BDD g_loads;

    static boolean USE_REPLACE_V2 = false;
    static boolean USE_REPLACE_H1 = false;
    static BDDPairing H1cToV2c, V2cToH1c;

    public void addRelations(MethodSummary ms) {
        BDDMethodSummary bms = this.getBDDSummary(ms);

        if (TRACE_RELATIONS)
            System.out.println("Adding relations for "+ms.getMethod());
        Assert._assert(bms != null, ms.getMethod().toString());
        
        Number npaths = pn.numberOfPathsTo(ms.getMethod());
        long time = System.currentTimeMillis();
        BDD v1ch1c = null, v1cv2c = null;
        
        BDD range = V1c.varRange(0, npaths.longValue());
        if (USE_REPLACE_V2 && !bms.m_pointsTo.isZero()) {
            v1ch1c = getV1H1Context(range);
            if (H1cToV2c == null) H1cToV2c = bdd.makePair(H1c, V2c);
            v1cv2c = v1ch1c.replace(H1cToV2c);
        } else if (USE_REPLACE_H1) {
            v1cv2c = getV1V2Context(range);
            if (V2cToH1c == null) V2cToH1c = bdd.makePair(V2c, H1c);
            v1ch1c = v1cv2c.replace(V2cToH1c);
        } else {
            if (!bms.m_pointsTo.isZero())
                v1ch1c = getV1H1Context(range);
            v1cv2c = getV1V2Context(range);
        }
        range.free();
        
        time = System.currentTimeMillis() - time;
        if (TRACE_TIMES || time > 500)
            System.out.println("Building context BDD: "+(time/1000.));
        
        time = System.currentTimeMillis();
        
        BDD t1;
        if (!bms.m_pointsTo.isZero()) {
            t1 = bms.m_pointsTo.id();
            t1.andWith(v1ch1c);
            if (TRACE_BDD) {
                System.out.println("Adding to g_pointsTo: "+t1.toStringWithDomains());
            }
            g_pointsTo.orWith(t1);
        }

        t1 = bms.m_loads.id();
        t1.andWith(v1cv2c.id());
        if (TRACE_BDD) {
            System.out.println("Adding to g_loads: "+t1.toStringWithDomains());
        }
        g_loads.orWith(t1);
        
        t1 = bms.m_stores.id();
        t1.andWith(v1cv2c);
        if (TRACE_BDD) {
            System.out.println("Adding to g_stores: "+t1.toStringWithDomains());
        }
        g_stores.orWith(t1);
        
        time = System.currentTimeMillis() - time;
        if (TRACE_TIMES || time > 500)
            System.out.println("Adding relations to global: "+(time/1000.));
        
        bms.dispose();
    }
    
    public void bindCallEdges(MethodSummary caller) {
        if (TRACE_CALLGRAPH) System.out.println("Adding call graph edges for "+caller.getMethod());
        for (Iterator i=caller.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            for (Iterator j=getTargetMethods(mc).iterator(); j.hasNext(); ) {
                jq_Method target = (jq_Method) j.next();
                if (target.getBytecode() == null) {
                    bindParameters_native(caller, mc);
                    continue;
                }
                ControlFlowGraph cfg = CodeCache.getCode(target);
                MethodSummary callee = MethodSummary.getSummary(cfg);
                bindParameters(caller, mc, callee);
            }
        }
    }
    
    public void bindParameters_native(MethodSummary caller, ProgramLocation mc) {
        // only handle return value for now.
        Object t = caller.getMethod().getReturnType();
        if (t instanceof jq_Reference) {
            ReturnValueNode rvn = caller.getRVN(mc);
            if (rvn != null) {
                jq_Reference r = (jq_Reference) t;
                UnknownTypeNode utn = UnknownTypeNode.get(r);
                addGlobalObjectAllocation(utn, utn);
                addAllocType(utn, r);
                addVarType(utn, r);
                addGlobalEdge(rvn, Collections.singleton(utn));
            }
        }
        ThrownExceptionNode ten = caller.getTEN(mc);
        if (ten != null) {
            jq_Reference r = PrimordialClassLoader.getJavaLangThrowable();
            UnknownTypeNode utn = UnknownTypeNode.get(r);
            addGlobalObjectAllocation(utn, utn);
            addAllocType(utn, r);
            addVarType(utn, r);
            addGlobalEdge(ten, Collections.singleton(utn));
        }
    }
    
    public void bindParameters(MethodSummary caller, ProgramLocation mc, MethodSummary callee) {
        if (TRACE_CALLGRAPH)
            System.out.println("Adding call graph edge "+caller.getMethod()+"->"+callee.getMethod());
        BDDMethodSummary caller_s = this.getBDDSummary(caller);
        BDDMethodSummary callee_s = this.getBDDSummary(callee);
        Pair p = new Pair(mapCall(mc), callee.getMethod());
        Range r_edge = pn.getEdge(p);
        Range r_caller = pn.getRange(caller.getMethod());
        //if (backEdges.contains(p))
        //    System.out.println("Back edge: "+p+"="+r);
        if (TRACE_CALLGRAPH)
            System.out.println("Caller context range "+r_caller+" matches callee context range "+r_edge);
        Assert._assert(r_caller.low.intValue() == 0);
        BDD context_map;
        // for parameters: V1 in caller matches V2 in callee
        context_map = buildVarContextMap(PathNumbering.toBigInt(r_caller.low),
                                         PathNumbering.toBigInt(r_caller.high),
                                         PathNumbering.toBigInt(r_edge.low),
                                         PathNumbering.toBigInt(r_edge.high));
        jq_Type[] paramTypes = mc.getParamTypes();
        for (int i=0; i<paramTypes.length; ++i) {
            if (i >= callee.getNumOfParams()) break;
            ParamNode pn = callee.getParamNode(i);
            if (pn == null) continue;
            PassedParameter pp = new PassedParameter(mc, i);
            Set s = caller.getNodesThatCall(pp);
            if (TRACE_EDGES) System.out.println("Adding edges for "+pn);
            addEdge(context_map, pn, s);
        }
        context_map.free();
        ReturnValueNode rvn = caller.getRVN(mc);
        ThrownExceptionNode ten = caller.getTEN(mc);
        if (rvn == null && ten == null) return;
        // for returns: V1 in callee matches V2 in caller
        context_map = buildVarContextMap(PathNumbering.toBigInt(r_edge.low),
                                         PathNumbering.toBigInt(r_edge.high),
                                         PathNumbering.toBigInt(r_caller.low),
                                         PathNumbering.toBigInt(r_caller.high));
        if (rvn != null) {
            Set s = callee.getReturned();
            if (TRACE_EDGES) System.out.println("Adding edges for "+rvn);
            addEdge(context_map, rvn, s);
        }
        if (ten != null) {
            Set s = callee.getThrown();
            if (TRACE_EDGES) System.out.println("Adding edges for "+ten);
            addEdge(context_map, ten, s);
        }
    }

    public void dumpGlobalSets() {
        System.out.print("g_pointsTo="+g_pointsTo.nodeCount());
        System.out.print(", g_edgeSet="+g_edgeSet.nodeCount());
        System.out.print(", g_loads="+g_loads.nodeCount());
        System.out.println(", g_stores="+g_stores.nodeCount());
    }
    
    public void dumpContextInsensitive() {
        BDD t = g_pointsTo.exist(V1c.set().and(H1c.set()));
        System.out.print("pointsTo (context-insensitive) = ");
        report(t, V1o.set().and(H1o.set()));
        t.free();
    }
    
    public void dumpGlobalSizes() {
        System.out.print("g_pointsTo = ");
        report(g_pointsTo, V1set.and(H1set));
        System.out.print("g_edgeSet = ");
        report(g_edgeSet, V1set.and(V2set));
        System.out.print("g_loads = ");
        report(g_loads, V1set.and(V2set).and(FDset));
        System.out.print("g_stores = ");
        report(g_stores, V1set.and(V2set).and(FDset));
    }
    
    static final void report(BDD bdd, BDD d) {
        System.out.print(bdd.satCount(d));
        System.out.println(" ("+bdd.nodeCount()+" nodes)");
    }
    
    public static final boolean MASK = true;
    
    public BDD buildVarContextMap(BigInteger startV1, BigInteger endV1, BigInteger startV2, BigInteger endV2) {
        if (!CONTEXT_SENSITIVE) {
            return bdd.one();
        }
        BDD r;
        BigInteger sizeV1 = endV1.subtract(startV1);
        BigInteger sizeV2 = endV2.subtract(startV2);
        if (TRACE_CALLGRAPH) {
            System.out.println("Matching V1c("+startV1+","+endV1+") to V2c("+startV2+","+endV2+")");
        }
        if (sizeV1.signum() == -1) {
            if (BREAK_RECURSION) {
                r = bdd.zero();
            } else {
                r = V2c.varRange(startV2.longValue(), endV2.longValue());
                r.andWith(V1c.ithVar(0));
            }
        } else if (sizeV2.signum() == -1) {
            if (BREAK_RECURSION) {
                r = bdd.zero();
            } else {
                r = V1c.varRange(startV1.longValue(), endV1.longValue());
                r.andWith(V2c.ithVar(0));
            }
        } else {
            if (sizeV1.compareTo(sizeV2) != -1) { // >=
                r = V1c.buildAdd(V2c, startV2.subtract(startV1).longValue());
                if (TRACE_SIZES) {
                    System.out.print("add = ");
                    report(r, V1c.set().and(V2c.set()));
                }
                if (MASK) {
                    r.andWith(V1c.varRange(startV1.longValue(), endV1.longValue()));
                    if (TRACE_SIZES) {
                        System.out.print("after mask = ");
                        report(r, V1c.set().and(V2c.set()));
                    }
                }
            } else {
                r = V1c.buildAdd(V2c, startV2.subtract(startV1).longValue());
                if (TRACE_SIZES) {
                    System.out.print("add = ");
                    report(r, V1c.set().and(V2c.set()));
                }
                if (MASK) {
                    r.andWith(V2c.varRange(startV2.longValue(), endV2.longValue()));
                    if (TRACE_SIZES) {
                        System.out.print("after mask = ");
                        report(r, V1c.set().and(V2c.set()));
                    }
                }
            }
        }
        if (TRACE_CALLGRAPH) {
            System.out.println("Result: "+r.toStringWithDomains());
        }
        return r;
    }

    public static BDD buildMod(BDDDomain d1, BDDDomain d2, long val) {
        Assert._assert(d1.varNum() == d2.varNum());

        BDDFactory bdd = d1.getFactory();
        
        BDDBitVector y = bdd.buildVector(d1);
        BDDBitVector z = y.divmod(val, false);
        
        BDDBitVector x = bdd.buildVector(d2);
        BDD result = bdd.one();
        for (int n = 0; n < x.size(); n++) {
            result.andWith(x.getBit(n).biimp(z.getBit(n)));
        }
        x.free(); y.free(); z.free();
        return result;
    }
    
    public void addEdge(BDD context_map, Variable dest, Set srcs) {
        //if (TRACE_EDGES) System.out.println(" Context map: "+context_map.toStringWithDomains());
        int dest_i = getVariableIndex(dest);
        BDD dest_bdd = V2o.ithVar(dest_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Variable src = (Variable) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            src_bdd.andWith(context_map.id());
            src_bdd.andWith(dest_bdd.id());
            if (TRACE_EDGES) System.out.println("Dest="+dest_i+" Src="+src_i);
            if (TRACE_BDD) {
                System.out.println("Adding to g_edgeSet: "+src_bdd.toStringWithDomains());
            }
            g_edgeSet.orWith(src_bdd);
        }
        dest_bdd.free();
    }

    public void solveIncremental() {

        calculateTypeFilter();
        
        BDD oldPointsTo = bdd.zero();
        BDD newPointsTo = g_pointsTo.id();

        BDD fieldPt = bdd.zero();
        BDD storePt = bdd.zero();
        BDD loadAss = bdd.zero();

        // start solving 
        for (int x = 1; ; ++x) {

            if (TRACE_MATCHING) {
                System.out.println("Outer iteration "+x+": ");
            }
            
            // repeat rule (1) in the inner loop
            for (int y = 1; ; ++y) {
                if (TRACE_MATCHING) {
                    System.out.println("Inner iteration "+y+": ");
                }
                if (TRACE_SIZES) {
                    System.out.print("g_pointsTo = ");
                    report(g_pointsTo, V1set.and(V2set));
                    dumpContextInsensitive();
                }
                BDD newPt1 = g_edgeSet.relprod(newPointsTo, V1set);
                newPointsTo.free();
                if (TRACE_SIZES) {
                    System.out.print("newPt1 = ");
                    report(newPt1, V2set.and(H1set));
                }
                BDD newPt2 = newPt1.replace(V2ToV1);
                newPt1.free();
                if (TRACE_SIZES) {
                    System.out.print("newPt2 = ");
                    report(newPt2, V1set.and(H1set));
                }
                newPt2.applyWith(g_pointsTo.id(), BDDFactory.diff);
                if (TRACE_SIZES) {
                    System.out.print("newPt2 (really) = ");
                    report(newPt2, V1set.and(H1set));
                }
                newPt2.andWith(typeFilter.id());
                newPointsTo = newPt2;
                if (TRACE_SIZES) {
                    System.out.print("newPointsTo = ");
                    report(newPointsTo, V1set.and(H1set));
                }
                if (newPointsTo.isZero()) break;
                g_pointsTo.orWith(newPointsTo.id());
            }
            newPointsTo.free();
            newPointsTo = g_pointsTo.apply(oldPointsTo, BDDFactory.diff);

            // apply rule (2)
            BDD tmpRel1 = g_stores.relprod(newPointsTo, V1set); // time-consuming!
            if (TRACE_SIZES) {
                System.out.print("tmpRel1 = ");
                report(tmpRel1, V2set.and(FDset).and(H1set));
            }
            // (V2xFD)xH1
            BDD tmpRel2 = tmpRel1.replace(V2ToV1);
            tmpRel1.free();
            if (TRACE_SIZES) {
                System.out.print("tmpRel2 = ");
                report(tmpRel2, V1set.and(FDset).and(H1set));
            }
            BDD tmpRel3 = tmpRel2.replace(H1ToH2);
            tmpRel2.free();
            if (TRACE_SIZES) {
                System.out.print("tmpRel3 = ");
                report(tmpRel3, V1set.and(FDset).and(H2set));
            }
            // (V1xFD)xH2
            tmpRel3.applyWith(storePt.id(), BDDFactory.diff);
            BDD newStorePt = tmpRel3;
            if (TRACE_SIZES) {
                System.out.print("newStorePt = ");
                report(newStorePt, V1set.and(FDset).and(H2set));
            }
            // cache storePt
            storePt.orWith(newStorePt.id()); // (V1xFD)xH2
            if (TRACE_SIZES) {
                System.out.print("storePt = ");
                report(storePt, V1set.and(FDset).and(H2set));
            }

            BDD newFieldPt = storePt.relprod(newPointsTo, V1set); // time-consuming!
            // (H1xFD)xH2
            newFieldPt.orWith(newStorePt.relprod(oldPointsTo, V1set));
            newStorePt.free();
            oldPointsTo.free();
            // (H1xFD)xH2
            newFieldPt.applyWith(fieldPt.id(), BDDFactory.diff);
            // cache fieldPt
            fieldPt.orWith(newFieldPt.id()); // (H1xFD)xH2
            if (TRACE_SIZES) {
                System.out.print("fieldPt = ");
                report(fieldPt, H1andFDset.and(H2set));
            }

            // apply rule (3)
            BDD tmpRel4 = g_loads.relprod(newPointsTo, V1set); // time-consuming!
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
            if (TRACE_SIZES) {
                System.out.print("loadAss = ");
                report(loadAss, V2set.and(H2set));
            }

            // update oldPointsTo
            oldPointsTo = g_pointsTo.id();

            // convert new points-to relation to normal type
            BDD tmpRel5 = newLoadPt.replace(V2ToV1);
            newPointsTo = tmpRel5.replace(H2ToH1);
            tmpRel5.free();
            newPointsTo.applyWith(g_pointsTo.id(), BDDFactory.diff);

            // apply typeFilter
            newPointsTo.andWith(typeFilter.id());
            if (newPointsTo.isZero()) break;
            g_pointsTo.orWith(newPointsTo.id());
        }
        
        newPointsTo.free();
        fieldPt.free();
        storePt.free();
        loadAss.free();
    }
    
    public void generateBDDSummaries() {
        for (Iterator i=cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            getOrCreateBDDSummary(m);
        }
    }
    
    public class BDDMethodSummary {
        
        /** The method summary that we correspond to. */
        MethodSummary ms;
        
        /** BDD representing all of the variables in this method and its callees.  For escape analysis. */
        BDD vars; // V1c
        int lowVarIndex, highVarIndex;
        int lowHeapIndex, highHeapIndex;
        
        BDD m_pointsTo;     // V1 x H1
        BDD m_stores;       // V1 x (V2 x FD) 
        BDD m_loads;        // (V1 x FD) x V2
        
        BDDMethodSummary(MethodSummary ms) {
            this.ms = ms;
            lowVarIndex = variableIndexMap.size();
            lowHeapIndex = heapobjIndexMap.size();
            reset();
            computeInitial();
            highVarIndex = variableIndexMap.size() - 1;
            highHeapIndex = heapobjIndexMap.size() - 1;
        }
        
        void reset() {
            // initialize relations to zero.
            m_pointsTo = bdd.zero();
            m_stores = bdd.zero();
            m_loads = bdd.zero();
        }
        
        void reportSize() {
            System.out.print("pointsTo = ");
            report(m_pointsTo, V1o.set().and(H1o.set()));
            System.out.print("stores = ");
            report(m_stores, V1o.set().and(FDset).and(V2o.set()));
            System.out.print("loads = ");
            report(m_loads, V1o.set().and(FDset).and(V2o.set()));
        }

        void dispose() {
            m_pointsTo.free();
            m_stores.free();
            m_loads.free();
        }
        
        void computeInitial() {
            long time;
            
            time = System.currentTimeMillis();
            // add edges for all local stuff.
            for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
                Node n = (Node) i.next();
                handleNode(n);
            }
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Converting "+ms.getMethod().getName()+"() to BDD sets: "+(time/1000.));
            
        }
        
        public void handleNode(Node n) {
            
            Iterator j;
            j = n.getEdges().iterator();
            while (j.hasNext()) {
                Map.Entry e = (Map.Entry) j.next();
                jq_Field f = (jq_Field) e.getKey();
                Object o = e.getValue();
                // n.f = o
                if (o instanceof Set) {
                    addStore(n, f, (Set) o);
                } else {
                    addStore(n, f, Collections.singleton(o));
                }
            }
            j = n.getAccessPathEdges().iterator();
            while (j.hasNext()) {
                Map.Entry e = (Map.Entry)j.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                // o = n.f
                if (o instanceof Set) {
                    addLoad((Set) o, n, f);
                } else {
                    addLoad(Collections.singleton(o), n, f);
                }
            }
            if (n instanceof HeapObject) {
                HeapObject ho = (HeapObject) n;
                addObjectAllocation(n, ho);
                addAllocType(ho, (jq_Reference) n.getDeclaredType());
            }
            if (n instanceof ParamNode ||
                ms.getReturned().contains(n) ||
                ms.getThrown().contains(n)) {
                addUpwardEscapeNode(n);
            }
            if (n instanceof ReturnedNode ||
                n.getPassedParameters() != null) {
                addDownwardEscapeNode(n);
            }
            if (n instanceof GlobalNode) {
                addGlobalEdge(GlobalNode.GLOBAL, Collections.singleton(n));
                addGlobalEdge(n, Collections.singleton(GlobalNode.GLOBAL));
                addVarType(n, PrimordialClassLoader.getJavaLangObject());
            } else {
                addVarType(n, (jq_Reference) n.getDeclaredType());
            }
        }
        
        public void addObjectAllocation(Variable dest, HeapObject site) {
            int dest_i = getVariableIndex(dest);
            int site_i = getHeapObjectIndex(site);
            BDD dest_bdd = V1o.ithVar(dest_i);
            BDD site_bdd = H1o.ithVar(site_i);
            dest_bdd.andWith(site_bdd);
            m_pointsTo.orWith(dest_bdd);
        }

        public void addLoad(Set dests, Variable base, jq_Field f) {
            int base_i = getVariableIndex(base);
            int f_i = getFieldIndex(f);
            BDD base_bdd = V1o.ithVar(base_i);
            BDD f_bdd = FD.ithVar(f_i);
            for (Iterator i=dests.iterator(); i.hasNext(); ) {
                // FieldNode
                Variable dest = (Variable) i.next();
                int dest_i = getVariableIndex(dest);
                BDD dest_bdd = V2o.ithVar(dest_i);
                dest_bdd.andWith(f_bdd.id());
                dest_bdd.andWith(base_bdd.id());
                m_loads.orWith(dest_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
    
        public void addStore(Variable base, jq_Field f, Set srcs) {
            int base_i = getVariableIndex(base);
            int f_i = getFieldIndex(f);
            BDD base_bdd = V2o.ithVar(base_i);
            BDD f_bdd = FD.ithVar(f_i);
            for (Iterator i=srcs.iterator(); i.hasNext(); ) {
                Variable src = (Variable) i.next();
                int src_i = getVariableIndex(src);
                BDD src_bdd = V1o.ithVar(src_i);
                src_bdd.andWith(f_bdd.id());
                src_bdd.andWith(base_bdd.id());
                m_stores.orWith(src_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
        
        public void addUpwardEscapeNode(Variable n) {
            int n_i = getVariableIndex(n);
        }
        
        public void addDownwardEscapeNode(Variable n) {
            int n_i = getVariableIndex(n);
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("BDD Summary for ");
            sb.append(ms.getMethod());
            sb.append(':');
            sb.append(Strings.lineSep);
            sb.append("Loads=");
            sb.append(m_loads.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Stores=");
            sb.append(m_stores.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Points-to=");
            sb.append(m_pointsTo.toStringWithDomains());
            sb.append(Strings.lineSep);
            return sb.toString();
        }
    }
    
    public static class IndexMap {
        private final String name;
        private final HashMap hash;
        private final Object[] list;
        private int index;
        
        public IndexMap(String name, int maxIndex) {
            this.name = name;
            hash = new HashMap();
            list = new Object[maxIndex];
            index = -1;
        }
        
        public int get(Object o) {
            Integer i = (Integer) hash.get(o);
            if (i == null) {
                int j = ++index;
                while (list[j] != null)
                    ++j;
                list[j] = o;
                hash.put(o, i = new Integer(j));
                if (TRACE_MAPS) System.out.println(this+"["+j+"] = "+o);
            }
            return i.intValue();
        }
        
        public Object get(int i) {
            return list[i];
        }
        
        public boolean contains(Object o) {
            return hash.containsKey(o);
        }
        
        public int size() {
            return index+1;
        }
        
        public String toString() {
            return name;
        }
        
        public Iterator iterator() {
            return new Iterator() {
                int foo = -1;

                public void remove() {
                    throw new UnsupportedOperationException();
                }

                public boolean hasNext() {
                    return foo < index;
                }

                public Object next() {
                    if (!hasNext()) throw new java.util.NoSuchElementException();
                    return list[++foo];
                }
            };
        }
        
    }

    public void goForIt() {
        List list = Traversals.reversePostOrder(cg.getNavigator(), cg.getRoots());
        for (Iterator i=list.iterator(); i.hasNext(); ) {
            jq_Method o = (jq_Method) i.next();
            if (o.getBytecode() == null) {
                continue;
            }
            ControlFlowGraph cfg = CodeCache.getCode(o);
            MethodSummary ms = MethodSummary.getSummary(cfg);
            addRelations(ms);
            bindCallEdges(ms);
        }
    }

    public static boolean TRACE_ESCAPE = false;

    public void escapeAnalysis() {
        
        BDD escapingLocations = bdd.zero();
        
        BDD myPointsTo;
        myPointsTo = g_pointsTo.exist(V1c.set().and(H1c.set()));
        
        List order = Traversals.postOrder(cg.getNavigator(), cg.getRoots());
        for (Iterator i=order.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            BDDMethodSummary bms = getOrCreateBDDSummary(m);
            if (bms == null) continue;
            BDD range;
            SCComponent scc = (SCComponent) pn.getSCC(m);
            if (scc.isLoop()) {
                bms.vars = bdd.zero();
            } else {
                bms.vars = V1o.varRange(bms.lowVarIndex, bms.highVarIndex);
                for (Iterator j=cg.getCallees(m).iterator(); j.hasNext(); ) {
                    jq_Method callee = (jq_Method) j.next();
                    BDDMethodSummary bms2 = getOrCreateBDDSummary(callee);
                    if (bms2 == null) continue;
                    bms.vars.orWith(bms2.vars.id());
                }
            }
            HashMap concreteNodes = new HashMap();
            MethodSummary ms = bms.ms;
            for (Iterator j=ms.nodeIterator(); j.hasNext(); ) {
                Node o = (Node) j.next();
                if (o instanceof ConcreteTypeNode) {
                    ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                    concreteNodes.put(ctn.getLocation(), ctn);
                }
                boolean bad = false;
                if (o.getEscapes()) {
                    if (TRACE_ESCAPE) System.out.println(o+" escapes, bad");
                    bad = true;
                } else if (cg.getRoots().contains(m) && ms.getThrown().contains(o)) {
                    if (TRACE_ESCAPE) System.out.println(o+" is thrown from root set, bad");
                    bad = true;
                } else {
                    Set passedParams = o.getPassedParameters();
                    if (passedParams != null) {
                        outer:
                        for (Iterator k=passedParams.iterator(); k.hasNext(); ) {
                            PassedParameter pp = (PassedParameter) k.next();
                            ProgramLocation mc = pp.getCall();
                            for (Iterator a=getTargetMethods(mc).iterator(); a.hasNext(); ) {
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
                if (bad) {
                    int v_i = getVariableIndex((Variable) o);
                    bms.vars.and(V1o.ithVar(v_i).not());
                }
            }
            if (TRACE_ESCAPE) System.out.println("Non-escaping locations for "+m+" = "+bms.vars.toStringWithDomains());
            ControlFlowGraph cfg = CodeCache.getCode(m);
            boolean trivial = false;
            for (QuadIterator j=new QuadIterator(cfg); j.hasNext(); ) {
                Quad q = j.nextQuad();
                if (q.getOperator() instanceof Operator.New ||
                    q.getOperator() instanceof Operator.NewArray) {
                    ProgramLocation pl = new QuadProgramLocation(m, q);
                    ConcreteTypeNode ctn = (ConcreteTypeNode) concreteNodes.get(pl);
                    if (ctn == null) {
                        //trivial = true;
                        trivial = q.getOperator() instanceof Operator.New;
                        System.out.println(cfg.getMethod()+": "+q+" trivially doesn't escape.");
                    } else {
                        int h_i = getHeapObjectIndex(ctn);
                        BDD h = H1o.ithVar(h_i);
                        if (TRACE_ESCAPE) {
                            System.out.println("Heap location: "+h.toStringWithDomains()+" = "+ctn);
                            System.out.println("Pointed to by: "+myPointsTo.restrict(h).toStringWithDomains());
                        }
                        h.andWith(bms.vars.not());
                        escapingLocations.orWith(h);
                    }
                }
            }
            if (trivial) {
                System.out.println(cfg.fullDump());
            }
        }
        BDD escapingHeap = escapingLocations.relprod(myPointsTo, V1set);
        System.out.println("Escaping heap: "+escapingHeap.satCount(H1o.set()));
        //System.out.println("Escaping heap: "+escapingHeap.toStringWithDomains());
        BDD capturedHeap = escapingHeap.not();
        capturedHeap.andWith(H1o.varRange(0, heapobjIndexMap.size()-1));
        System.out.println("Captured heap: "+capturedHeap.satCount(H1o.set()));
        
        int capturedSites = 0;
        int escapedSites = 0;
        long capturedSize = 0L;
        long escapedSize = 0L;
        
        for (Iterator i=heapobjIndexMap.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            int ndex = heapobjIndexMap.get(n);
            if (n instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) n;
                jq_Reference t = (jq_Reference) ctn.getDeclaredType();
                int size = 0;
                if (t instanceof jq_Class)
                    size = ((jq_Class) t).getInstanceSize();
                else
                    continue;
                BDD bdd = capturedHeap.and(H1o.ithVar(ndex));
                if (capturedHeap.and(H1o.ithVar(ndex)).isZero()) {
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
    }

    BDD getAllHeapOfType(jq_Reference type) {
        if (false) {
            int j=0;
            BDD result = bdd.zero();
            for (Iterator i=heapobjIndexMap.iterator(); i.hasNext(); ++j) {
                Node n = (Node) i.next();
                Assert._assert(this.heapobjIndexMap.get(n) == j);
                if (n.getDeclaredType() == type)
                    result.orWith(V1o.ithVar(j));
            }
            return result;
        } else {
            int i = typeIndexMap.get(type);
            BDD a = T2.ithVar(i);
            BDD result = aC.restrict(a);
            a.free();
            return result;
        }
    }

}

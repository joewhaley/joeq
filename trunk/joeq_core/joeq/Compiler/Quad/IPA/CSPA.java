// CSPA.java, created Jun 15, 2003 10:08:38 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad.IPA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDBitVector;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.BuDDyFactory;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Quad.CachedCallGraph;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.LoadedCallGraph;
import Compil3r.Quad.MethodInline;
import Compil3r.Quad.MethodSummary;
import Compil3r.Quad.ProgramLocation;
import Compil3r.Quad.RootedCHACallGraph;
import Compil3r.Quad.MethodSummary.ConcreteObjectNode;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.MethodSummary.ReturnValueNode;
import Compil3r.Quad.MethodSummary.ReturnedNode;
import Compil3r.Quad.MethodSummary.ThrownExceptionNode;
import Compil3r.Quad.MethodSummary.UnknownTypeNode;
import Main.HostedVM;
import Run_Time.TypeCheck;
import Util.Assert;
import Util.Strings;
import Util.Collections.Pair;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;

/**
 * CSPA
 * 
 * @author John Whaley
 * @version $Id$
 */
public class CSPA {

    public static final boolean TRACE_ALL = false;
    
    public static final boolean TRACE_MATCHING  = true || TRACE_ALL;
    public static final boolean TRACE_TYPES     = false || TRACE_ALL;
    public static final boolean TRACE_MAPS      = false || TRACE_ALL;
    public static final boolean TRACE_SIZES      = true || TRACE_ALL;
    public static final boolean TRACE_EDGES     = false || TRACE_ALL;
    public static final boolean TRACE_TIMES     = false || TRACE_ALL;
    public static final boolean TRACE_VARORDER     = true || TRACE_ALL;
    public static final boolean TRACE_NUMBERING     = false || TRACE_ALL;
    
    public static final boolean USE_CHA     = true;
    public static final boolean DO_INLINING = true;

    public static boolean LOADED_CALLGRAPH = false;
    public static final boolean TEST_CALLGRAPH = false;
    
    public static boolean BREAK_RECURSION = false;
    
    public static final boolean CONTEXT_SENSITIVE = true;
    public static final boolean CONTEXT_SENSITIVE_HEAP = false;
    
    public static void main(String[] args) {
        HostedVM.initialize();
        
        CodeCache.AlwaysMap = true;
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        CallGraph cg = null;
        if (new java.io.File("callgraph").exists()) {
            try {
                System.out.print("Loading initial call graph...");
                long time = System.currentTimeMillis();
                cg = new LoadedCallGraph("callgraph");
                time = System.currentTimeMillis() - time;
                System.out.println("done. ("+time/1000.+" seconds)");
                //Compil3r.Quad.RootedCHACallGraph.test(cg);
                roots = cg.getRoots();
                LOADED_CALLGRAPH = true;
            } catch (java.io.IOException x) {
                x.printStackTrace();
            }
        }
        if (cg == null) {
            System.out.print("Setting up initial call graph...");
            long time = System.currentTimeMillis();
            if (USE_CHA) {
                cg = new RootedCHACallGraph();
                cg = new CachedCallGraph(cg);
                cg.setRoots(roots);
            } else {
            }
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
        
            System.out.print("Calculating reachable methods...");
            time = System.currentTimeMillis();
            /* Calculate the reachable methods once to touch each method,
               so that the set of types are stable. */
            cg.calculateReachableMethods(roots);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
        
            try {
                java.io.FileWriter fw = new java.io.FileWriter("callgraph");
                java.io.PrintWriter pw = new java.io.PrintWriter(fw);
                LoadedCallGraph.write(cg, pw);
                pw.close();
            } catch (java.io.IOException x) {
                x.printStackTrace();
            }
            
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
                MethodSummary ms = MethodSummary.getSummary(cfg);
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
        
        // Allocate CSPA object.  Also initializes BDD package.
        CSPA dis = new CSPA(cg);
        dis.roots = roots;
        
        // Add edges for existing globals.
        dis.addGlobals();
        
        System.out.print("Initial generation and counting paths...");
        time = System.currentTimeMillis();
        long paths = dis.countPaths();
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        System.out.println(paths+" paths");
        
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
    
    public void addGlobals() {
        GlobalNode.GLOBAL.addDefaultStatics();
        addGlobalObjectAllocation(GlobalNode.GLOBAL, null);
        addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        handleGlobalNode(GlobalNode.GLOBAL);
        for (Iterator i=ConcreteObjectNode.getAll().iterator(); i.hasNext(); ) {
            handleGlobalNode((ConcreteObjectNode) i.next());
        }
    }
    
    public void addGlobalV1Context(BDD b) {
        if (false)
            b.andWith(V1c.ithVar(0));
        else
            b.andWith(V1c.domain());
    }
    public void addGlobalV2Context(BDD b) {
        if (false)
            b.andWith(V2c.ithVar(0));
        else
            b.andWith(V2c.domain());
    }
    public void addGlobalH1Context(BDD b) {
        if (false)
            b.andWith(H1c.ithVar(0));
        else
            b.andWith(V2c.domain());
    }
    public void addV1Context(BDD b, long lo, long hi) {
        if (CONTEXT_SENSITIVE)
            b.andWith(V1c.varRange(lo, hi));
    }
    public void addV2Context(BDD b, long lo, long hi) {
        if (CONTEXT_SENSITIVE)
            b.andWith(V2c.varRange(lo, hi));
    }
    public void addH1Context(BDD b, long lo, long hi) {
        if (CONTEXT_SENSITIVE) {
            if (CONTEXT_SENSITIVE_HEAP)
                b.andWith(H1c.varRange(lo, hi));
            else
                b.andWith(H1c.ithVar(0));
        }
    }
    
    public void addGlobalObjectAllocation(Node dest, Node site) {
        int dest_i = getVariableIndex(dest);
        int site_i = getHeapobjIndex(site);
        BDD dest_bdd = V1o.ithVar(dest_i);
        addGlobalV1Context(dest_bdd);
        BDD site_bdd = H1o.ithVar(site_i);
        addGlobalH1Context(site_bdd);
        dest_bdd.andWith(site_bdd);
        g_pointsTo.orWith(dest_bdd);
    }
    
    public void addGlobalLoad(Set dests, Node base, jq_Field f) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V1o.ithVar(base_i);
        addGlobalV1Context(base_bdd);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=dests.iterator(); i.hasNext(); ) {
            FieldNode dest = (FieldNode) i.next();
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V2o.ithVar(dest_i);
            addGlobalV2Context(dest_bdd);
            dest_bdd.andWith(f_bdd.id());
            dest_bdd.andWith(base_bdd.id());
            g_loads.orWith(dest_bdd);
        }
        base_bdd.free(); f_bdd.free();
    }
    
    public void addGlobalStore(Node base, jq_Field f, Set srcs) {
        int base_i = getVariableIndex(base);
        int f_i = getFieldIndex(f);
        BDD base_bdd = V2o.ithVar(base_i);
        addGlobalV2Context(base_bdd);
        BDD f_bdd = FD.ithVar(f_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Node src = (Node) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            addGlobalV1Context(src_bdd);
            src_bdd.andWith(f_bdd.id());
            src_bdd.andWith(base_bdd.id());
            g_stores.orWith(src_bdd);
        }
        base_bdd.free(); f_bdd.free();
    }
    
    // v2 = v1;
    public void addGlobalEdge(Node dest, Collection srcs) {
        int dest_i = getVariableIndex(dest);
        BDD dest_bdd = V2o.ithVar(dest_i);
        addGlobalV2Context(dest_bdd);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Node src = (Node) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            addGlobalV1Context(src_bdd);
            src_bdd.andWith(dest_bdd.id());
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
        if (n instanceof ConcreteTypeNode ||
            n instanceof UnknownTypeNode ||
            n instanceof ConcreteObjectNode) {
            addGlobalObjectAllocation(n, n);
            addAllocType(n, (jq_Reference) n.getDeclaredType());
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
    private final BDDFactory bdd;
    
    public static final int VARBITS = 15;
    public static final int HEAPBITS = 12;
    public static final int FIELDBITS = 10;
    public static final int CLASSBITS = 10;
    public static final int CONTEXTBITS = 35;
    
    // the size of domains, can be changed to reflect the size of inputs
    int domainBits[] = {VARBITS, CONTEXTBITS,
                        VARBITS, CONTEXTBITS,
                        FIELDBITS,
                        HEAPBITS, CONTEXTBITS,
                        HEAPBITS, CONTEXTBITS};
    // to be computed in sysInit function
    int domainSpos[] = {0,  0,  0,  0,  0,  0,  0,  0,  0}; 
    
    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1o, V2o, H1o, H2o;
    BDDDomain V1c, V2c, H1c, H2c;
    BDDDomain FD;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2, T3, T4; 
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

    public CSPA(CallGraph cg) {
        this(cg, DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
    
    CallGraph cg;
    Collection roots;
    
    public CSPA(CallGraph cg, int nodeCount, int cacheSize) {
        this.cg = cg;
        
        bdd = BuDDyFactory.init(nodeCount, cacheSize);
        
        bdd.setCacheRatio(4);
        bdd.setMaxIncrease(nodeCount/4);
        
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
        T3 = H2o;
        T4 = V2o;
        for (int i=0; i<domainBits.length; ++i) {
            Assert._assert(bdd_domains[i].varNum() == domainBits[i]);
        }
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "FD_H2cxH2o_V2cxV2oxV1cxV1o_H1cxH1o");
        
        int[] varorder = makeVarOrdering(reverseLocal, ordering);
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

    int[] makeVarOrdering(boolean reverseLocal, String ordering) {
        
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
            if (s.equals("V1o")) d = V1o;
            else if (s.equals("V1c")) d = V1c;
            else if (s.equals("V2o")) d = V2o;
            else if (s.equals("V2c")) d = V2c;
            else if (s.equals("FD")) d = FD;
            else if (s.equals("H1o")) d = H1o;
            else if (s.equals("H1c")) d = H1c;
            else if (s.equals("H2o")) d = H2o;
            else if (s.equals("H2c")) d = H2c;
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
            bitIndex = fillInVarIndices(doms, i-numberOfDomains, numberOfDomains+1,
                                        localOrders, bitIndex, varorder);
            if (!st.hasMoreTokens()) {
                Collection not_done = new ArrayList(Arrays.asList(bdd_domains));
                not_done.removeAll(Arrays.asList(doms));
                Assert._assert(not_done.isEmpty(), not_done.toString());
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
    
    int fillInVarIndices(BDDDomain[] doms, int domainIndex, int numDomains,
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
    
    void getVariableMap(int[] map, BDDDomain[] doms) {
        int idx = 0;
        for (int var = 0; var < doms.length; var++) {
            int[] vars = doms[var].vars();
            for (int i = 0; i < vars.length; i++) {
                map[idx++] = vars[i];
            }
        }
    }
    
    /* remap according to a map */
    void remapping(int[] varorder, int[] maps) {
        int[] varorder2 = new int[varorder.length];
        for (int i = 0; i < varorder.length; i++) {
            varorder2[i] = maps[varorder[i]];
        }
        System.arraycopy(varorder2, 0, varorder, 0, varorder.length);
    }
    
    IndexMap/* Node->index */ variableIndexMap = new IndexMap("Variable", 1 << VARBITS);
    IndexMap/* Node->index */ heapobjIndexMap = new IndexMap("HeapObj", 1 << HEAPBITS);
    IndexMap/* jq_Field->index */ fieldIndexMap = new IndexMap("Field", 1 << FIELDBITS);
    IndexMap/* jq_Reference->index */ typeIndexMap = new IndexMap("Class", 1 << CLASSBITS);

    int getVariableIndex(Node dest) {
        return variableIndexMap.get(dest);
    }
    int getHeapobjIndex(Node site) {
        return heapobjIndexMap.get(site);
    }
    int getFieldIndex(jq_Field f) {
        return fieldIndexMap.get(f);
    }
    int getTypeIndex(jq_Reference f) {
        return typeIndexMap.get(f);
    }
    Node getVariable(int index) {
        return (Node) variableIndexMap.get(index);
    }
    Node getHeapobj(int index) {
        return (Node) heapobjIndexMap.get(index);
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

    public void addAllocType(Node site, jq_Reference type) {
        addClassType(type);
        int site_i = getHeapobjIndex(site);
        int type_i = getTypeIndex(type);
        BDD site_bdd = H1o.ithVar(site_i);
        BDD type_bdd = T2.ithVar(type_i);
        type_bdd.andWith(site_bdd);
        if (TRACE_TYPES) System.out.println("Adding alloc type: "+type_bdd.toStringWithDomains());
        aC.orWith(type_bdd);
    }

    public void addVarType(Node var, jq_Reference type) {
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
        }
        return bms;
    }
    
    Map bddSummaries = new HashMap();
    BDDMethodSummary getBDDSummary(MethodSummary ms) {
        BDDMethodSummary result = (BDDMethodSummary) bddSummaries.get(ms);
        return result;
    }
    
    public ProgramLocation mapCall(ProgramLocation callSite) {
        if (LOADED_CALLGRAPH && callSite instanceof ProgramLocation.QuadProgramLocation) {
            jq_Method m = (jq_Method) callSite.getMethod();
            Map map = CodeCache.getBCMap(m);
            int bcIndex = ((Integer) map.get(((ProgramLocation.QuadProgramLocation) callSite).getQuad())).intValue();
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

    public void addRelations(MethodSummary ms) {
        BDDMethodSummary bms = this.getBDDSummary(ms);

        //System.out.println("Adding relations for "+ms.getMethod());
        
        BDD t1 = bms.m_pointsTo.id();
        addV1Context(t1, 0, bms.n_paths);
        addH1Context(t1, 0, bms.n_paths);
        g_pointsTo.orWith(t1);

        t1 = bms.m_loads.id();
        addV1Context(t1, 0, bms.n_paths);
        addV2Context(t1, 0, bms.n_paths);
        g_loads.orWith(t1);
        
        t1 = bms.m_stores.id();
        addV1Context(t1, 0, bms.n_paths);
        addV2Context(t1, 0, bms.n_paths);
        g_stores.orWith(t1);
        
        bms.dispose();
    }
    
    public void bindCallEdges(MethodSummary caller) {
        if (TRACE_EDGES) System.out.println("Adding call graph edges for "+caller.getMethod());
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
        Object t = caller.getMethod().and_getReturnType();
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
        if (TRACE_EDGES)
            System.out.println("Adding call graph edge "+caller.getMethod()+"->"+callee.getMethod());
        BDDMethodSummary caller_s = this.getBDDSummary(caller);
        BDDMethodSummary callee_s = this.getBDDSummary(callee);
        Pair p = new Pair(mapCall(mc), callee.getMethod());
        Range r = (Range) callGraphEdges.get(p);
        if (backEdges.contains(p))
            System.out.println("Back edge: "+p+"="+r);
        if (TRACE_EDGES)
            System.out.println("Context range "+r);
        BDD context_map;
        // for parameters: V1 in caller matches V2 in callee
        context_map = buildVarContextMap(0, caller_s.n_paths - 1, r.low, r.high);
        for (int i=0; i<mc.getNumParams(); ++i) {
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
        if (rvn != null) {
            Set s = callee.getReturned();
            // for returns: V1 in callee matches V2 in caller
            context_map = buildVarContextMap(r.low, r.high, 0, caller_s.n_paths - 1);
            if (TRACE_EDGES) System.out.println("Adding edges for "+rvn);
            addEdge(context_map, rvn, s);
            context_map.free();
        }
        ThrownExceptionNode ten = caller.getTEN(mc);
        if (ten != null) {
            Set s = callee.getThrown();
            context_map = buildVarContextMap(r.low, r.high, 0, caller_s.n_paths - 1);
            if (TRACE_EDGES) System.out.println("Adding edges for "+ten);
            addEdge(context_map, ten, s);
            context_map.free();
        }
    }

    public void dumpGlobalNodes() {
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
    
    public BDD buildRecursiveMap(long sizeV1, long sizeV2) {
        if (!CONTEXT_SENSITIVE) {
            return bdd.one();
        }
        BDD r;
        if (sizeV1 == sizeV2) {
            r = buildVarContextMap(0, sizeV1 - 1, 0, sizeV1 - 1);
        } else {
            System.out.print("buildRecursiveMap: "+sizeV1+", "+sizeV2+" = ");
            if (sizeV1 > sizeV2) {
                r = buildMod(V1c, V2c, sizeV2);
                r.andWith(V1c.varRange(0, sizeV1));
                if (sizeV1 < 256L) System.out.println(r.toStringWithDomains());
            } else {
                r = buildMod(V2c, V1c, sizeV1);
                r.andWith(V2c.varRange(0, sizeV2));
                if (sizeV2 < 256L) System.out.println(r.toStringWithDomains());
            }
        }
        return r;
    }
    
    public static final boolean MASK = true;
    
    public BDD buildVarContextMap(long startV1, long endV1, long startV2, long endV2) {
        if (!CONTEXT_SENSITIVE) {
            return bdd.one();
        }
        BDD r;
        long sizeV1 = endV1 - startV1;
        long sizeV2 = endV2 - startV2;
        if (sizeV1 < 0L) {
            if (BREAK_RECURSION) {
                r = bdd.zero();
            } else {
                r = V2c.varRange(startV2, endV2);
                r.andWith(V1c.ithVar(0));
            }
        } else if (sizeV2 < 0L) {
            if (BREAK_RECURSION) {
                r = bdd.zero();
            } else {
                r = V1c.varRange(startV1, endV1);
                r.andWith(V2c.ithVar(0));
            }
        } else {
            if (sizeV1 >= sizeV2) {
                r = V1c.buildAdd(V2c, startV2 - startV1);
                if (MASK)
                    r.andWith(V1c.varRange(startV1, endV1));
            } else {
                r = V1c.buildAdd(V2c, startV2 - startV1);
                if (MASK)
                    r.andWith(V2c.varRange(startV2, endV2));
            }
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
    
    public void addEdge(BDD context_map, Node dest, Set srcs) {
        if (TRACE_EDGES) System.out.println(" Context map: "+context_map.toStringWithDomains());
        int dest_i = getVariableIndex(dest);
        BDD dest_bdd = V2o.ithVar(dest_i);
        for (Iterator i=srcs.iterator(); i.hasNext(); ) {
            Node src = (Node) i.next();
            int src_i = getVariableIndex(src);
            BDD src_bdd = V1o.ithVar(src_i);
            src_bdd.andWith(context_map.id());
            src_bdd.andWith(dest_bdd.id());
            if (TRACE_EDGES) System.out.println("Dest="+dest_i+" Src="+src_i);
            //if (TRACE_EDGES) System.out.println(" Adding edge: "+src_bdd.toStringWithDomains());
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
    
    public static class Range {
        public long low, high;
        public Range(long l, long h) {
            this.low = l; this.high = h;
        }
        public String toString() {
            return "<"+low+','+high+'>';
        }
    }
    
    HashMap callGraphEdges = new HashMap();
    
    long max_paths = 0L;
    
    public long countPaths() {
        List list = Traversals.reversePostOrder(cg.getNavigator(), cg.getRoots());
        boolean again;
        do {
            again = false;
            for (Iterator i=list.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method) i.next();
                if (countPaths_helper(m)) {
                    if (TRACE_NUMBERING) System.out.println(m+" changed.");
                    again = true;
                }
            }
        } while (again);
        return max_paths;
    }
    
    public long countPaths2() {
        if (TRACE_NUMBERING) System.out.print("Building and sorting SCCs...");
        Navigator navigator = cg.getNavigator();
        Set sccs = SCComponent.buildSCC(roots, navigator);
        SCCTopSortedGraph graph = SCCTopSortedGraph.topSort(sccs);
        if (TRACE_NUMBERING) System.out.print("done.");
        
        /* Walk through SCCs in forward order. */
        SCComponent scc = graph.getFirst();
        while (scc != null) {
            /* Visit each method in the SCC. */
            if (TRACE_NUMBERING) System.out.println("Visiting SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)"));
            Object[] nodes = scc.nodes();
            boolean change = false;
            for (int i=0; i<nodes.length; ++i) {
                jq_Method m = (jq_Method) nodes[i];
                if (countPaths_helper(m)) {
                    if (TRACE_NUMBERING && scc.isLoop()) System.out.println(m+" changed.");
                    change = true;
                }
            }
            if (scc.isLoop() && change) {
                if (TRACE_NUMBERING) System.out.println("Loop changed, redoing SCC.");
                continue;
            }
            scc = scc.nextTopSort();
        }
        return max_paths;
    }
        
    HashSet backEdges = new HashSet();
    
    public boolean countPaths_helper(jq_Method callee) {
        BDDMethodSummary ms = getOrCreateBDDSummary(callee);
        if (ms == null) {
            return false;
        }
        boolean already_visited = ms.n_paths != 0L;
        long myPaths = 0L, maxPaths = 0L;
        boolean change = false;
        // iterate over the callers to find the number of paths to this method.
        for (Iterator j=cg.getCallers(callee).iterator(); j.hasNext(); ) {
            jq_Method caller = (jq_Method) j.next();
            BDDMethodSummary ms2 = getOrCreateBDDSummary(caller);
            if (ms2 == null) {
                continue;
            }
            for (Iterator k=cg.getCallSites(caller).iterator(); k.hasNext(); ) {
                ProgramLocation mc = (ProgramLocation) k.next();
                Assert._assert(mc == mapCall(mc));
                if (getTargetMethods(mc).contains(callee)) {
                    Pair edge = new Pair(mc, callee);
                    Range r = (Range) callGraphEdges.get(edge);
                    if (r == null) {
                        // never visited this edge before.
                        Assert._assert(!already_visited);
                        if (isCallInteresting(edge, myPaths, ms2.n_paths)) {
                            r = new Range(myPaths, myPaths + ms2.n_paths - 1L);
                            myPaths = r.high + 1;
                        } else {
                            r = new Range(0, ms2.n_paths - 1L);
                        }
                        //System.out.println("Registering call graph edge "+edge+": "+r);
                        callGraphEdges.put(edge, r);
                        if (ms2.n_paths == 0L) {
                            //System.out.println("Back edge "+edge);
                            if (BREAK_RECURSION)
                                backEdges.add(edge);
                            else
                                change = true;
                        }
                        maxPaths = Math.max(maxPaths, r.high + 1);
                    } else {
                        // edge has been visited before (loop)
                        long new_paths = Math.max(r.high, ms2.n_paths-1);
                        change |= r.high != new_paths;
                        if (TRACE_NUMBERING && r.high != new_paths)
                            System.out.println(edge+" updated, old high = "+r.high+" new high = "+new_paths);
                        Assert._assert(new_paths <= max_paths);
                        r.high = new_paths;
                    }
                    if (TRACE_NUMBERING)
                        System.out.println(edge+": "+r);
                }
            }
        }
        if (maxPaths == 0L) maxPaths = 1L;
        ms.n_paths = maxPaths;
        max_paths = Math.max(max_paths, maxPaths);
        return change;
    }
    
    public boolean isCallInteresting(Pair p, long callee_paths, long caller_paths) {
        if (callee_paths + caller_paths >= (1L << CONTEXTBITS))
            return false;
        return true;
    }
    
    public class BDDMethodSummary {
        
        /** The method summary that we correspond to. */
        MethodSummary ms;
        
        /** The number of paths that reach this method. */
        long n_paths;
        
        /** BDD representing all of the paths that reach this method. */
        BDD context; // V1c
        
        BDD m_pointsTo;     // V1 x H1
        BDD m_stores;       // V1 x (V2 x FD) 
        BDD m_loads;        // (V1 x FD) x V2
        
        BDDMethodSummary(MethodSummary ms) {
            this.ms = ms;
            reset();
            computeInitial();
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
            if (n instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) n;
                addObjectAllocation(ctn, ctn);
                addAllocType(ctn, (jq_Reference) ctn.getDeclaredType());
            } else if (n instanceof UnknownTypeNode) {
                UnknownTypeNode utn = (UnknownTypeNode) n;
                addObjectAllocation(utn, utn);
                addAllocType(utn, (jq_Reference) utn.getDeclaredType());
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
        
        public void addObjectAllocation(Node dest, Node site) {
            int dest_i = getVariableIndex(dest);
            int site_i = getHeapobjIndex(site);
            BDD dest_bdd = V1o.ithVar(dest_i);
            BDD site_bdd = H1o.ithVar(site_i);
            dest_bdd.andWith(site_bdd);
            m_pointsTo.orWith(dest_bdd);
        }

        public void addLoad(Set dests, Node base, jq_Field f) {
            int base_i = getVariableIndex(base);
            int f_i = getFieldIndex(f);
            BDD base_bdd = V1o.ithVar(base_i);
            BDD f_bdd = FD.ithVar(f_i);
            for (Iterator i=dests.iterator(); i.hasNext(); ) {
                FieldNode dest = (FieldNode) i.next();
                int dest_i = getVariableIndex(dest);
                BDD dest_bdd = V2o.ithVar(dest_i);
                dest_bdd.andWith(f_bdd.id());
                dest_bdd.andWith(base_bdd.id());
                m_loads.orWith(dest_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
    
        public void addStore(Node base, jq_Field f, Set srcs) {
            int base_i = getVariableIndex(base);
            int f_i = getFieldIndex(f);
            BDD base_bdd = V2o.ithVar(base_i);
            BDD f_bdd = FD.ithVar(f_i);
            for (Iterator i=srcs.iterator(); i.hasNext(); ) {
                Node src = (Node) i.next();
                int src_i = getVariableIndex(src);
                BDD src_bdd = V1o.ithVar(src_i);
                src_bdd.andWith(f_bdd.id());
                src_bdd.andWith(base_bdd.id());
                m_stores.orWith(src_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
        
        public void addUpwardEscapeNode(Node n) {
            int n_i = getVariableIndex(n);
        }
        
        public void addDownwardEscapeNode(Node n) {
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

}

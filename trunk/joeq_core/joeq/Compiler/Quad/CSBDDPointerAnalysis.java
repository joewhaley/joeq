// CSBDDPointerAnalysis.java, created Mon May  5  2:59:09 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.BuDDyFactory;

import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.IPA.*;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.FieldNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.PassedParameter;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnValueNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnedNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ThrownExceptionNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Main.HostedVM;
import Run_Time.TypeCheck;
import Util.Assert;
import Util.Strings;
import Util.Collections.Pair;
import Util.Collections.Triple;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class CSBDDPointerAnalysis {

    public static final boolean TRACE_ALL = false;

    public static final boolean TRACE_WORKLIST  = false || TRACE_ALL;
    public static final boolean TRACE_SUMMARIES = false || TRACE_ALL;
    public static final boolean TRACE_CALLEE    = false || TRACE_ALL;
    public static final boolean TRACE_OVERLAP   = false || TRACE_ALL;
    public static final boolean TRACE_MATCHING  = false || TRACE_ALL;
    public static final boolean TRACE_TRIMMING  = false || TRACE_ALL;
    public static final boolean TRACE_TYPES     = false || TRACE_ALL;
    public static final boolean TRACE_MAPS      = false || TRACE_ALL;
    public static final boolean TRACE_TIMES     = false || TRACE_ALL;

    public static final boolean USE_CHA = true;
    public static final boolean DO_INLINING = true;

    public static void main(String[] args) {
        HostedVM.initialize();
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        System.out.print("Setting up initial call graph...");
        long time = System.currentTimeMillis();
        CallGraph cg;
        if (USE_CHA) {
            cg = new RootedCHACallGraph();
            cg = new CachedCallGraph(cg);
            cg.setRoots(roots);
        } else {
            BDDPointerAnalysis pa = new BDDPointerAnalysis();
            pa.reset();
            cg = pa.goIncremental(roots);
            pa.done();
        }
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        if (DO_INLINING) {
            System.out.print("Doing inlining on call graph...");
            time = System.currentTimeMillis();
            // pre-initialize all classes so that we can inline more.
            for (Iterator i=cg.getAllMethods().iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method) i.next();
                m.getDeclaringClass().cls_initialize();
            }
            MethodInline mi = new MethodInline(cg);
            Navigator navigator = cg.getNavigator();
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
            cg.setRoots(roots);
            time = System.currentTimeMillis() - time;
            System.out.println("done. ("+time/1000.+" seconds)");
        }
        
        System.out.print("Calculating reachable methods...");
        time = System.currentTimeMillis();
        /* Calculate the reachable methods once to touch each method,
           so that the set of types are stable. */
        cg.calculateReachableMethods(roots);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        CSBDDPointerAnalysis dis = new CSBDDPointerAnalysis(cg);
        
        dis.go(roots);
        
        if (DUMP)
            dis.dumpResults();
        
    }
    
    void go(Collection roots) {
        
        long time = System.currentTimeMillis();
        
        /* Build SCCs. */
        Navigator navigator = cg.getNavigator();
        Set sccs = SCComponent.buildSCC(roots, navigator);
        SCCTopSortedGraph graph = SCCTopSortedGraph.topSort(sccs);
        
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        
        /* Walk through SCCs in reverse order. */
        SCComponent scc = graph.getLast();
        while (scc != null) {
            if (TRACE_WORKLIST) System.out.println("Visiting SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)"));
            Object[] nodes = scc.nodes();
            boolean change = false;
            for (int i=0; i<nodes.length; ++i) {
                jq_Method m = (jq_Method) nodes[i];
                System.out.print(Strings.left("SCC"+scc.getId()+" node "+(i+1)+"/"+nodes.length+": "+m.getDeclaringClass().shortName()+"."+m.getName()+"() "+variableIndexMap.size()+" vars", 78));
                if (TRACE_WORKLIST) System.out.println();
                else System.out.print("\r");
                if (m.getBytecode() == null) continue;
                long time2 = System.currentTimeMillis();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                /* Get the cached summary for this method. */
                BDDMethodSummary s = (BDDMethodSummary) bddSummaries.get(ms);
                if (s == null) {
                    /* Not yet visited, build a new summary. */
                    if (TRACE_WORKLIST) System.out.println("Building a new summary for "+m);
                    bddSummaries.put(ms, s = new BDDMethodSummary(ms));
                    if (TRACE_SUMMARIES) System.out.println(s.toString());
                    change = true;
                } else {
                    if (TRACE_WORKLIST) System.out.println("Using existing summary for "+m);
                }
                if (s.doCallees()) {
                    change = true;
                }
                if (change && !scc.isLoop()) {
                    s.trim();
                }
                time2 = System.currentTimeMillis() - time2;
                if (TRACE_TIMES || time2 > 1000) System.out.println("Total for "+m.getName()+"()="+(time2/1000.)+" seconds");
            }
            if (scc.isLoop() && change) {
                if (TRACE_WORKLIST) System.out.println("Loop changed, redoing SCC.");
            } else {
                if (TRACE_WORKLIST) System.out.println("Finished SCC"+scc.getId());
                Set entrypoints = scc.getEntrypoints(navigator);
                for (int i=0; i<nodes.length; ++i) {
                    jq_Method m = (jq_Method) nodes[i];
                    if (m.getBytecode() == null) continue;
                    if (!entrypoints.contains(m)) {
                        ControlFlowGraph cfg = CodeCache.getCode(m);
                        MethodSummary ms = MethodSummary.getSummary(cfg);
                        BDDMethodSummary s = (BDDMethodSummary) bddSummaries.get(ms);
                        if (TRACE_WORKLIST) System.out.println("Disposing internal summary for "+m.getName()+"()");
                        s.dispose();
                        bddSummaries.remove(s);
                    } else {
                        int number = 0;
                        for (Iterator j=navigator.prev(m).iterator(); j.hasNext(); ) {
                            Object p = j.next();
                            if (scc.contains(p)) continue;
                            ++number;
                        }
                        Assert._assert(number > 0);
                        sccEdges.put(m, new Integer(number));
                        if (TRACE_WORKLIST) System.out.println(m.getName()+" is an entrypoint with "+number+" predecessors.");
                    }
                }
                Set exitedges = scc.getExitEdges(navigator);
                for (Iterator i=exitedges.iterator(); i.hasNext(); ) {
                    Pair p = (Pair) i.next();
                    if (TRACE_WORKLIST) System.out.println("Looking at edge "+p);
                    jq_Method that = (jq_Method) p.get(1);
                    if (that.getBytecode() == null) continue;
                    int v = ((Integer) sccEdges.get(that)).intValue() - 1;
                    if (v == 0) {
                        ControlFlowGraph cfg = CodeCache.getCode(that);
                        MethodSummary ms = MethodSummary.getSummary(cfg);
                        BDDMethodSummary s = (BDDMethodSummary) bddSummaries.get(ms);
                        if (TRACE_WORKLIST) System.out.println("Disposing external summary for "+that.getName()+"()");
                        s.dispose();
                        bddSummaries.remove(s);
                        sccEdges.remove(that);
                    } else {
                        if (TRACE_WORKLIST) System.out.println("Method "+that.getName()+" still has "+v+" predecessors.");
                        sccEdges.put(that, new Integer(v));
                    }
                }
                scc = scc.prevTopSort();
            }
        }
    }
    
    Map sccEdges = new HashMap();
    
    void dumpResults() {
        // TODO.
    }
    
    /**
     * The default initial node count.  Smaller values save memory for
     * smaller problems, larger values save the time to grow the node tables
     * on larger problems.
     */
    public static final int DEFAULT_NODE_COUNT = 1000000;

    /**
     * The absolute maximum number of variables that we will ever use
     * in the BDD.  Smaller numbers will be more efficient, larger
     * numbers will allow larger programs to be analyzed.
     */
    public static final int DEFAULT_CACHE_SIZE = 100000;

    /**
     * Singleton BDD object that provides access to BDD functions.
     */
    private final BDDFactory bdd;
    
    /**
     * Initial call graph that we use to seed the analysis.
     */
    private final CallGraph cg;

    public static final int VARBITS = 20;
    public static final int HEAPBITS = 20;
    public static final int FIELDBITS = 13;
    public static final int CLASSBITS = 13;
    
    // the size of domains, can be changed to reflect the size of inputs
    int domainBits[] = {20, 20, 20, 13, 14, 14};
    // to be computed in sysInit function
    int domainSpos[] = {0,  0,  0,  0,  0,  0}; 
    
    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1, V2, V3, FD, H1, H2;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2, T3, T4; 

    // domain pairs for bdd_replace
    BDDPairing V1ToV2;
    BDDPairing V1ToV3;
    BDDPairing V2ToV1;
    BDDPairing V2ToV3;
    BDDPairing V3ToV1;
    BDDPairing V3ToV2;
    BDDPairing H1ToH2;
    BDDPairing H2ToH1;
    BDDPairing T2ToT1;

    // global BDDs
    BDD aC; // H1 x T2
    BDD vC; // V1 x T1
    BDD cC; // T1 x T2
    BDD typeFilter; // V1 x H1

    public CSBDDPointerAnalysis(CallGraph cg) {
        this(cg, DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
    
    public CSBDDPointerAnalysis(CallGraph cg, int nodeCount, int cacheSize) {
        this.cg = cg;
        
        bdd = BuDDyFactory.init(nodeCount, cacheSize);
        
        bdd.setCacheRatio(4);
        bdd.setMaxIncrease(cacheSize);
        
        int[] domains = new int[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1 << domainBits[i]);
        }
        BDDDomain[] bdd_domains = bdd.extDomain(domains);
        V1 = bdd_domains[0];
        V2 = bdd_domains[1];
        V3 = bdd_domains[2];
        FD = bdd_domains[3];
        H1 = bdd_domains[4];
        H2 = bdd_domains[5];
        T1 = V2;
        T2 = V1;
        T3 = H2;
        T4 = V2;
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "V3_FD_H2_V2xV1_H1");
        
        int[] varorder = makeVarOrdering(reverseLocal, ordering);
        for (int i=0; i<varorder.length; ++i) {
            //System.out.println("varorder["+i+"]="+varorder[i]);
        }
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
        
        V1ToV2 = bdd.makePair(V1, V2);
        V1ToV3 = bdd.makePair(V1, V3);
        V2ToV1 = bdd.makePair(V2, V1);
        V2ToV3 = bdd.makePair(V2, V3);
        V3ToV1 = bdd.makePair(V3, V1);
        V3ToV2 = bdd.makePair(V3, V2);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
        T2ToT1 = bdd.makePair(T2, T1);
        
        reset();
    }

    void reset() {
        aC = bdd.zero();
        vC = bdd.zero();
        cC = bdd.zero();
        typeFilter = bdd.zero();
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
            if (s.equals("V1")) d = V1;
            else if (s.equals("V2")) d = V2;
            else if (s.equals("V3")) d = V3;
            else if (s.equals("FD")) d = FD;
            else if (s.equals("H1")) d = H1;
            else if (s.equals("H2")) d = H2;
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
                if (bitNumber < domainBits[di])
                    varorder[bitIndex++] = domainSpos[di] + localOrders[di][bitNumber];
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
    
    boolean change;

    IndexMap getIndexMap(BDDDomain d) {
        if (d == V1 || d == V2) return variableIndexMap;
        if (d == FD) return fieldIndexMap;
        if (d == H1 || d == H2) return heapobjIndexMap;
        return null;
    }

    HashSet visitedMethods = new HashSet();

    HashMap callSiteToTargets = new HashMap();

    HashSet callGraphEdges = new HashSet();
    
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
        BDD site_bdd = H1.ithVar(site_i);
        BDD type_bdd = T2.ithVar(type_i);
        type_bdd.andWith(site_bdd);
        if (TRACE_TYPES) System.out.println("Adding alloc type: "+type_bdd.toStringWithDomains());
        aC.orWith(type_bdd);
    }

    public void addVarType(Node var, jq_Reference type) {
        addClassType(type);
        int var_i = getVariableIndex(var);
        int type_i = getTypeIndex(type);
        BDD var_bdd = V1.ithVar(var_i);
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
                BDD type2_bdd = T2.set();
                type1_bdd.andWith(type2_bdd);
                cC.orWith(type1_bdd);
                continue;
            }
            t1.prepare();
            int i2 = (i1 < last_typeIndex) ? last_typeIndex : 0;
            for ( ; i2<n1; ++i2) {
                jq_Type t2 = (jq_Type) typeIndexMap.get(i2);
                if (t2 == null) {
                    BDD type1_bdd = T1.set();
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
        
        BDD T1set = T1.set();
        BDD T2set = T2.set();
        // (T1 x T2) * (H1 x T2) => (T1 x H1)
        BDD assignableTypes = cC.relprod(aC, T2set);
        // (T1 x H1) * (V1 x T1) => (V1 x H1)
        typeFilter = assignableTypes.relprod(vC, T1set);
        T1set.free(); T2set.free();
        //cC.free(); vC.free(); aC.free();

        if (false) typeFilter = bdd.one();
    }

    jq_Reference getVariableType(int index) {
        Object o = variableIndexMap.get(index);
        while (o instanceof Triple) {
            Triple t = (Triple) o;
            int v = ((Integer) t.get(2)).intValue();
            o = variableIndexMap.get(v);
        }
        return (jq_Reference) ((Node) o).getDeclaredType();
    }
    
    int getNewVariableIndex(ProgramLocation mc, jq_Method callee, int p) {
        Object o = variableIndexMap.get(p);
        while (o instanceof Triple) {
            Triple t = (Triple) o;
            int v = ((Integer) t.get(2)).intValue();
            if (mc == t.get(0) && callee == t.get(1))
                return v;
            o = variableIndexMap.get(v);
        }
        return variableIndexMap.get(new Triple(mc, callee, new Integer(p)));
    }
    
    Map bddSummaries = new HashMap();
    BDDMethodSummary getBDDSummary(MethodSummary ms) {
        BDDMethodSummary result = (BDDMethodSummary) bddSummaries.get(ms);
        if (result == null) {
            if (TRACE_WORKLIST) System.out.println(" Recursive cycle? No summary for "+ms.getMethod());
            return null;
        }
        return result;
    }
    
    public class BDDMethodSummary {
        
        /** The method summary that we correspond to. */
        MethodSummary ms;
        
        BDD pointsTo;     // V1 x H1
        BDD edgeSet;      // V1 x V2
        BDD stores;       // V1 x (V2 x FD) 
        BDD loads;        // (V1 x FD) x V2

        /** Root set of locally-escaping nodes. (Parameter nodes, returned and thrown nodes.) */
        BDD roots; // V1
        
        /** Set of all locally-escaping nodes. */
        BDD nodes; // V1
        
        BDDMethodSummary(MethodSummary ms) {
            this.ms = ms;
            reset();
            computeInitial();
        }
        
        void reset() {
            // initialize relations to zero.
            pointsTo = bdd.zero();
            edgeSet = bdd.zero();
            stores = bdd.zero();
            loads = bdd.zero();
            roots = bdd.zero();
            nodes = bdd.zero();
        }
        
        void dispose() {
            pointsTo.free();
            edgeSet.free();
            stores.free();
            loads.free();
            roots.free();
            nodes.free();
        }
        
        void computeInitial() {
            long time = System.currentTimeMillis();
            // add edges for all local stuff.
            for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
                Node n = (Node) i.next();
                handleNode(n);
            }
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Converting method to BDD sets: "+(time/1000.));
            
            time = System.currentTimeMillis();
            // match up edges for local stuff.
            solveIncremental();
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Matching local edges: "+(time/1000.));
            
            time = System.currentTimeMillis();
            // calculate the set of things reachable from local stuff.
            transitiveClosure(nodes);
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Local transitive closure: "+(time/1000.));
        }
        
        public void handleNode(Node n) {
            
            if (n instanceof GlobalNode) {
                // TODO.
                return;
            }
            
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
                n instanceof ReturnedNode ||
                ms.getReturned().contains(n) ||
                ms.getThrown().contains(n)) {
                addLocalEscapeNode(n);
            }
            if (n.isPassedAsParameter()) {
                addNode(n);
            }
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }

        public void addObjectAllocation(Node dest, Node site) {
            int dest_i = getVariableIndex(dest);
            int site_i = getHeapobjIndex(site);
            BDD dest_bdd = V1.ithVar(dest_i);
            BDD site_bdd = H1.ithVar(site_i);
            dest_bdd.andWith(site_bdd);
            pointsTo.orWith(dest_bdd);
        }

        public void addEdge(Node dest, Node src) {
            addEdge(dest, Collections.singleton(src));
        }

        public void addEdge(Node dest, Set srcs) {
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V2.ithVar(dest_i);
            for (Iterator i=srcs.iterator(); i.hasNext(); ) {
                Node src = (Node) i.next();
                int src_i = getVariableIndex(src);
                BDD src_bdd = V1.ithVar(src_i);
                src_bdd.andWith(dest_bdd.id());
                edgeSet.orWith(src_bdd);
            }
            dest_bdd.free();
        }
        
        public void addLoad(Set dests, Node base, jq_Field f) {
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
                loads.orWith(dest_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
    
        public void addStore(Node base, jq_Field f, Set srcs) {
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
                stores.orWith(src_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
        
        public void addLocalEscapeNode(Node n) {
            int n_i = getVariableIndex(n);
            BDD n_bdd = V1.ithVar(n_i);
            roots.orWith(n_bdd.id());
            nodes.orWith(n_bdd);
        }
        
        public void addNode(Node n) {
            int n_i = getVariableIndex(n);
            BDD n_bdd = V1.ithVar(n_i);
            nodes.orWith(n_bdd);
        }
        
        public void solveNonincremental() {
            BDD oldPt1;

            calculateTypeFilter();
            
            if (TRACE_MATCHING) {
                System.out.println("Solving pointers for "+this.ms.getMethod());
            }

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
                    BDD newPt1 = edgeSet.relprod(pointsTo, V1.set());
                    // (V2 x H1) => (V1 x H1)
                    BDD newPt2 = newPt1.replace(V2ToV1);

                    /* --- apply type filtering and merge into pointsTo relation --- */
                    // (V1 x H1)
                    BDD newPt3 = newPt2.and(typeFilter);
                    if (TRACE_MATCHING) {
                        System.out.println("Removed by type filter: "+newPt2.apply(newPt3, BDDFactory.diff).toStringWithDomains());
                    }
                    // (V1 x H1)
                    pointsTo = pointsTo.or(newPt3);
                
                } while (!oldPt2.equals(pointsTo));

                if (TRACE_MATCHING) {
                    System.out.println("After transitive closure, points-to is "+pointsTo.toStringWithDomains());
                }
                
                // propagate points-to set over field loads and stores
                /* --- rule (2) --- */
                //
                //   o2 \in pt(l)   l -> q.f   o1 \in pt(q)
                // -----------------------------------------
                //                  o2 \in pt(o1.f) 
                // (V1 x (V2 x FD)) * (V1 x H1) => ((V2 x FD) x H1)
                BDD tmpRel1 = stores.relprod(pointsTo, V1.set());
                // ((V2 x FD) x H1) => ((V1 x FD) x H2)
                BDD tmpRel2 = tmpRel1.replace(V2ToV1).replace(H1ToH2);
                // ((V1 x FD) x H2) * (V1 x H1) => ((H1 x FD) x H2)
                BDD fieldPt = tmpRel2.relprod(pointsTo, V1.set());

                /* --- rule (3) --- */
                //
                //   p.f -> l   o1 \in pt(p)   o2 \in pt(o1)
                // -----------------------------------------
                //                 o2 \in pt(l)
                // ((V1 x FD) x V2) * (V1 x H1) => ((H1 x FD) x V2)
                BDD tmpRel3 = loads.relprod(pointsTo, V1.set());
                // ((H1 x FD) x V2) * ((H1 x FD) x H2) => (V2 x H2)
                BDD newPt4 = tmpRel3.relprod(fieldPt, H1.set().and(FD.set()));
                // (V2 x H2) => (V1 x H1)
                BDD newPt5 = newPt4.replace(V2ToV1).replace(H2ToH1);

                /* --- apply type filtering and merge into pointsTo relation --- */
                BDD newPt6 = newPt5.and(typeFilter);
                pointsTo = pointsTo.or(newPt6);

                if (TRACE_MATCHING) {
                    System.out.println("After matching loads/stores, points-to is now "+pointsTo.toStringWithDomains());
                }
            }
            while (!oldPt1.equals(pointsTo));

        }
        
        public void solveIncremental() {

            calculateTypeFilter();
            
            BDD empty = bdd.zero();
        
            BDD oldPointsTo = bdd.zero();
            BDD newPointsTo = pointsTo.id();
            BDD V1set = V1.set();
            BDD H1andFDset = H1.set();
            H1andFDset.andWith(FD.set());

            BDD fieldPt = bdd.zero();
            BDD storePt = bdd.zero();
            BDD loadAss = bdd.zero();

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
                    if (newPointsTo.equals(empty)) break;
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
                if (newPointsTo.equals(empty)) break;
                pointsTo.orWith(newPointsTo.id());
            }
        
            newPointsTo.free();
            empty.free();
            V1set.free(); H1andFDset.free();
            fieldPt.free();
            storePt.free();
            loadAss.free();
        }
        
        boolean transitiveClosure(BDD srcNodes) {
            BDD V2set, FDset;
            V2set = V2.set();
            FDset = FD.set();
            
            // Keep track of whether there was a change.
            boolean change = false;
            
            for (;;) {
                BDD oldNodes = srcNodes.id();
                BDD srcNodes2 = srcNodes.replace(V1ToV2);
                {
                    // Transitive along store edges.
                    // V2 x (V2xFD)xV1  =>  FDxV1
                    BDD newNodes = srcNodes2.relprod(stores, V2set);
                    // FDxV1  =>  V1
                    BDD newNodes2 = newNodes.exist(FDset);
                    newNodes.free();
                    srcNodes.orWith(newNodes2);
                }
                
                {
                    // Transitive along assignment edges.
                    // V2 x V1xV2  =>  V1
                    BDD newNodes = srcNodes2.relprod(edgeSet, V2set);
                    srcNodes2.free();
                    srcNodes.orWith(newNodes);
                }
                boolean done = oldNodes.equals(srcNodes);
                oldNodes.free();
                if (done) break;
                change = true;
            }
            
            V2set.free();
            FDset.free();
            return change;
        }
        
        public static final boolean RENUMBER = false;
        
        boolean doCallees() {
            BDD newEdges = bdd.zero();
            BDD newLoads = bdd.zero();
            BDD newStores = bdd.zero();
            
            // find all call sites.
            for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
                ProgramLocation mc = (ProgramLocation) i.next();
                if (TRACE_CALLEE) System.out.println("Visiting call site "+mc);
                
                // build up an array of BDD's corresponding to each of the
                // parameters passed into this method call.
                jq_Type[] paramTypes = mc.getParamTypes();
                BDD[] params = new BDD[paramTypes.length];
                for (int j=0; j<paramTypes.length; j++) {
                    jq_Type t = (jq_Type) paramTypes[j];
                    if (!(t instanceof jq_Reference)) continue;
                    PassedParameter pp = new PassedParameter(mc, j);
                    Set s = ms.getNodesThatCall(pp);
                    params[j] = bdd.zero();
                    for (Iterator k=s.iterator(); k.hasNext(); ) {
                        int m = getVariableIndex((Node) k.next());
                        params[j].orWith(V1.ithVar(m));
                    }
                    if (TRACE_CALLEE) System.out.println("Params["+j+"]="+params[j].toStringWithDomains());
                }
                
                // find all targets of this call.
                Collection targets = cg.getTargetMethods(mc);
                for (Iterator j=targets.iterator(); j.hasNext(); ) {
                    jq_Method target = (jq_Method) j.next();
                    if (TRACE_CALLEE) System.out.print("Target "+target);
                    if (target.getBytecode() == null) {
                        // TODO: calls to native methods.
                        if (TRACE_CALLEE) System.out.println("... native method!");
                        continue;
                    }
                    ControlFlowGraph cfg = CodeCache.getCode(target);
                    MethodSummary ms_callee = MethodSummary.getSummary(cfg);
                    BDDMethodSummary callee = getBDDSummary(ms_callee);
                    if (callee == null) {
                        if (TRACE_CALLEE) System.out.println("... no BDD summary yet!");
                        continue;
                    }
                    
                    // renumber if there is any overlap in node numbers.
                    BDD renumbering13 = null;
                    BDD renumbering23 = null;
                    if (RENUMBER) {
                        BDD overlap = nodes.and(callee.nodes);
                        if (!overlap.equals(bdd.zero())) {
                            if (TRACE_OVERLAP) System.out.println("... non-zero overlap! "+overlap.toStringWithDomains());
                            long time = System.currentTimeMillis();
                            BDD callee_used = callee.nodes.id();
                            renumbering13 = bdd.zero();
                            for (;;) {
                                int p = callee_used.scanVar(V1);
                                if (p < 0) break;
                                BDD pth = V1.ithVar(p);
                                int q;
                                if (nodes.and(pth).equals(bdd.zero())) {
                                    q = p;
                                } else {
                                    q = getNewVariableIndex(mc, target, p);
                                    if (TRACE_OVERLAP) System.out.println("Variable "+p+" overlaps, new variable index "+q);
                                    jq_Reference type = getVariableType(p);
                                    BDD var_bdd = V1.ithVar(q);
                                    BDD type_bdd = T1.ithVar(getTypeIndex(type));
                                    type_bdd.andWith(var_bdd);
                                    if (TRACE_TYPES) System.out.println("Adding var type: "+type_bdd.toStringWithDomains());
                                    vC.orWith(type_bdd);
                                }
                                BDD qth = V3.ithVar(q);
                                qth.andWith(pth.id());
                                renumbering13.orWith(qth);
                                callee_used.applyWith(pth, BDDFactory.diff);
                            }
                            renumbering23 = renumbering13.replace(V1ToV2);
                            time = System.currentTimeMillis() - time;
                            if (TRACE_TIMES || time > 400) System.out.println("Build renumbering: "+(time/1000.));
                        } else {
                            if (TRACE_CALLEE) System.out.println("...zero overlap!");
                        }
                        overlap.free();
                    }
                    long time = System.currentTimeMillis();
                    BDD callee_loads = renumber(callee.loads, renumbering13, V1.set(), V3ToV1, renumbering23, V2.set(), V3ToV2);
                    BDD callee_stores = renumber(callee.stores, renumbering13, V1.set(), V3ToV1, renumbering23, V2.set(), V3ToV2);
                    BDD callee_edges = renumber(callee.edgeSet, renumbering13, V1.set(), V3ToV1, renumbering23, V2.set(), V3ToV2);
                    BDD callee_nodes = renumber(callee.nodes, renumbering13, V1.set(), V3ToV1);
                    time = System.currentTimeMillis() - time;
                    if (TRACE_TIMES || time > 400) System.out.println("Renumbering: "+(time/1000.));
                    
                    if (TRACE_CALLEE) { 
                        System.out.println("New loads: "+callee_loads.toStringWithDomains());
                        System.out.println("New stores: "+callee_stores.toStringWithDomains());
                        System.out.println("New edges: "+callee_edges.toStringWithDomains());
                    }
                    
                    // incorporate callee operations into caller.
                    newLoads.orWith(callee_loads);
                    newStores.orWith(callee_stores);
                    newEdges.orWith(callee_edges);
                    nodes.orWith(callee_nodes);
                    
                    // add edges for parameters.
                    for (int k=0; k<callee.ms.getNumOfParams(); ++k) {
                        ParamNode pn = callee.ms.getParamNode(k);
                        if (pn == null) continue;
                        int pnIndex = getVariableIndex(pn);
                        BDD tmp = V2.ithVar(pnIndex);
                        BDD paramEdge = renumber(tmp, renumbering23, V2.set(), V3ToV2);
                        tmp.free();
                        paramEdge.andWith(params[k].id());
                        if (TRACE_CALLEE) System.out.println("Param#"+k+" edges "+paramEdge.toStringWithDomains());
                        newEdges.orWith(paramEdge);
                    }
                    
                    // add edges for return value, if one exists.
                    if (((jq_Method)callee.ms.getMethod()).getReturnType().isReferenceType() &&
                        !callee.ms.getReturned().isEmpty()) {
                        ReturnedNode rvn = (ReturnValueNode) ms.getRVN(mc);
                        if (rvn != null) {
                            BDD retVal = bdd.zero();
                            for (Iterator k=callee.ms.getReturned().iterator(); k.hasNext(); ) {
                                int nIndex = getVariableIndex((Node) k.next());
                                BDD tmp = V1.ithVar(nIndex);
                                retVal.orWith(renumber(tmp, renumbering13, V1.set(), V3ToV1));
                                tmp.free();
                            }
                            int rIndex = getVariableIndex(rvn);
                            retVal.andWith(V2.ithVar(rIndex));
                            if (TRACE_CALLEE) System.out.println("Return value edges "+retVal.toStringWithDomains());
                            newEdges.orWith(retVal);
                        }
                    }
                    // add edges for thrown exception, if one exists.
                    if (!callee.ms.getThrown().isEmpty()) {
                        ReturnedNode rvn = (ThrownExceptionNode) ms.getTEN(mc);
                        if (rvn != null) {
                            BDD retVal = bdd.zero();
                            for (Iterator k=callee.ms.getThrown().iterator(); k.hasNext(); ) {
                                int nIndex = getVariableIndex((Node) k.next());
                                BDD tmp = V1.ithVar(nIndex);
                                retVal.orWith(renumber(tmp, renumbering13, V1.set(), V3ToV1));
                                tmp.free();
                            }
                            int rIndex = getVariableIndex(rvn);
                            retVal.andWith(V2.ithVar(rIndex));
                            if (TRACE_CALLEE) System.out.println("Thrown exception edges "+retVal.toStringWithDomains());
                            newEdges.orWith(retVal);
                        }
                    }
                    
                    if (renumbering13 != null) {
                        renumbering13.free();
                        renumbering23.free();
                    }
                }
                for (int j=0; j<paramTypes.length; ++j) {
                    if (params[j] != null)
                        params[j].free();
                }
            }
            
            long time = System.currentTimeMillis();
            BDD newerEdges = matchNewLoadsAndStores(newLoads, newStores);
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Matching new loads and stores: "+(time/1000.));
            
            newEdges.orWith(newerEdges);
            
            time = System.currentTimeMillis();
            boolean b2 = matchNewEdges(newEdges);
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Matching new edges: "+(time/1000.));
            
            return b2;
        }
        
        public void handleNativeCall(MethodSummary caller, ProgramLocation mc) {
            // only handle return value for now.
            jq_Type at = mc.getTargetMethod().getReturnType();
            if (at instanceof jq_Reference) {
                jq_Reference t = (jq_Reference) at;
                ReturnedNode rvn = (ReturnedNode) caller.getRVN(mc);
                if (rvn == null) return;
                UnknownTypeNode utn = UnknownTypeNode.get((jq_Reference) t);
                addObjectAllocation(utn, utn);
                addAllocType(utn, (jq_Reference) t);
                addVarType(utn, (jq_Reference) t);
                addEdge(rvn, utn);
            }
        }
        
        BDD renumber(BDD src, BDD renumbering_ac, BDD Aset, BDDPairing CtoA) {
            if (renumbering_ac == null) return src.id();
            BDD t1;
            t1 = src.relprod(renumbering_ac, Aset);
            Aset.free();
            t1.replaceWith(CtoA);
            return t1;
        }
        
        BDD renumber(BDD src, BDD renumbering_ac, BDD Aset, BDDPairing CtoA, BDD renumbering_bc, BDD Bset, BDDPairing CtoB) {
            if (renumbering_ac == null) return src.id();
            BDD t1, t2;
            t1 = src.relprod(renumbering_ac, Aset);
            Aset.free();
            t1.replaceWith(CtoA);
            t2 = t1.relprod(renumbering_bc, Bset);
            t1.free(); Bset.free();
            t2.replaceWith(CtoB);
            return t2;
        }
        
        void trim() {
            // recalculate reachable nodes.
            nodes = roots.id();
            transitiveClosure(nodes);
            
            // trim the stuff that doesn't escape.
            trim(nodes);
        }
        
        void trim(BDD set) {
            if (TRACE_TRIMMING) {
                System.out.println("Trimming edges outside of the set "+set.toStringWithDomains());
                System.out.println("Before: edges="+edgeSet.nodeCount()+
                                          " loads="+loads.nodeCount()+
                                         " stores="+stores.nodeCount()+
                                       " pointsTo="+pointsTo.nodeCount());
            }
            
            BDD h1 = H1.domain();
            BDD v1xh1 = set.and(h1);
            pointsTo.andWith(v1xh1);
            
            BDD v2_set = set.replace(V1ToV2);
            BDD v1xv2 = set.and(v2_set);
            edgeSet.andWith(v1xv2);

            BDD v1xv2xfd = set.and(v2_set);
            v1xv2xfd.andWith(FD.domain());
            loads.andWith(v1xv2xfd.id());
            stores.andWith(v1xv2xfd);
            
            h1.free();
            v2_set.free();

            if (TRACE_TRIMMING) {
                System.out.println("After: edges="+edgeSet.nodeCount()+
                                         " loads="+loads.nodeCount()+
                                        " stores="+stores.nodeCount()+
                                      " pointsTo="+pointsTo.nodeCount());
            }
        }
        
        BDD matchNewLoadsAndStores(BDD newLoads, BDD newStores) {
            // TODO.
            loads.orWith(newLoads);
            stores.orWith(newStores);
            return bdd.zero();
        }
        
        boolean matchNewEdges(BDD newEdges) {
            // TODO.
            boolean b;
            BDD oldEdges = edgeSet.id();
            edgeSet.orWith(newEdges);
            solveIncremental();
            b = !oldEdges.equals(edgeSet);
            oldEdges.free();
            return b;
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("BDD Summary for ");
            sb.append(ms.getMethod());
            sb.append(':');
            sb.append(Strings.lineSep);
            sb.append("Roots=");
            sb.append(roots.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Nodes=");
            System.out.println("Nodes: "+nodes.toStringWithDomains());
            sb.append(nodes.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Loads=");
            sb.append(loads.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Stores=");
            sb.append(stores.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Edges=");
            sb.append(edgeSet.toStringWithDomains());
            sb.append(Strings.lineSep);
            sb.append("Points-to=");
            sb.append(pointsTo.toStringWithDomains());
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
            return index;
        }
        
        public String toString() {
            return name;
        }
        
    }
}

// FullyContextSensitiveBDD.java, created Mon Apr 21 13:49:22 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
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
import Util.Assert;
import Util.Strings;
import Util.Collections.IndexMap;
import Util.Collections.Triple;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class FullyContextSensitiveBDD {

    public static final boolean TRACE_ALL = false;

    public static final boolean TRACE_WORKLIST  = false || TRACE_ALL;
    public static final boolean TRACE_SUMMARIES = false || TRACE_ALL;
    public static final boolean TRACE_CALLEE    = false || TRACE_ALL;
    public static final boolean TRACE_OVERLAP   = false || TRACE_ALL;
    public static final boolean TRACE_MATCHING  = false || TRACE_ALL;
    public static final boolean TRACE_TRIMMING  = false || TRACE_ALL;
    public static final boolean TRACE_TIMES     = false || TRACE_ALL;

    public static final boolean USE_CHA = true;

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
        
        System.out.print("Calculating reachable methods...");
        time = System.currentTimeMillis();
        /* Calculate the reachable methods once to touch each method,
           so that the set of types are stable. */
        cg.calculateReachableMethods(roots);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+time/1000.+" seconds)");
        
        FullyContextSensitiveBDD dis = new FullyContextSensitiveBDD(cg, DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
        
        dis.go(roots);
        
        if (DUMP)
            dis.dumpResults();
        
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
    
    // the size of domains, can be changed to reflect the size of inputs
    int domainBits[] = {20, 20, 20, 20, 13};
    int domainSpos[] = {0,  0,  0,  0,  0 };
    BDDDomain           V1, V2, V3, V4, FD;

    BDDPairing V1toV2;
    BDDPairing V2toV1;
    BDDPairing V2toV3;
    BDDPairing V2toV4;
    BDDPairing V3toV2;
    BDDPairing V3toV4;
    BDDPairing V4toV1;
    BDDPairing V4toV2;
    BDDPairing V4toV3;
    BDDPairing V1V3toV2V4;
    
    public FullyContextSensitiveBDD() {
        this(CHACallGraph.INSTANCE, DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
    
    public FullyContextSensitiveBDD(CallGraph cg, int nodeCount, int cacheSize) {
        this.cg = cg;
        
        bdd = BuDDyFactory.init(nodeCount, cacheSize);
        //bdd.setCacheRatio(4);
        //bdd.setMaxIncrease(cacheSize);
        
        initialize();
    }
    
    void initialize() {
        int[] domains = new int[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1 << domainBits[i]);
        }
        BDDDomain[] bdd_domains = bdd.extDomain(domains);
        V1 = bdd_domains[0];
        V2 = bdd_domains[1];
        V3 = bdd_domains[2];
        V4 = bdd_domains[3];
        FD = bdd_domains[4];
        
        int[] varorder = new int[bdd.varNum()];
        makeVarOrdering(varorder);
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
        
        V1toV2 = bdd.makePair(V1, V2);
        V2toV1 = bdd.makePair(V2, V1);
        V2toV3 = bdd.makePair(V2, V3);
        V2toV4 = bdd.makePair(V2, V4);
        V3toV2 = bdd.makePair(V3, V2);
        V3toV4 = bdd.makePair(V3, V4);
        V4toV1 = bdd.makePair(V4, V1);
        V4toV2 = bdd.makePair(V4, V2);
        V4toV3 = bdd.makePair(V4, V3);
        V1V3toV2V4 = bdd.makePair();
        V1V3toV2V4.set(new BDDDomain[] {V1, V3}, new BDDDomain[] {V2, V4} );
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
            if (TRACE_WORKLIST) System.out.println("Visiting SCC"+scc.getId());
            Object[] nodes = scc.nodes();
            boolean change = false;
            for (int i=0; i<nodes.length; ++i) {
                jq_Method m = (jq_Method) nodes[i];
                System.out.print(Strings.left("SCC"+scc.getId()+" node "+(i+1)+"/"+nodes.length+": "+m.getDeclaringClass().shortName()+"."+m.getName()+"() "+variableIndexMap.size()+" vars", 78));
                if (TRACE_WORKLIST) System.out.println();
                else System.out.print("\r");
                if (m.getBytecode() == null) continue;
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
                if (s.visit()) {
                    change = true;
                }
                if (change && !scc.isLoop()) {
                    s.trim();
                }
            }
            if (scc.isLoop() && change) {
                if (TRACE_WORKLIST) System.out.println("Loop changed, redoing SCC.");
            } else {
                scc = scc.prevTopSort();
            }
        }
    }
    
    void dumpResults() {
        // TODO.
    }

    IndexMap/* Node->index */ variableIndexMap = new IndexMap("Variable");
    IndexMap/* Node->index */ heapobjIndexMap = new IndexMap("HeapObj");
    IndexMap/* jq_Field->index */ fieldIndexMap = new IndexMap("Field");

    int getVariableIndex(Node dest) {
        return variableIndexMap.get(dest);
    }
    int getHeapobjIndex(Node site) {
        return heapobjIndexMap.get(site);
    }
    int getFieldIndex(jq_Field f) {
        return fieldIndexMap.get(f);
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
        
        /** Root set of locally-escaping nodes. (Parameter nodes, returned and thrown nodes.) */
        BDD roots; // V2
        
        /** Set of all locally-escaping nodes. */
        BDD nodes; // V2
        
        /** Locally-escaping load operations within this method (and its callees). */
        BDD loads;  // V1x(V2xFD)   v1=v2.fd;
        
        /** Locally-escaping store operations within this method (and its callees). */
        BDD stores; // (V2xFD)xV3   v2.fd=v3;
        
        /** Locally-escaping assignments within this method (and its callees). */
        BDD edges;  // V1xV3        v1=v3;
        BDD edges24;  // V2xV4        v2=v4;
        
        BDDMethodSummary(MethodSummary ms) {
            this.ms = ms;
            initialize();
        }
        
        void initialize() {
            roots = bdd.zero();
            nodes = bdd.zero();
            loads = bdd.zero();
            stores = bdd.zero();
            edges = bdd.zero();
            edges24 = bdd.zero();
            
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
            matchEdges2();
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Matching local edges: "+(time/1000.));
            
            time = System.currentTimeMillis();
            // calculate the set of things reachable from local stuff.
            transitiveClosure(nodes);
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Local transitive closure: "+(time/1000.));
            
        }
        
        boolean visit() {
            boolean change = false;
            
            // add edges for the effects of all callee methods.
            long time = System.currentTimeMillis();
            if (doCallees2()) change = true;
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Doing callees on visit: "+(time/1000.));
            
            return change;
        }
        
        void trim() {
            // recalculate reachable nodes.
            nodes = roots.id();
            transitiveClosure(nodes);
            
            // trim the stuff that doesn't escape.
            trim(nodes);
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("BDD Summary for ");
            sb.append(ms.getMethod());
            sb.append(':');
            sb.append(Strings.lineSep);
            sb.append("Roots=");
            sb.append(roots.toStringWithDomains(ts));
            sb.append(Strings.lineSep);
            sb.append("Nodes=");
            System.out.println("Nodes: "+nodes.toStringWithDomains());
            sb.append(nodes.toStringWithDomains(ts));
            sb.append(Strings.lineSep);
            sb.append("Loads=");
            sb.append(loads.toStringWithDomains(ts));
            sb.append(Strings.lineSep);
            sb.append("Stores=");
            sb.append(stores.toStringWithDomains(ts));
            sb.append(Strings.lineSep);
            sb.append("Edges=");
            sb.append(edges.toStringWithDomains(ts));
            sb.append(Strings.lineSep);
            return sb.toString();
        }
        
        void free() {
            roots.free(); roots = null;
            nodes.free(); nodes = null;
            loads.free(); loads = null;
            stores.free(); stores = null;
            edges.free(); edges = null;
            edges24.free(); edges24 = null;
        }
        
        boolean doCallees2() {
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
                        params[j].orWith(V3.ithVar(m));
                    }
                    if (TRACE_CALLEE) System.out.println("Params["+j+"]="+params[j].toStringWithDomains(ts));
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
                    BDD overlap = nodes.and(callee.nodes);
                    BDD renumbering24 = null;
                    BDD renumbering14 = null;
                    BDD renumbering34 = null;
                    if (!overlap.equals(bdd.zero())) {
                        if (TRACE_OVERLAP) System.out.println("... non-zero overlap! "+overlap.toStringWithDomains(ts));
                        long time = System.currentTimeMillis();
                        BDD callee_used = callee.nodes.id();
                        renumbering24 = bdd.zero();
                        for (;;) {
                            int p = callee_used.scanVar(V2);
                            if (p < 0) break;
                            BDD pth = V2.ithVar(p);
                            int q;
                            if (nodes.and(pth).equals(bdd.zero())) {
                                q = p;
                            } else {
                                q = getNewVariableIndex(mc, target, p);
                                if (TRACE_OVERLAP) System.out.println("Variable "+p+" overlaps, new variable index "+q);
                            }
                            BDD qth = V4.ithVar(q);
                            qth.andWith(pth.id());
                            renumbering24.orWith(qth);
                            callee_used.applyWith(pth, BDDFactory.diff);
                        }
                        renumbering14 = renumbering24.replace(V2toV1);
                        renumbering34 = renumbering24.replace(V2toV3);
                        time = System.currentTimeMillis() - time;
                        if (TRACE_TIMES || time > 400) System.out.println("Build renumbering: "+(time/1000.));
                    } else {
                        if (TRACE_CALLEE) System.out.println("...zero overlap!");
                    }
                    
                    overlap.free();
                    long time = System.currentTimeMillis();
                    BDD callee_loads = renumber(callee.loads, renumbering14, V1.set(), V4toV1, renumbering24, V2.set(), V4toV2);
                    BDD callee_stores = renumber(callee.stores, renumbering24, V2.set(), V4toV2, renumbering34, V3.set(), V4toV3);
                    BDD callee_edges = renumber(callee.edges, renumbering14, V1.set(), V4toV1, renumbering34, V3.set(), V4toV3);
                    BDD callee_nodes = renumber(callee.nodes, renumbering24, V2.set(), V4toV2);
                    time = System.currentTimeMillis() - time;
                    if (TRACE_TIMES || time > 400) System.out.println("Renumbering: "+(time/1000.));
                    
                    if (TRACE_CALLEE) { 
                        System.out.println("New loads: "+callee_loads.toStringWithDomains(ts));
                        System.out.println("New stores: "+callee_stores.toStringWithDomains(ts));
                        System.out.println("New edges: "+callee_edges.toStringWithDomains(ts));
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
                        BDD tmp = V1.ithVar(pnIndex);
                        BDD paramEdge = renumber(tmp, renumbering14, V1.set(), V4toV1);
                        tmp.free();
                        paramEdge.andWith(params[k].id());
                        if (TRACE_CALLEE) System.out.println("Param#"+k+" edges "+paramEdge.toStringWithDomains(ts));
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
                                BDD tmp = V3.ithVar(nIndex);
                                retVal.orWith(renumber(tmp, renumbering34, V3.set(), V4toV3));
                                tmp.free();
                            }
                            int rIndex = getVariableIndex(rvn);
                            retVal.andWith(V1.ithVar(rIndex));
                            if (TRACE_CALLEE) System.out.println("Return value edges "+retVal.toStringWithDomains(ts));
                            newEdges.orWith(retVal);
                        }
                    }
                    // add edges for thrown exception, if one exists.
                    if (!callee.ms.getThrown().isEmpty()) {
                        ReturnedNode rvn = (ThrownExceptionNode) ms.getTEN(mc);
                        if (rvn != null) {
                            BDD retVal = bdd.zero();
                            for (Iterator k=callee.ms.getReturned().iterator(); k.hasNext(); ) {
                                int nIndex = getVariableIndex((Node) k.next());
                                BDD tmp = V3.ithVar(nIndex);
                                retVal.orWith(renumber(tmp, renumbering34, V3.set(), V4toV3));
                                tmp.free();
                            }
                            int rIndex = getVariableIndex(rvn);
                            retVal.andWith(V1.ithVar(rIndex));
                            if (TRACE_CALLEE) System.out.println("Thrown exception edges "+retVal.toStringWithDomains(ts));
                            newEdges.orWith(retVal);
                        }
                    }
                    
                    if (renumbering14 != null) {
                        renumbering14.free();
                        renumbering24.free();
                        renumbering34.free();
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
            //BDD newEdges24 = newEdges.replace(V1V3toV2V4);
            
            time = System.currentTimeMillis();
            boolean b2 = matchNewEdges(newEdges);
            time = System.currentTimeMillis() - time;
            if (TRACE_TIMES || time > 400) System.out.println("Matching new edges: "+(time/1000.));
            
            return b2;
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
            t1 = src.relprod(renumbering_ac, Aset); // 5%
            Aset.free();
            t1.replaceWith(CtoA); // 4%
            t2 = t1.relprod(renumbering_bc, Bset); // 6%
            t1.free(); Bset.free();
            t2.replaceWith(CtoB); // 3%
            return t2;
        }
        
        void handleNode(Node n) {
            
            if (n instanceof GlobalNode) {
                // TODO.
                return;
            }
            
            // add inclusion edge from a node to itself.
            addEdge(n, n);
            
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
        
        public void addLoad(Set dests, Node base, jq_Field f) {
            int base_i = getVariableIndex(base);
            int f_i = getFieldIndex(f);
            BDD base_bdd = V2.ithVar(base_i);
            BDD f_bdd = FD.ithVar(f_i);
            for (Iterator i=dests.iterator(); i.hasNext(); ) {
                FieldNode dest = (FieldNode) i.next();
                int dest_i = getVariableIndex(dest);
                BDD dest_bdd = V1.ithVar(dest_i);
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
                BDD src_bdd = V3.ithVar(src_i);
                src_bdd.andWith(f_bdd.id());
                src_bdd.andWith(base_bdd.id());
                stores.orWith(src_bdd);
            }
            base_bdd.free(); f_bdd.free();
        }
        
        public void addAllocType(Node base, jq_Reference type) {
            // TODO.
        }
        
        public void addVarType(Node base, jq_Reference type) {
            // TODO.
        }
        
        public void addObjectAllocation(Node dest, Node site) {
            // TODO.
        }
            
        public void addLocalEscapeNode(Node n) {
            int n_i = getVariableIndex(n);
            BDD n_bdd = V2.ithVar(n_i);
            roots.orWith(n_bdd.id());
            nodes.orWith(n_bdd);
        }
        
        public void addNode(Node n) {
            int n_i = getVariableIndex(n);
            BDD n_bdd = V2.ithVar(n_i);
            nodes.orWith(n_bdd);
        }
        
        public void addEdge(Node n1, Node n2) {
            int n1_i = getVariableIndex(n1);
            BDD n1_bdd = V1.ithVar(n1_i);
            int n2_i = getVariableIndex(n2);
            BDD n2_bdd = V3.ithVar(n2_i);
            n1_bdd.andWith(n2_bdd);
            edges24.orWith(n1_bdd.replace(V1V3toV2V4));
            edges.orWith(n1_bdd);
        }
        
        boolean matchEdges() {
            if (TRACE_MATCHING) System.out.println("Matching edges...");
            
            BDD V2set, V4andFDset;
            V2set = V2.set();
            V4andFDset = V4.set(); V4andFDset.andWith(FD.set());
            
            // Keep track of whether there was a change.
            boolean change = false;
            
            // Make a copy of edges that goes V2xV4.
            BDD edges24 = edges.replace(V1V3toV2V4);
            for (;;) {
                
                // Repeat for transitive closure.
                for (;;) {
                    BDD edges12 = edges.replace(V3toV2);
                    BDD edges23 = edges24.replace(V4toV3);
                    // Transitive closure.
                    // V1xV2 , V2xV3                        v1=v2; v2=v3;
                    // -------------                        -------------
                    //     V1xV3                                v1=v3;
                    BDD newEdges = edges12.relprod(edges23, V2set);
                    edges12.free(); edges23.free();
                    edges.orWith(newEdges);
                    
                    // Check for termination.
                    BDD newEdges24 = edges.replace(V1V3toV2V4);
                    boolean done = edges24.equals(newEdges24);
                    if (TRACE_MATCHING && !done) {
                        System.out.println("New edges: "+newEdges24.apply(edges24, BDDFactory.diff).toStringWithDomains(ts));
                    }
                    edges24.free();
                    edges24 = newEdges24;
                    if (done) break;
                    change = true;
                }
                
                // Matching rules:
                
                // Propagate loads to their children.
                // V1x(V2xFD) , V2xV4                   v1=v2.fd; v2=v4;
                // ------------------                   ----------------
                //     V1x(V4xFD)                           v1=v4.fd;
                BDD tmpRel1 = loads.relprod(edges24, V2set);
                // Temporarily make loads into V1x(V4xFD)
                loads.replaceWith(V2toV4);
                // Add result to loads.
                loads.orWith(tmpRel1);
                
                // Match stores to children's loads.
                // (V2xFD)xV3 , V2xV4 , V1x(V4xFD)      v2.fd=v3; v2=v4; v1=v4.fd; 
                // ---------------------------------    --------------------------
                //              V1xV3                             v1=v3;
                BDD tmpRel2 = stores.relprod(edges24, V2set);
                BDD newEdges = tmpRel2.relprod(loads, V4andFDset);
                tmpRel2.free(); // Free tmpRel2, as it is no longer used.
                // Add result to edges.
                edges.orWith(newEdges);
                
                // Change loads back into V1x(V2xFD)
                loads.replaceWith(V4toV2);
                
                // Check if any edges were added.
                BDD newEdges24 = edges.replace(V1V3toV2V4);
                boolean done = edges24.equals(newEdges24);
                if (TRACE_MATCHING && !done) {
                    System.out.println("New edges: "+newEdges24.apply(edges24, BDDFactory.diff).toStringWithDomains(ts));
                }
                edges24.free();
                edges24 = newEdges24;
                if (done) break;
                change = true;
            }
            edges24.free();
            V2set.free();
            V4andFDset.free();
            if (TRACE_MATCHING) System.out.println("Done matching edges.");
            return change;
        }
        
        boolean matchEdges2() {
            if (TRACE_MATCHING) System.out.println("Matching edges...");
            
            BDD V2set, V4set, V2andFDset;
            V2set = V2.set();
            V4set = V4.set();
            V2andFDset = V2.set(); V2andFDset.andWith(FD.set());
            
            // Keep track of whether there was a change.
            boolean change = false;
            
            for (;;) {
                
                // Repeat for transitive closure.
                for (;;) {
                    BDD edges12 = edges.replace(V3toV2);
                    BDD edges23 = edges24.replace(V4toV3);
                    // Transitive closure.
                    // V1xV2 , V2xV3                        v1=v2; v2=v3;
                    // -------------                        -------------
                    //     V1xV3                                v1=v3;
                    BDD newEdges = edges12.relprod(edges23, V2set);
                    edges12.free(); edges23.free();
                    edges.orWith(newEdges);
                    
                    // Check for termination.
                    BDD newEdges24 = edges.replace(V1V3toV2V4);
                    boolean done = edges24.equals(newEdges24);
                    if (TRACE_MATCHING && !done) {
                        System.out.println("New edges: "+newEdges24.apply(edges24, BDDFactory.diff).toStringWithDomains(ts));
                    }
                    edges24.free();
                    edges24 = newEdges24;
                    if (done) break;
                    change = true;
                }
                
                // Matching rules:
                
                // Match loads and stores.
                // V1x(V2xFD) , V2xV4 , V3xV4 , (V3xFD)xV5   v1=v2.fd; v2=v4; v3=v4; v3.fd=v5;
                // ---------------------------------------   --------------------------------
                //               V1x(V4xFD)                               v1=v5;
                BDD tmpRel1 = loads.relprod(edges24, V2set);
                BDD tmpRel2 = tmpRel1.relprod(edges24, V4set);
                tmpRel1.free();
                BDD tmpRel3 = tmpRel2.relprod(stores, V2andFDset);
                tmpRel2.free();
                edges.orWith(tmpRel3);
                
                // Check if any edges were added.
                BDD newEdges24 = edges.replace(V1V3toV2V4);
                boolean done = edges24.equals(newEdges24);
                if (TRACE_MATCHING && !done) {
                    System.out.println("New edges: "+newEdges24.apply(edges24, BDDFactory.diff).toStringWithDomains(ts));
                }
                edges24.free();
                edges24 = newEdges24;
                if (done) break;
                change = true;
            }
            V2set.free();
            V4set.free();
            V2andFDset.free();
            if (TRACE_MATCHING) System.out.println("Done matching edges.");
            return change;
        }
        
        BDD matchNewLoadsAndStores(BDD newLoads, BDD newStores) {
            if (newLoads.isZero() && newStores.isZero())
                return bdd.zero();
            
            if (TRACE_MATCHING) System.out.println("Matching new loads and stores...");
            
            // subtract out loads/stores that already exist.
            newLoads.applyWith(loads.id(), BDDFactory.diff);
            newStores.applyWith(stores.id(), BDDFactory.diff);
            
            if (newLoads.isZero() && newStores.isZero())
                return bdd.zero();
            
            loads.orWith(newLoads.id());
            stores.orWith(newStores.id());
            
            BDD V2set = V2.set();
            BDD V4set = V4.set();
            BDD V2andFDset = V2.set(); V2andFDset.andWith(FD.set());
            
            BDD newerEdges = bdd.zero();
            
            if (!newLoads.isZero()) {
                if (TRACE_MATCHING) {
                    System.out.println("New loads: "+newLoads.toStringWithDomains(ts));
                }
                
                BDD tmpRel1 = newLoads.relprod(edges24, V2set);
                BDD tmpRel2 = tmpRel1.relprod(edges24, V4set);
                tmpRel1.free();
                BDD tmpRel3 = tmpRel2.relprod(stores, V2andFDset);
                tmpRel2.free();
                newerEdges.orWith(tmpRel3);
            }
            if (!newStores.isZero()) {
                if (TRACE_MATCHING) {
                    System.out.println("New stores: "+newStores.toStringWithDomains(ts));
                }
                
                BDD tmpRel1 = newStores.relprod(edges24, V2set);
                BDD tmpRel2 = tmpRel1.relprod(edges24, V4set);
                tmpRel1.free();
                BDD tmpRel3 = tmpRel2.relprod(loads, V2andFDset);
                tmpRel2.free();
                newerEdges.orWith(tmpRel3);
            }
            
            V2set.free(); V4set.free(); V2andFDset.free();
            
            if (TRACE_MATCHING && !newerEdges.isZero()) {
                System.out.println("New edges from new loads/stores: "+newerEdges.toStringWithDomains(ts));
            }
            return newerEdges;
        }
            
        boolean matchNewEdges(BDD newEdges) {
            if (newEdges.isZero()) {
                newEdges.free();
                return false;
            }
            
            if (TRACE_MATCHING) System.out.println("Matching new edges...");
            
            // subtract out edges that already exist.
            newEdges.applyWith(edges.id(), BDDFactory.diff);
            
            if (newEdges.isZero()) {
                newEdges.free();
                return false;
            }
            
            BDD addedEdges24 = bdd.zero();
            BDD oldEdges12 = edges.replace(V3toV2);
            BDD oldEdges23 = edges.replace(V1toV2);
            BDD oldEdges24 = edges24.id();
            
            BDD V2set = V2.set();
            
            // Repeat for transitive closure.
            for (;;) {
                if (TRACE_MATCHING) {
                    System.out.println("New edges: "+newEdges.toStringWithDomains());
                }
                
                BDD newEdges12 = newEdges.replace(V3toV2);
                BDD newEdges23 = newEdges.replace(V1toV2);
                BDD newEdges24 = newEdges23.replace(V3toV4); // 24%
                
                // match old edges against new edges.
                BDD newerEdges = oldEdges12.relprod(newEdges23, V2set); // 26%
                newerEdges.orWith(newEdges12.relprod(oldEdges23, V2set)); // 6%
                
                // add new edges to edge set.
                addedEdges24.orWith(newEdges24.id());
                edges.orWith(newEdges);
                edges24.orWith(newEdges24);
                
                // subtract out edges that already exist.
                newerEdges.applyWith(edges.id(), BDDFactory.diff);
                
                if (newerEdges.isZero()) {
                    newerEdges.free();
                    break;
                }
                
                // update oldEdges12 and oldEdges23 for next iteration.
                oldEdges12.orWith(newEdges12);
                oldEdges23.orWith(newEdges23);
                
                // update newEdges for next iteration.
                newEdges = newerEdges;
                
            }
            oldEdges12.free();
            oldEdges23.free();
            
            // get the set of edges that we added.
            /*
            BDD newEdges24 = edges.id();
            newEdges24.applyWith(oldEdges13, BDDFactory.diff);
            newEdges24.replaceWith(V1V3toV2V4);
            */
            /*
            BDD edges24 = edges.replace(V1V3toV2V4);
            BDD newEdges24 = edges24.id();
            newEdges24.applyWith(oldEdges24, BDDFactory.diff);
            */
            addedEdges24.applyWith(oldEdges24, BDDFactory.diff);
            
            if (TRACE_MATCHING) {
                System.out.println("Total new edges: "+addedEdges24.toStringWithDomains());
            }
            
            BDD V4set = V4.set();
            BDD V2andFDset = V2.set(); V2andFDset.andWith(FD.set());
            
            BDD newerEdges = bdd.zero();
            BDD tmpRel1, tmpRel2, tmpRel3;
            
            // Matching rules:
            
            // Match loads and stores.
            // V1x(V2xFD) , V2xV4 , V3xV4 , (V3xFD)xV5   v1=v2.fd; v2=v4; v3=v4; v3.fd=v5;
            // ---------------------------------------   --------------------------------
            //               V1x(V4xFD)                               v1=v5;
            
            // new edges that cause more loads to match.
            tmpRel1 = loads.relprod(addedEdges24, V2set);
            tmpRel2 = tmpRel1.relprod(edges24, V4set);
            tmpRel1.free();
            tmpRel3 = tmpRel2.relprod(stores, V2andFDset);
            tmpRel2.free();
            newerEdges.orWith(tmpRel3);
            
            // new edges that cause more stores to match.
            tmpRel1 = stores.relprod(addedEdges24, V2set);
            tmpRel2 = tmpRel1.relprod(edges24, V4set);
            tmpRel1.free();
            tmpRel3 = tmpRel2.relprod(loads, V2andFDset);
            tmpRel2.free();
            newerEdges.orWith(tmpRel3);
            
            // subtract out existing edges.
            newerEdges.applyWith(edges.id(), BDDFactory.diff);

            if (TRACE_MATCHING) {
                System.out.println("New edges from load/store matching: "+newerEdges.toStringWithDomains());
            }
            
            V2set.free();
            V4set.free();
            V2andFDset.free();
            
            if (newerEdges.isZero()) {
                newerEdges.free();
            } else {
                matchNewEdges(newerEdges);
            }
            return true;
        }
        
        boolean matchEdges_incremental() {
            if (TRACE_MATCHING) System.out.println("Matching edges...");
            
            BDD V2set, V4andFDset;
            V2set = V2.set();
            V4andFDset = V4.set(); V4andFDset.andWith(FD.set());
            
            // Keep track of whether there was a change.
            boolean change = false;
            
            BDD allEdges12 = edges.replace(V3toV2);
            BDD allEdges23 = edges.replace(V1toV2);
            BDD allEdges24 = edges.replace(V1V3toV2V4);
            BDD newEdges12 = allEdges12.id();
            BDD newEdges23 = allEdges23.id();
            BDD newEdges24 = allEdges24.id();
            
            BDD newLoads14 = loads.replace(V2toV4);
            BDD newStores43 = stores.replace(V2toV4);
            
            for (;;) {
                
                // loop-carried dependencies: allEdges12, allEdges23, allEdges24,
                //                            newEdges12, newEdges23, newEdges24,
                //                            newLoads14, newStores43
                
                // cache old edges.
                BDD oldEdges13 = edges.id();
                
                // Repeat for transitive closure.
                for (;;) {
                    // Transitive closure.
                    // V1xV2 , V2xV3                        v1=v2; v2=v3;
                    // -------------                        -------------
                    //     V1xV3                                v1=v3;
                    BDD newEdges13 = allEdges12.relprod(newEdges23, V2set);
                    if (!allEdges12.equals(newEdges12) || !newEdges23.equals(allEdges23)) {
                        newEdges13.orWith(newEdges12.relprod(allEdges23, V2set));
                    }
                    newEdges12.free(); newEdges23.free();
                    newEdges13.applyWith(edges.id(), BDDFactory.diff);
                    if (newEdges13.isZero()) {
                        newEdges13.free();
                        break;
                    }
                    
                    change = true;
                    
                    if (TRACE_MATCHING) {
                        System.out.println("New edges due to transitive closure: "+newEdges13.toStringWithDomains());
                    }
                    
                    newEdges12 = newEdges13.replace(V3toV2);
                    newEdges23 = newEdges13.replace(V1toV2);
                    
                    edges.orWith(newEdges13);
                    allEdges12.orWith(newEdges12.id());
                    allEdges23.orWith(newEdges23.id());
                }
                
                // Matching rules:
                
                // Propagate loads to their children.
                // V1x(V2xFD) , V2xV4                   v1=v2.fd; v2=v4;
                // ------------------                   ----------------
                //     V1x(V4xFD)                           v1=v4.fd;
                BDD newLoadSet1 = loads.relprod(newEdges24, V2set);
                
                // Temporarily make loads into V1x(V4xFD)
                loads.replaceWith(V2toV4);
                
                // Add result to loads.
                newLoadSet1.applyWith(loads.id(), BDDFactory.diff);
                if (TRACE_MATCHING) System.out.println("New loads: "+newLoadSet1.toStringWithDomains());
                loads.orWith(newLoadSet1.id());
                newLoads14.orWith(newLoadSet1);
                
                // Propagate stores to their children.
                // (V2xFD)xV3 , V2xV4                   v2.fd=v3; v2=v4;
                // ------------------                   ----------------
                //     (V4xFD)xV3                           v4.fd=v3;
                BDD newStoreSet1 = stores.relprod(newEdges24, V2set);
                
                // Temporarily make stores into (V4xFD)xV3
                stores.replaceWith(V2toV4);
                
                // Add result to stores.
                newStoreSet1.applyWith(stores.id(), BDDFactory.diff);
                if (TRACE_MATCHING) System.out.println("New stores: "+newStoreSet1.toStringWithDomains());
                stores.orWith(newStoreSet1.id());
                newStores43.orWith(newStoreSet1);
                
                // Match loads and stores.
                // V1x(V4xFD) , (V4xFD)xV3           v1=v4.fd; v4.fd=v3; 
                // -----------------------           -------------------
                //          V1xV3                           v1=v3;
                BDD newEdges13 = loads.relprod(newStores43, V4andFDset);
                if (!loads.equals(newLoads14) || !newStores43.equals(stores)) {
                    newEdges13.orWith(newLoads14.relprod(stores, V4andFDset));
                }
                
                // calculate the diff.
                newEdges13.applyWith(edges.id(), BDDFactory.diff);
                
                // Change loads back into V1x(V2xFD)
                loads.replaceWith(V4toV2);
                // Change stores back into (V2xFD)xV3
                stores.replaceWith(V4toV2);
                
                // test for termination.
                if (newEdges13.isZero()) {
                    newEdges13.free();
                    break;
                }
                
                change = true;
                
                if (TRACE_MATCHING) {
                    System.out.println("New edges due to matching load/stores: "+newEdges13.toStringWithDomains());
                }
                
                // recalculate loop-carried dependencies
                newEdges12 = newEdges13.replace(V3toV2);
                newEdges23 = newEdges13.replace(V1toV2);
                allEdges12.orWith(newEdges12.id());
                allEdges23.orWith(newEdges23.id());
                newLoads14 = bdd.zero();
                newStores43 = bdd.zero();

                // Add result to edges.
                edges.orWith(newEdges13);
                
                // recalculate newEdges24 and allEdges24
                newEdges24 = edges.id();
                newEdges24.applyWith(oldEdges13, BDDFactory.diff);
                newEdges24.replaceWith(V1V3toV2V4);
                allEdges24.orWith(newEdges24.id());
                
                if (TRACE_MATCHING) {
                    System.out.println("Total new edges: "+newEdges24.toStringWithDomains());
                }
            }
            V2set.free();
            V4andFDset.free();
            if (TRACE_MATCHING) System.out.println("Done matching edges.");
            return change;
        }
        
        boolean transitiveClosure(BDD srcNodes) {
            BDD V1set, V2set, FDset;
            V1set = V1.set();
            V2set = V2.set();
            FDset = FD.set();
            
            // Keep track of whether there was a change.
            boolean change = false;
            
            for (;;) {
                BDD oldNodes = srcNodes.id();
                {
                    // Transitive along store edges.
                    // V2 x (V2xFD)xV3  =>  FDxV3
                    BDD newNodes = srcNodes.relprod(stores, V2set);
                    // FDxV3  =>  V3
                    BDD newNodes2 = newNodes.exist(FDset);
                    newNodes.free();
                    newNodes2.replaceWith(V3toV2);
                    srcNodes.orWith(newNodes2);
                }
                
                {
                    // Transitive along assignment edges.
                    srcNodes.replaceWith(V2toV1);
                    // V1 x V1xV3  =>  V3
                    BDD newNodes = srcNodes.relprod(edges, V1set);
                    srcNodes.replaceWith(V1toV2);
                    newNodes.replaceWith(V3toV2);
                    srcNodes.orWith(newNodes);
                }
                boolean done = oldNodes.equals(srcNodes);
                oldNodes.free();
                if (done) break;
                change = true;
            }
            
            V1set.free();
            V2set.free();
            FDset.free();
            return change;
        }
        
        void trim(BDD set) {
            if (TRACE_TRIMMING) {
                System.out.println("Trimming edges outside of the set "+set.toStringWithDomains(ts));
                System.out.println("Before: edges="+edges.nodeCount()+
                                          " loads="+loads.nodeCount()+
                                         " stores="+stores.nodeCount());
            }
                                     
            BDD v1_set = set.replace(V2toV1);
            BDD v3_set = set.replace(V2toV3);
            BDD v1xv3 = v1_set.and(v3_set);
            edges.andWith(v1xv3);

            BDD v1xv2xfd = v1_set.and(set);
            v1xv2xfd.andWith(FD.domain());
            loads.andWith(v1xv2xfd);
            
            BDD v2xfdxv3 = set.and(v3_set);
            v2xfdxv3.andWith(FD.domain());
            stores.andWith(v2xfdxv3);
            
            v1_set.free();
            v3_set.free();

            if (TRACE_TRIMMING) {
                System.out.println("After: edges="+edges.nodeCount()+
                                         " loads="+loads.nodeCount()+
                                        " stores="+stores.nodeCount());
            }
        }
    }
    
    public final ToString ts = new ToString();
    public class ToString extends BDD.BDDToString {
        ToString() { super(); }
        public String domainName(int i) {
            switch (i) {
            case 0: return "V1";
            case 1: return "V2";
            case 2: return "V3";
            case 3: return "V4";
            case 4: return "FD";
            default: throw new InternalError();
            }
        }
        public String elementName(int i, int j) {
            switch (i) {
            case 0: 
            case 1: 
            case 2: 
            case 3:
                if (j >= variableIndexMap.size())
                    return "ERROR: Out of range ("+j+")";
                return variableIndexMap.get(j).toString()+"("+j+")";
            case 4:
                if (j >= fieldIndexMap.size())
                    return "ERROR: Out of range ("+j+")";
                Object o = fieldIndexMap.get(j);
                return o==null?"[]":o.toString();
            default:
                throw new InternalError();
            }
        }
    }

    void makeVarOrdering(int[] varorder) {
        
        boolean reverseLocal = System.getProperty("csbddreverse", "true").equals("true");
        String ordering = System.getProperty("csbddordering", "FD_V2xV4xV1xV3");
        
        int varnum = bdd.varNum();
        
        int[][] localOrders = new int[domainBits.length][];
        localOrders[0] = new int[domainBits[0]];
        localOrders[1] = localOrders[0];
        localOrders[2] = new int[domainBits[2]];
        localOrders[3] = new int[domainBits[3]];
        localOrders[4] = localOrders[3];
        
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
        
        System.out.println("Ordering: "+ordering);
        StringTokenizer st = new StringTokenizer(ordering, "x_", true);
        int a = 0, idx = 0;
        for (;;) {
            String s = st.nextToken();
            BDDDomain d;
            if (s.equals("V1")) d = V1;
            else if (s.equals("V2")) d = V2;
            else if (s.equals("V3")) d = V3;
            else if (s.equals("V4")) d = V4;
            else if (s.equals("FD")) d = FD;
            else {
                Assert.UNREACHABLE("bad domain: "+s);
                return;
            }
            doms[a] = d;
            if (!st.hasMoreTokens()) {
                idx = fillInVarIndices(localOrders, idx, varorder, a+1, doms);
                break;
            }
            s = st.nextToken();
            if (s.equals("_")) {
                idx = fillInVarIndices(localOrders, idx, varorder, a+1, doms);
                a = 0;
            } else if (s.equals("x")) {
                a++;
            } else {
                Assert.UNREACHABLE("bad token: "+s);
                return;
            }
        }
        
        // according to the documentation of buddy, the default ordering is x1, y1, z1, x2, y2, z2, .....
        // V1[0] -> default variable number
        int[] outside2inside = new int[varnum];
        doms[0] = V1; doms[1] = V2; doms[2] = V3; doms[3] = V4;
        doms[4] = FD;
        getVariableMap(outside2inside, doms, domainBits.length);
        
        remapping(varorder, outside2inside);
    }
    
    int fillInVarIndices(int[][] localOrders, int start, int[] varorder, int numdoms, BDDDomain[] doms) {
        int totalvars = 0;
        int[] bits = new int[numdoms];
        for (int i = 0; i < numdoms; i++) {
            totalvars += domainBits[doms[i].getIndex()];
            bits[i] = 0;
        }

        for (int i = start, n = start + totalvars, j = 0; i < n; i++) {
            int dji = doms[j].getIndex();
            while (bits[j] >= domainBits[dji]) {
                j = (j + 1) % numdoms;
            }
            varorder[i] = domainSpos[dji] + localOrders[dji][bits[j]++];
            j = (j + 1) % numdoms;
        }

        return start + totalvars;
    }

    static void getVariableMap(int[] map, BDDDomain[] doms, int domnum) {
        int idx = 0;
        for (int var = 0; var < domnum; var++) {
            int[] vars = doms[var].vars();
            for (int i = 0; i < vars.length; i++) {
                map[idx++] = vars[i];
            }
        }
    }
    
    /* remap according to a map */
    static void remapping(int[] varorder, int[] maps) {
        for (int i = 0; i < varorder.length; i++) {
            varorder[i] = maps[varorder[i]];
        }
    }
    
}

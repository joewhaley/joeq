// BDDPointerAnalysis.java, created Sun Feb  2  2:22:10 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.BuDDyFactory;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_Field;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.CallTargets;
import Compil3r.Quad.AndersenInterface.AndersenType;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.MethodSummary.ReturnValueNode;
import Compil3r.Quad.MethodSummary.ThrownExceptionNode;
import Compil3r.Quad.MethodSummary.UnknownTypeNode;
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
    public static final boolean TRACE_INST = false;
    public static final boolean TRACE_VIRTUAL = false;

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

    // the size of domains, can be changed to reflect the size of inputs
    int domainBits[] = {18, 18, 13, 14, 14};
    // to be computed in sysInit function
    int domainSpos[] = {0, 0, 0, 0, 0}; 
    
    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1, V2, FD, H1, H2;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2, T3, T4; 

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

    public BDDPointerAnalysis() {
        this(DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
    
    public BDDPointerAnalysis(int nodeCount, int cacheSize) {
        bdd = BuDDyFactory.init(nodeCount, cacheSize);
        
        bdd.setCacheRatio(4);
        bdd.setMaxIncrease(nodeCount / 4);
        
        int[] domains = new int[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1 << domainBits[i]);
        }
        BDDDomain[] bdd_domains = bdd.extDomain(domains);
        V1 = bdd_domains[0];
        V2 = bdd_domains[1];
        FD = bdd_domains[2];
        H1 = bdd_domains[3];
        H2 = bdd_domains[4];
        T1 = V2;
        T2 = V1;
        T3 = H2;
        T4 = V2;
        
        int varnum = bdd.varNum();
        int[] varorder = new int[varnum];
        makeVarOrdering(varorder);
        for (int i=0; i<varorder.length; ++i) {
            //System.out.println("varorder["+i+"]="+varorder[i]);
        }
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
        
        V1ToV2 = bdd.makePair(V1, V2);
        V2ToV1 = bdd.makePair(V2, V1);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
        T2ToT1 = bdd.makePair(T2, T1);
    }

    int[][] localOrders;

    void makeVarOrdering(int[] varorder) {
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "FD_H2_V2xV1_H1");
        
        int varnum = bdd.varNum();
        
        localOrders = new int[domainBits.length][];
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
            else if (s.equals("FD")) d = FD;
            else if (s.equals("H1")) d = H1;
            else if (s.equals("H2")) d = H2;
            else {
                Assert.UNREACHABLE("bad domain: "+s);
                return;
            }
            doms[a] = d;
            if (!st.hasMoreTokens()) {
                idx = fillInVarIndices(idx, varorder, a+1, doms);
                break;
            }
            s = st.nextToken();
            if (s.equals("_")) {
                idx = fillInVarIndices(idx, varorder, a+1, doms);
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
        doms[0] = V1; doms[1] = V2;
        doms[2] = FD; doms[3] = H1; doms[4] = H2;
        getVariableMap(outside2inside, doms, domainBits.length);
        
        remapping(varorder, outside2inside);
    }
    
    int fillInVarIndices(int start, int[] varorder, int numdoms, BDDDomain[] doms) {
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

    void getVariableMap(int[] map, BDDDomain[] doms, int domnum) {
        int idx = 0;
        for (int var = 0; var < domnum; var++) {
            int[] vars = doms[var].vars();
            for (int i = 0; i < vars.length; i++) {
                map[idx++] = vars[i];
            }
        }
    }
    
    /* remap according to a map */
    void remapping(int[] varorder, int[] maps) {
        for (int i = 0; i < varorder.length; i++) {
            varorder[i] = maps[varorder[i]];
        }
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
    
    void done() {
        aC.andWith(bdd.zero());
        vC.andWith(bdd.zero());
        cC.andWith(bdd.zero());
        cTypes.andWith(bdd.zero());
        cTypes.andWith(bdd.zero());
        pointsTo.andWith(bdd.zero());
        edgeSet.andWith(bdd.zero());
        typeFilter.andWith(bdd.zero());
        stores.andWith(bdd.zero());
        loads.andWith(bdd.zero());
        storePt.andWith(bdd.zero());
        fieldPt.andWith(bdd.zero());
        loadAss.andWith(bdd.zero());
        loadPt.andWith(bdd.zero());
        vtable_bdd.andWith(bdd.zero());
        System.gc(); System.gc();
        bdd.done();
    }

    public static boolean INCREMENTAL_POINTSTO = true;
    public static boolean INCREMENTAL_ITERATION = true;
    public static boolean FORCE_GC = false;

    public static void main(String[] args) {
        HostedVM.initialize();
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        BDDPointerAnalysis dis = new BDDPointerAnalysis();
        dis.reset();
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
        
        if (DUMP)
            dis.dumpResults(cg);
            
    }
    
    public CallGraph goNonincremental(Collection roots) {
        long time = System.currentTimeMillis();
        
        this.addObjectAllocation(GlobalNode.GLOBAL, null);
        this.addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        this.addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod) i.next();
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                this.handleMethodSummary(ms);
            }
        }
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        int iteration = 1;
        do {
            long time2 = System.currentTimeMillis();

            System.out.println("--> Iteration "+iteration+" Methods: "+this.visitedMethods.size()+" Call graph edges: "+this.callGraphEdges.size());
            this.change = false;
            this.calculateTypeFilter();

            long time3 = System.currentTimeMillis();
            System.out.println("Calculate type filter:\t"+(time3-time2)/1000.+" seconds.");

            if (INCREMENTAL_POINTSTO) this.solveIncremental();
            else this.solveNonincremental();

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
        
        CallGraph cg = new BDDCallGraph(roots);
        return cg;
    }
    
    public CallGraph goIncremental(Collection roots) {
        long time = System.currentTimeMillis();
        
        this.addObjectAllocation(GlobalNode.GLOBAL, null);
        this.addAllocType(null, PrimordialClassLoader.getJavaLangObject());
        this.addVarType(GlobalNode.GLOBAL, PrimordialClassLoader.getJavaLangObject());
        
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod) i.next();
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                this.handleMethodSummary(ms);
            }
        }
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        int iteration = 1;
        BDD oldPointsTo = this.bdd.zero();
        do {
            long time2 = System.currentTimeMillis();

            System.out.println("--> Iteration "+iteration+" Methods: "+this.visitedMethods.size()+" Call graph edges: "+this.callGraphEdges.size());
            this.change = false;
            this.calculateTypeFilter();

            long time3 = System.currentTimeMillis();
            System.out.println("Calculate type filter:\t"+(time3-time2)/1000.+" seconds.");

            if (INCREMENTAL_POINTSTO) this.solveIncremental();
            else this.solveNonincremental();
            BDD newPointsTo = this.pointsTo.apply(oldPointsTo, BDDFactory.diff);
            oldPointsTo.free();

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
        
        CallGraph cg = new BDDCallGraph(roots);
        return cg;
    }

    boolean change;

    IndexMap getIndexMap(BDDDomain d) {
        if (d == V1 || d == V2) return variableIndexMap;
        if (d == FD) return fieldIndexMap;
        if (d == H1 || d == H2) return heapobjIndexMap;
        return null;
    }

    void printSet(String desc, BDD b) {
        System.out.print(desc+": ");
        System.out.flush();
        //if (desc.startsWith(" "))
            b.printSetWithDomains();
        System.out.println();
    }
    
    void printSet(String desc, BDD b, String type) {
        printSet(desc+" ("+type+")", b);
        
        if (false) {
            int i = 0, n = 1;
            while ((i = type.indexOf('x', i)+1) > 0) {
                n++;
            }
            BDDDomain[] d = new BDDDomain[n];
            IndexMap[] m = new IndexMap[n];
            String[] names = new String[n];
            i = 0;
            for (int j=0; j<d.length; ++j) {
                i = type.indexOf('x', i);
                String s;
                if (i >= 0) {
                    s = type.substring(0, i);
                    type = type.substring(i+1);
                } else {
                    Assert._assert(j == d.length-1);
                    s = type;
                }
                names[j] = s;
                BDDDomain t1;
                IndexMap t2;
                if (s.equals("V1")) {
                    t1 = V1; t2 = variableIndexMap;
                } else if (s.equals("V2")) {
                    t1 = V1; t2 = variableIndexMap;
                } else if (s.equals("FD")) {
                    t1 = FD; t2 = fieldIndexMap;
                } else if (s.equals("H1")) {
                    t1 = H1; t2 = heapobjIndexMap;
                } else if (s.equals("H2")) {
                    t1 = H2; t2 = heapobjIndexMap;
                } else if (s.equals("T1")) {
                    t1 = T1; t2 = typeIndexMap;
                } else if (s.equals("T2")) {
                    t1 = T2; t2 = typeIndexMap;
                } else if (s.equals("T3")) {
                    t1 = T3; t2 = methodIndexMap;
                } else if (s.equals("T4")) {
                    t1 = T4; t2 = targetIndexMap;
                } else {
                    System.out.println("Unknown domain "+s);
                    return;
                }
                d[j] = t1;
                m[j] = t2;
            }
    
            System.out.print(desc+": ");
            printSet_recurse(b.id(), d, m, names, 0);
            System.out.println();
        }
    }

    void printSet_recurse(BDD b, BDDDomain[] d, IndexMap[] m, String[] names, int i) {
        System.out.print(names[i]+": ");
        int k=0;
        for (;;++k) {
            int p = b.scanVar(d[i]);
            if (p < 0) break;
            if (k != 0) System.out.print("/");
            System.out.print(m[i].get(p));
            BDD r = d[i].ithVar(p);
            if (i < d.length-1) {
                BDD b2 = b.restrict(r);
                System.out.print(", ");
                printSet_recurse(b2, d, m, names, i+1);
                b2.free();
            }
            b.applyWith(r, BDDFactory.diff);
        }
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
            for (int i=0, n=variableIndexMap.size(); i<n; ++i) {
                Node node = (Node) variableIndexMap.get(i);
                System.out.print(i+": "+node.toString());
                BDD var = V1.ithVar(i);
                BDD p = pointsTo.restrict(var);
                printSet(" can point to", p, "H1");
            }
        }
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

    public void addClassInit(jq_Type t) {
        if (IGNORE_CLINIT) return;
        if (t instanceof jq_Class) {
            jq_Class c = (jq_Class) t;
            c.prepare();
            jq_ClassInitializer i = c.getClassInitializer();
            if (i != null && i.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(i);
                MethodSummary ms = MethodSummary.getSummary(cfg);
                handleMethodSummary(ms);
            }
        }
    }

    HashSet visitedMethods = new HashSet();

    public void handleNode(Node n) {
        Iterator j;
        j = n.getEdges().iterator();
        while (j.hasNext()) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Object o = e.getValue();
            // n.f = o
            if (o instanceof Set) {
                addFieldStore(n, f, (Set) o);
            } else {
                addFieldStore(n, f, (Node) o);
                if (n instanceof GlobalNode)
                    addClassInit(f.getDeclaringClass());
            }
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
                if (n instanceof GlobalNode)
                    addClassInit(f.getDeclaringClass());
            }
        }
        if (n instanceof ConcreteTypeNode) {
            ConcreteTypeNode ctn = (ConcreteTypeNode) n;
            addObjectAllocation(ctn, ctn);
            addAllocType(ctn, (jq_Reference) ctn.getDeclaredType());
            addClassInit((jq_Reference) ctn.getDeclaredType());
        } else if (n instanceof UnknownTypeNode) {
            UnknownTypeNode utn = (UnknownTypeNode) n;
            addObjectAllocation(utn, utn);
            addAllocType(utn, (jq_Reference) utn.getDeclaredType());
        }
        if (n instanceof GlobalNode) {
            addDirectAssignment(GlobalNode.GLOBAL, n);
            addDirectAssignment(n, GlobalNode.GLOBAL);
            addVarType(n, PrimordialClassLoader.getJavaLangObject());
        } else {
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }
    }

    public void handleMethodSummary(MethodSummary ms) {
        if (visitedMethods.contains(ms)) return;
        visitedMethods.add(ms);
        this.change = true;
        for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            handleNode(n);
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
                int methodIndex = getMethodIndex(method);
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
                    printSet("receiverBDD", receiverBDD, "V1");
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
        BDD V1set = V1.set();
        BDD H1andT3set = H1.set(); H1andT3set.andWith(T3.set());
        for (int index=0; h.hasNext(); ++index) {
            CallSite cs = (CallSite) h.next();
            MethodSummary caller = cs.caller;
            ProgramLocation mc = cs.m;
            BDD receiverVars = (BDD) i.next();
            if (TRACE_VIRTUAL) {
                System.out.println("Caller: "+caller.getMethod());
                System.out.println("Call: "+mc);
                printSet(" receiverVars", receiverVars, "V1");
            }
            BDD pt = (index < last_vcalls) ? newPointsTo : pointsTo;
            BDD receiverObjects;
            if (receiverVars.satCount(V1set) == 1.0) {
                receiverObjects = pt.restrict(receiverVars);
                //jq.Assert(receiverObjects.equals(pt.relprod(receiverVars, V1set)));
            } else {
                receiverObjects = pt.relprod(receiverVars, V1set); // time-consuming!
            }
            if (TRACE_VIRTUAL) {
                printSet(" receiverObjects", receiverObjects, "H1");
            }
            jq_InstanceMethod method = (jq_InstanceMethod) j.next();
            if (receiverObjects.equals(bdd.zero())) {
                continue;
            }
            int methodIndex = getMethodIndex(method);
            BDD methodBDD = T3.ithVar(methodIndex);
            if (TRACE_VIRTUAL) {
                printSet("Method "+method+" index "+methodIndex, methodBDD, "T3");
            }
            receiverObjects.andWith(methodBDD);
            if (TRACE_VIRTUAL) {
                printSet(" receiverObjects", receiverObjects, "H1xT3");
            }
            // (H1 x T3) * (H1 x T3 x T4) 
            BDD targets = receiverObjects.relprod(vtable_bdd, H1andT3set);
            receiverObjects.free();
            if (TRACE_VIRTUAL) {
                printSet(" targets", targets, "T4");
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
                    ControlFlowGraph cfg = CodeCache.getCode(target);
                    MethodSummary ms2 = MethodSummary.getSummary(cfg);
                    handleMethodSummary(ms2);
                    bindParameters(caller, mc, ms2);
                } else {
                    bindParameters_native(caller, mc);
                }
                targets.applyWith(T4.ithVar(p), BDDFactory.diff);
            }
            targets.free();
        }
        V1set.free();
        H1andT3set.free();
        last_vcalls = n;
    }
    
    HashSet callGraphEdges = new HashSet();
    
    public void bindParameters(MethodSummary caller, ProgramLocation mc, MethodSummary callee) {
        Object key = new CallSite(callee, mc);
        if (callGraphEdges.contains(key)) return;
        if (true && !this.change) {
            System.out.println("Adding call graph edge "+caller.getMethod()+"->"+callee.getMethod());
        }
        callGraphEdges.add(key);
        this.change = true;
        for (int i=0; i<mc.getNumParams(); ++i) {
            if (i >= callee.getNumOfParams()) break;
            ParamNode pn = callee.getParamNode(i);
            if (pn == null) continue;
            PassedParameter pp = new PassedParameter(mc, i);
            Set s = caller.getNodesThatCall(pp);
            addDirectAssignment(pn, s);
        }
        Set rvn_s;
        Object rvn_o = caller.callToRVN.get(mc);
        if (rvn_o instanceof Set) rvn_s = (Set) rvn_o;
        else if (rvn_o != null) rvn_s = Collections.singleton(rvn_o);
        else rvn_s = Collections.EMPTY_SET;
        for (Iterator i=rvn_s.iterator(); i.hasNext(); ) {
            ReturnValueNode rvn = (ReturnValueNode) i.next();
            if (rvn != null) {
                Set s = callee.returned;
                addDirectAssignment(rvn, s);
            }
        }
        Set ten_s;
        Object ten_o = caller.callToTEN.get(mc);
        if (ten_o instanceof Set) ten_s = (Set) ten_o;
        else if (ten_o != null) ten_s = Collections.singleton(ten_o);
        else ten_s = Collections.EMPTY_SET;
        for (Iterator i=ten_s.iterator(); i.hasNext(); ) {
            ThrownExceptionNode ten = (ThrownExceptionNode) i.next();
            if (ten != null) {
                Set s = callee.thrown;
                addDirectAssignment(ten, s);
            }
        }
    }

    public void bindParameters_native(MethodSummary caller, ProgramLocation mc) {
        // only handle return value for now.
        AndersenType t = caller.getMethod().and_getReturnType();
        if (t instanceof jq_Reference) {
            Set rvn_s;
            Object rvn_o = caller.callToRVN.get(mc);
            if (rvn_o instanceof Set) rvn_s = (Set) rvn_o;
            else if (rvn_o != null) rvn_s = Collections.singleton(rvn_o);
            else rvn_s = Collections.EMPTY_SET;
            UnknownTypeNode utn = UnknownTypeNode.get((jq_Reference) t);
            addObjectAllocation(utn, utn);
            addAllocType(utn, (jq_Reference) t);
            addVarType(utn, (jq_Reference) t);
            for (Iterator i=rvn_s.iterator(); i.hasNext(); ) {
                ReturnValueNode rvn = (ReturnValueNode) i.next();
                if (rvn != null) {
                    addDirectAssignment(rvn, utn);
                }
            }
        }
    }
        
    IndexMap/* Node->index */ variableIndexMap = new IndexMap("Variable");
    IndexMap/* Node->index */ heapobjIndexMap = new IndexMap("HeapObj");
    IndexMap/* jq_Field->index */ fieldIndexMap = new IndexMap("Field");
    IndexMap/* jq_Reference->index */ typeIndexMap = new IndexMap("Class");
    IndexMap/* jq_InstanceMethod->index */ methodIndexMap = new IndexMap("MethodCall");
    IndexMap/* jq_InstanceMethod->index */ targetIndexMap = new IndexMap("MethodTarget");

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
    int getMethodIndex(jq_InstanceMethod f) {
        return methodIndexMap.get(f);
    }
    int getTargetIndex(jq_InstanceMethod f) {
        return targetIndexMap.get(f);
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
            //printSet("Adding object allocation site="+site_i+" dest="+dest_i, dest_bdd, "V1xH1");
        }
        pointsTo.orWith(dest_bdd);
        if (TRACE) {
            printSet("Points-to is now", pointsTo, "V1xH1");
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
                //printSet("Adding direct assignment dest="+dest_i+" src="+src_i, src_bdd, "V1xV2");
            }
            edgeSet.orWith(src_bdd);
            if (TRACE) {
                printSet("Edge-set is now", edgeSet, "V1xV2");
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
            //printSet("Adding direct assignment dest="+dest_i+" src="+src_i, dest_bdd, "V1xV2");
        }
        edgeSet.orWith(dest_bdd);
        if (TRACE) {
            printSet("Edge-set is now", edgeSet, "V1xV2");
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
            //printSet("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i, dest_bdd, "V1xV2xFD");
        }
        loads.orWith(dest_bdd);
        if (TRACE) {
            printSet("Loads-set is now", loads, "V1xV2xFD");
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
                //printSet("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i, dest_bdd, "V1xV2xFD");
            }
            loads.orWith(dest_bdd);
            if (TRACE) {
                printSet("Loads-set is now", loads, "V1xV2xFD");
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
            //printSet("Adding store field base="+base_i+" f="+f_i+" src="+src_i, base_bdd, "V1xV2xFD");
        }
        stores.orWith(base_bdd);
        if (TRACE) {
            printSet("Stores-set is now", stores, "V1xV2xFD");
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
            if (TRACE) {
                printSet("Stores-set is now", stores, "V1xV2xFD");
            }
        }
        base_bdd.free(); f_bdd.free();
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
            printSet("Adding alloc type site="+site_i+" type="+type_i, type_bdd, "H1xT2");
        }
        aC.orWith(type_bdd);
        if (TRACE) {
            printSet("AllocClass is now", aC, "H1xT2");
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
            printSet("Adding var type var="+var_i+" type="+type_i, type_bdd, "V1xT1");
        }
        vC.orWith(type_bdd);
        if (TRACE) {
            printSet("VarClass is now", vC, "V1xT1");
        }
    }
    
    int last_typeIndex;
    
    void calculateTypeHierarchy() {
        int n1=typeIndexMap.size();
        if (TRACE) System.out.println(n1-last_typeIndex + " new types");
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
        cTypes = aC.replace(T2ToT1);
        //cC.free(); vC.free(); aC.free();

        if (false) typeFilter = bdd.one();
    }
    
    BDD vtable_bdd; // H1 x T3 x T4
    
    int last_methodIndex;
    int last_heapobjIndex;
    
    public void calculateVTables() {
        int n1 = methodIndexMap.size();
        int n2 = heapobjIndexMap.size();
        if (TRACE) {
            System.out.println(n1-last_methodIndex + " new distinct virtual methods");
            System.out.println(n2-last_heapobjIndex + " new heap objects");
        }
        for (int i1=0; i1<n1; ++i1) {
            jq_InstanceMethod m = (jq_InstanceMethod) methodIndexMap.get(i1);
            BDD method_bdd = T3.ithVar(i1);
            int i2 = (i1 < last_methodIndex) ? last_heapobjIndex : 0;
            for ( ; i2<n2; ++i2) {
                Node c = (Node) heapobjIndexMap.get(i2);
                if (c == null) continue;
                jq_Reference r2 = (jq_Reference) c.getDeclaredType();
                if (r2 == null) continue;
                r2.prepare(); m.getDeclaringClass().prepare();
                if (!r2.isSubtypeOf(m.getDeclaringClass())) continue;
                BDD heapobj_bdd = H1.ithVar(i2);
                if (c instanceof ConcreteTypeNode) {
                    jq_InstanceMethod target = r2.getVirtualMethod(m.getNameAndDesc());
                    if (target != null) {
                        int i3 = getTargetIndex(target);
                        BDD target_bdd = T4.ithVar(i3);
                        target_bdd.andWith(heapobj_bdd.id());
                        target_bdd.andWith(method_bdd.id());
                        vtable_bdd.orWith(target_bdd);
                    }
                } else {
                    CallTargets ct = CallTargets.getTargets(m.getDeclaringClass(), m, BytecodeVisitor.INVOKE_VIRTUAL, r2, false, true);
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
        if (TRACE) {
            printSet("vtable", vtable_bdd, "H1xT3xT4");
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
                BDD newPt1 = edgeSet.relprod(pointsTo, V1.set());
                // (V2 x H1) => (V1 x H1)
                BDD newPt2 = newPt1.replace(V2ToV1);

                /* --- apply type filtering and merge into pointsTo relation --- */
                // (V1 x H1)
                BDD newPt3 = newPt2.and(typeFilter);
                if (TRACE) {
                    BDD temp = newPt2.apply(typeFilter, BDDFactory.diff);
                    printSet("removed by type filter", temp, "V1xH1");
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
            BDD tmpRel1 = stores.relprod(pointsTo, V1.set());
            // ((V2 x FD) x H1) => ((V1 x FD) x H2)
            BDD tmpRel2 = tmpRel1.replace(V2ToV1).replace(H1ToH2);
            // ((V1 x FD) x H2) * (V1 x H1) => ((H1 x FD) x H2)
            fieldPt = tmpRel2.relprod(pointsTo, V1.set());

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
            if (TRACE) {
                printSet("before type filter", newPt5, "V1xH1");
            }
            BDD newPt6 = newPt5.and(typeFilter);
            if (TRACE) {
                printSet("after type filter", newPt6, "V1xH1");
            }
            pointsTo = pointsTo.or(newPt6);

        }
        while (!oldPt1.equals(pointsTo));

    }
    
    public void solveIncremental() {

        BDD empty = bdd.zero();
        
        BDD oldPointsTo = bdd.zero();
        BDD newPointsTo = pointsTo.id();
        BDD V1set = V1.set();
        BDD H1andFDset = H1.set();
        H1andFDset.andWith(FD.set());

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
    }

    public class BDDCallGraph extends CallGraph {

        Collection roots;

        BDDCallGraph(Collection roots) {
            this.roots = roots;
        }

        /**
         * @see Compil3r.Quad.CallGraph#getTargetMethods(java.lang.Object, Compil3r.Quad.ProgramLocation)
         */
        public Collection getTargetMethods(Object context, ProgramLocation callSite) {
            jq_Method method = (jq_Method) callSite.getTargetMethod();
            if (callSite.isSingleTarget())
                return Collections.singleton(method);
            Collection targets = (Collection) callSiteToTargets.get(callSite);
            if (targets != null) return targets;
            else return Collections.EMPTY_SET;
        }

        /* (non-Javadoc)
         * @see Compil3r.Quad.CallGraph#setRoots(java.util.Collection)
         */
        public void setRoots(Collection roots) {
            throw new UnsupportedOperationException();
        }

        /* (non-Javadoc)
         * @see Compil3r.Quad.CallGraph#getRoots()
         */
        protected Collection getRoots() {
            return roots;
        }
        
        /* (non-Javadoc)
         * @see Compil3r.Quad.CallGraph#getAllCallSites()
         */
        public Collection getAllCallSites() {
            return callSiteToTargets.keySet();
        }

        /* (non-Javadoc)
         * @see Compil3r.Quad.CallGraph#getAllMethods()
         */
        public Collection getAllMethods() {
            LinkedHashSet s = new LinkedHashSet();
            s.addAll(roots);
            for (Iterator i=callSiteToTargets.values().iterator(); i.hasNext(); ) {
                Collection c = (Collection) i.next();
                s.addAll(c);
            }
            return s;
        }

    }
}

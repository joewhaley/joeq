
package Compil3r.Quad;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_Field;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.MethodSummary.ReturnValueNode;
import Compil3r.Quad.MethodSummary.ThrownExceptionNode;
import Main.HostedVM;
import Main.jq;
import Run_Time.TypeCheck;
import Util.HashCodeComparator;
import Util.SortedArraySet;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.BuDDyFactory;

/**
 * @author John Whaley
 * @version $Id$
 */
public class BDDPointerAnalysis {

    public static final boolean TRACE = false;

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

    // relations
    BDD pointsTo;     // V1 x H1
    BDD edgeSet;      // V1 x V2
    BDD typeFilter;   // V1 x H1
    BDD stores;       // V1 x (V2 x FD) 
    BDD loads;        // (V1 x FD) x V2

    // cached temporary relations
    BDD storePt, fieldPt, loadAss, loadPt;

    public BDDPointerAnalysis() {
        this(DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }
        
    public BDDPointerAnalysis(int nodeCount, int cacheSize) {
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
    }

    public static final int FD_V1_V2_H1_H2 = 0;
    public static final int FD_V1xV2_H1_H2 = 1;
    public static final int FD_V2_V1_H1_H2 = 2;
    public static final int FD_V2xV1_H1_H2 = 3;
    public static final int FDxV1xV2xH1xH2 = 4;
    public static final int FD_H1_V1xV2_H2 = 5;
    public static final int FD_H1_V2xV1_H2 = 6;
    public static final int H1_V1xV2_FD_H2 = 7;
    public static final int H1_V2xV1_FD_H2 = 8;
    public static final int FD_V1xV2xH1_H2 = 9;

    int[][] localOrders;

    void makeVarOrdering(int[] varorder) {
        
        boolean reverseLocal = true;
        int ordering = FD_H1_V2xV1_H2;
        
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
        
        if (ordering == FD_V1_V2_H1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FD_V1xV2_H1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1; doms[1] = V2;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FD_V2_V1_H1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FD_V2xV1_H1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V2; doms[1] = V1;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FDxV1xV2xH1xH2) {
            int idx = 0;
            doms[0] = FD; doms[1] = V1; doms[2] = V2;
            doms[3] = H1; doms[4] = H2;
            idx = fillInVarIndices(idx, varorder, 5, doms);
        } else if (ordering == FD_H1_V1xV2_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1; doms[1] = V2;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FD_H1_V2xV1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V2; doms[1] = V1;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == H1_V1xV2_FD_H2) {
            int idx = 0;
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1; doms[1] = V2;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == H1_V2xV1_FD_H2) {
            int idx = 0;
            doms[0] = H1;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V2; doms[1] = V1;
            idx = fillInVarIndices(idx, varorder, 2, doms);
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else if (ordering == FD_V1xV2xH1_H2) {
            int idx = 0;
            doms[0] = FD;
            idx = fillInVarIndices(idx, varorder, 1, doms);
            doms[0] = V1; doms[1] = V2; doms[2] = H1;
            idx = fillInVarIndices(idx, varorder, 3, doms);
            doms[0] = H2;
            idx = fillInVarIndices(idx, varorder, 1, doms);
        } else {
            jq.UNREACHABLE("Ordering is not defined");
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
    }

    public static boolean INCREMENTAL = true;

    public static void main(String[] args) {
        HostedVM.initialize();
        
        long time = System.currentTimeMillis();
        
        BDDPointerAnalysis dis = new BDDPointerAnalysis();
        dis.reset();
        
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        List methods = java.util.Arrays.asList(c.getStaticMethods());
        for (Iterator i=methods.iterator(); i.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod) i.next();
            if (m.getBytecode() == null) continue;
            ControlFlowGraph cfg = CodeCache.getCode(m);
            MethodSummary ms = MethodSummary.getSummary(cfg);
            dis.handleMethodSummary(ms);
        }
        System.out.println("Setup: "+(System.currentTimeMillis()-time)/1000.+" seconds.");
        int iteration = 1;
        do {
            long time2 = System.currentTimeMillis();
            System.out.println("Iteration "+iteration+" Methods: "+dis.visitedMethods.size()+" Call graph edges: "+dis.callGraphEdges.size());
            dis.change = false;
            dis.calculateTypeFilter();
            if (INCREMENTAL) dis.solveIncremental();
            else dis.solveNonincremental();
            dis.calculateVTables();
            dis.handleVirtualCalls();
            if (true) System.gc();
            time2 = System.currentTimeMillis() - time2;
            System.out.println(time2/1000.+" seconds.");
            ++iteration;
        } while (dis.change);
        
        time = System.currentTimeMillis() - time;
        System.out.println("Total time: "+time/1000.+" seconds.");
        
        //dis.dumpResults();
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
                    jq.Assert(j == d.length-1);
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
            int p = b.scanVar(d[i].getIndex());
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

    BDD cTypes;

    public void dumpResults() {
        System.out.println(visitedMethods.size()+" methods");
        
        System.out.println(AndersenPointerAnalysis.dumpResults(callSiteToTargets));
        
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

    public void handleMethodSummary(MethodSummary ms) {
        if (visitedMethods.contains(ms)) return;
        visitedMethods.add(ms);
        this.change = true;
        for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            for (Iterator j=n.getEdges().iterator(); j.hasNext(); ) {
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
            for (Iterator j=n.getAccessPathEdges().iterator(); j.hasNext(); ) {
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
            }
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }
        
        addClassInit(((jq_Method) ms.getMethod()).getDeclaringClass());
        
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            CallSite cs = new CallSite(ms, mc);
            jq.Assert(!callSiteToTargets.containsKey(cs));
            if (!mc.isVirtual()) {
                jq_Method target = (jq_Method) mc.getTargetMethod();
                Set definite_targets = Collections.singleton(target);
                callSiteToTargets.put(cs, definite_targets);
                if (target.getBytecode() == null) continue;
                ControlFlowGraph cfg2 = CodeCache.getCode(target);
                MethodSummary ms2 = MethodSummary.getSummary(cfg2);
                handleMethodSummary(ms2);
                bindParameters(ms, mc, ms2);
            } else {
                Set definite_targets = SortedArraySet.FACTORY.makeSet(HashCodeComparator.INSTANCE);
                callSiteToTargets.put(cs, definite_targets);
                jq_InstanceMethod method = (jq_InstanceMethod) mc.getTargetMethod();
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

    public void handleVirtualCalls() {
        Iterator h=new LinkedList(virtualCallSites).iterator();
        Iterator i=new LinkedList(virtualCallReceivers).iterator();
        Iterator j=new LinkedList(virtualCallMethods).iterator();
        for (; i.hasNext(); ) {
            CallSite cs = (CallSite) h.next();
            MethodSummary caller = cs.caller;
            if (TRACE) {
                System.out.println("Caller: "+caller.getMethod());
            }
            ProgramLocation mc = cs.m;
            if (TRACE) {
                System.out.println("Call: "+mc);
            }
            BDD receiverVars = (BDD) i.next();
            if (TRACE) {
                printSet("receiverVars", receiverVars, "V1");
            }
            BDD receiverObjects = pointsTo.restrict(receiverVars);
            if (TRACE) {
                printSet("receiverObjects", receiverObjects, "H1");
            }
            jq_InstanceMethod method = (jq_InstanceMethod) j.next();
            int methodIndex = getMethodIndex(method);
            BDD methodBDD = T3.ithVar(methodIndex);
            if (TRACE) {
                printSet("Method "+method+" index "+methodIndex, methodBDD, "T3");
            }
            receiverObjects.andWith(methodBDD);
            if (TRACE) {
                printSet("receiverObjects", receiverObjects, "H1xT3");
            }
            // (H1 x T3) * (H1 x T3 x T4) 
            BDD targets = receiverObjects.relprod(vtable_bdd, H1.set().and(T3.set()));
            if (TRACE) {
                printSet("targets", targets, "T4");
            }
            if (TRACE) {
                System.out.println("# of targets: "+targets.satCount(T4.set()));
            }
            Set definite_targets = (Set) callSiteToTargets.get(cs);
            for (;;) {
                int p = targets.scanVar(T4.getIndex());
                if (p < 0) break;
                jq_InstanceMethod target = getTarget(p);
                if (TRACE) {
                    System.out.println("Target "+p+": "+target);
                }
                definite_targets.add(target);
                if (target.getBytecode() != null) {
                    ControlFlowGraph cfg = CodeCache.getCode(target);
                    MethodSummary ms2 = MethodSummary.getSummary(cfg);
                    handleMethodSummary(ms2);
                    bindParameters(caller, mc, ms2);
                }
                targets.applyWith(T4.ithVar(p), BDDFactory.diff);
            }
        }
    }
    
    HashSet callGraphEdges = new HashSet();
    
    public void bindParameters(MethodSummary caller, ProgramLocation mc, MethodSummary callee) {
        Object key = new CallSite(callee, mc);
        if (callGraphEdges.contains(key)) return;
        if (TRACE && !this.change) {
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

    IndexMap/* Node->index */ variableIndexMap = new IndexMap("Variable");
    IndexMap/* ConcreteTypeNode->index */ heapobjIndexMap = new IndexMap("HeapObj");
    IndexMap/* jq_Field->index */ fieldIndexMap = new IndexMap("Field");
    IndexMap/* jq_Reference->index */ typeIndexMap = new IndexMap("Class");
    IndexMap/* jq_InstanceMethod->index */ methodIndexMap = new IndexMap("MethodCall");
    IndexMap/* jq_InstanceMethod->index */ targetIndexMap = new IndexMap("MethodTarget");

    int getVariableIndex(Node dest) {
        return variableIndexMap.get(dest);
    }
    int getHeapobjIndex(ConcreteTypeNode site) {
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
    ConcreteTypeNode getHeapobj(int index) {
        return (ConcreteTypeNode) heapobjIndexMap.get(index);
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

    public void addObjectAllocation(Node dest, ConcreteTypeNode site) {
        int dest_i = getVariableIndex(dest);
        int site_i = getHeapobjIndex(site);
        BDD dest_bdd = V1.ithVar(dest_i);
        BDD site_bdd = H1.ithVar(site_i);
        dest_bdd.andWith(site_bdd);
        if (TRACE) {
            printSet("object allocation site="+site_i+" dest="+dest_i, dest_bdd, "V1");
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
            if (TRACE) {
                printSet("Adding direct assignment dest="+dest_i+" src="+src_i, src_bdd, "V1xV2");
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
        if (TRACE) {
            printSet("Adding direct assignment dest="+dest_i+" src="+src_i, dest_bdd, "V1xV2");
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
        if (TRACE) {
            printSet("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i, dest_bdd, "V1xV2xFD");
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
            if (TRACE) {
                printSet("Adding load field dest="+dest_i+" base="+base_i+" f="+f_i, dest_bdd, "V1xV2xFD");
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
        if (TRACE) {
            printSet("Adding store field base="+base_i+" f="+f_i+" src="+src_i, base_bdd, "V1xV2xFD");
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
            if (TRACE) {
                printSet("Adding store field base="+base_i+" f="+f_i+" src="+src_i, base_bdd, "V1xV2xFD");
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

    BDD aC, vC, cC;

    public void addAllocType(ConcreteTypeNode site, jq_Reference type) {
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
    
    void calculateTypeHierarchy() {
        // not very efficient.
        for (int i1=0, n1=typeIndexMap.size(); i1<n1; ++i1) {
            jq_Type t1 = (jq_Type) typeIndexMap.get(i1);
            if (t1 == null) {
                BDD type1_bdd = T1.ithVar(i1);
                BDD type2_bdd = T2.set();
                type1_bdd.andWith(type2_bdd);
                cC.orWith(type1_bdd);
                continue;
            }
            t1.prepare();
            for (int i2=0, n2=typeIndexMap.size(); i2<n2; ++i2) {
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
        cTypes = aC.replace(V1ToV2);
        //cC.free(); vC.free(); aC.free();
    }
    
    BDD vtable_bdd;
    
    public void calculateVTables() {
        // not very efficient.
        vtable_bdd = bdd.zero();
        for (int i1=0, n1=methodIndexMap.size(); i1<n1; ++i1) {
            jq_InstanceMethod m = (jq_InstanceMethod) methodIndexMap.get(i1);
            BDD method_bdd = T3.ithVar(i1);
            for (int i2=0, n2=heapobjIndexMap.size(); i2<n2; ++i2) {
                ConcreteTypeNode c = (ConcreteTypeNode) heapobjIndexMap.get(i2);
                jq_Reference r2 = (jq_Reference) c.getDeclaredType();
                if (r2 == null) continue;
                r2.prepare(); m.getDeclaringClass().prepare();
                if (!r2.isSubtypeOf(m.getDeclaringClass())) continue;
                BDD heapobj_bdd = H1.ithVar(i2);
                jq_InstanceMethod target = r2.getVirtualMethod(m.getNameAndDesc());
                if (target != null) {
                    int i3 = getTargetIndex(target);
                    BDD target_bdd = T4.ithVar(i3);
                    target_bdd.andWith(heapobj_bdd.id());
                    target_bdd.andWith(method_bdd.id());
                    vtable_bdd.orWith(target_bdd);
                }
                heapobj_bdd.free();
            }
            method_bdd.free();
        }
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
                if (TRACE) {
                    printSet("oldPt2", oldPt2, "V1xH1");
                }
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
                if (TRACE) {
                    printSet("before type filter", newPt2, "V1xH1");
                }
                BDD newPt3 = newPt2.and(typeFilter);
                if (TRACE) {
                    BDD temp = newPt2.apply(typeFilter, BDDFactory.diff);
                    printSet("removed by type filter", temp, "V1xH1");
                    temp.free();
                }
                // (V1 x H1)
                pointsTo = pointsTo.or(newPt3);
                
            } while (!oldPt2.equals(pointsTo));

            if (TRACE) {
                System.out.println("out of loop.");
                System.out.flush();
            }
            // propagate points-to set over field loads and stores
            /* --- rule (2) --- */
            //
            //   o2 \in pt(l)   l -> q.f   o1 \in pt(q)
            // -----------------------------------------
            //                  o2 \in pt(o1.f) 
            // (V1 x (V2 x FD)) * (V1 x H1) => ((V2 x FD) x H1)
            BDD tmpRel1 = stores.relprod(pointsTo, V1.set());
            if (TRACE) {
                printSet("tmpRel1", tmpRel1, "V2xFDxH1");
            }
            // ((V2 x FD) x H1) => ((V1 x FD) x H2)
            BDD tmpRel2 = tmpRel1.replace(V2ToV1).replace(H1ToH2);
            // ((V1 x FD) x H2) * (V1 x H1) => ((H1 x FD) x H2)
            fieldPt = tmpRel2.relprod(pointsTo, V1.set());
            if (TRACE) {
                printSet("fieldPt", fieldPt, "H1xFDxH2");
            }

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
            if (TRACE) {
                printSet("newPt5", newPt5, "V1xH1");
            }

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
            BDD tmpRel1 = stores.relprod(newPointsTo, V1set);
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

            BDD newFieldPt = storePt.relprod(newPointsTo, V1set);
            // (H1xFD)xH2
            newFieldPt.orWith(newStorePt.relprod(oldPointsTo, V1set));
            newStorePt.free();
            oldPointsTo.free();
            // (H1xFD)xH2
            newFieldPt.applyWith(fieldPt.id(), BDDFactory.diff);
            // cache fieldPt
            fieldPt.orWith(newFieldPt.id()); // (H1xFD)xH2

            // apply rule (3)
            BDD tmpRel4 = loads.relprod(newPointsTo, V1set);
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

    public static class IndexMap {
        private final String name;
        private final HashMap hash;
        private final ArrayList list;
        
        public IndexMap(String name) {
            this.name = name;
            hash = new HashMap();
            list = new ArrayList();
        }
        
        public int get(Object o) {
            Integer i = (Integer) hash.get(o);
            if (i == null) {
                hash.put(o, i = new Integer(list.size()));
                list.add(o);
                if (TRACE) System.out.println(this+"["+i+"] = "+o);
            }
            return i.intValue();
        }
        
        public Object get(int i) {
            return list.get(i);
        }
        
        public boolean contains(Object o) {
            return hash.containsKey(o);
        }
        
        public int size() {
            return list.size();
        }
        
        public String toString() {
            return name;
        }
        
    }

}

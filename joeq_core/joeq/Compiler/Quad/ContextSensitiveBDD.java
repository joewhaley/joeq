package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.BuDDyFactory;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Quad.BDDPointerAnalysis.IndexMap;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.UnknownTypeNode;
import Main.HostedVM;
import Util.Collections.WorkSet;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley
 */
public class ContextSensitiveBDD {
    
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
    int domainBits[] = {18, 18, 18, 18, 13, 14, 14};
    // to be computed in sysInit function
    int domainSpos[] = {0, 0, 0, 0, 0, 0, 0}; 

    // V1 V2 are domains for variables 
    // H1 H2 are domains for heap objects
    // FD is a domain for field signature
    BDDDomain V1, V2, V3, V4, FD, H1, H2;
    // T1 and T2 are used to compute typeFilter
    // T1 = V2, and T2 = V1
    BDDDomain T1, T2, T3, T4; 

    // domain pairs for bdd_replace
    BDDPairing V1ToV2;
    BDDPairing V2ToV1;
    BDDPairing V1ToV3;
    BDDPairing V2ToV4;
    BDDPairing V3ToV4;
    BDDPairing V4ToV3;
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

    public ContextSensitiveBDD() {
        this(DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }

    public ContextSensitiveBDD(int nodeCount, int cacheSize) {
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
        V4 = bdd_domains[3];
        FD = bdd_domains[4];
        H1 = bdd_domains[5];
        H2 = bdd_domains[6];
        T1 = V2;
        T2 = V1;
        T3 = H2;
        T4 = V2;
    
        int varnum = bdd.varNum();
        int[] varorder = new int[varnum];
        //makeVarOrdering(varorder);
        for (int i=0; i<varorder.length; ++i) {
            //System.out.println("varorder["+i+"]="+varorder[i]);
        }
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
    
        V1ToV2 = bdd.makePair(V1, V2);
        V2ToV1 = bdd.makePair(V2, V1);
        V1ToV3 = bdd.makePair(V1, V3);
        V2ToV4 = bdd.makePair(V2, V4);
        V3ToV4 = bdd.makePair(V3, V4);
        V4ToV3 = bdd.makePair(V4, V3);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
        T2ToT1 = bdd.makePair(T2, T1);
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
    }

    public static void main(String[] args) {
        HostedVM.initialize();
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        ContextSensitiveBDD dis = new ContextSensitiveBDD();
        dis.reset();
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        if (DUMP)
            dis.dumpResults();
            
    }
    
    public void dumpResults() {
    }
    
    LinkedHashMap summaries = new LinkedHashMap();
    WorkSet worklist = new WorkSet();
    CallGraph initial_cg;
    
    public CallGraph go(Collection roots) {
        long time = System.currentTimeMillis();
        
        initial_cg = CHACallGraph.INSTANCE;
        
        Navigator navigator = initial_cg.getNavigator(roots);
        Set sccs = SCComponent.buildSCC(roots.toArray(), navigator);
        SCCTopSortedGraph graph = SCCTopSortedGraph.topSort(sccs);
        
        worklist.add(graph.getLast());
        
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        
        while (!worklist.isEmpty()) {
            SCComponent scc = (SCComponent) worklist.getFirst();
            Object[] nodes = scc.nodes();
            boolean change = false;
            for (int i=0; i<nodes.length; ++i) {
                jq_Method m = (jq_Method) nodes[i];
                BDDMethodSummary s = (BDDMethodSummary) summaries.get(m);
                if (s == null) {
                    summaries.put(m, s = new BDDMethodSummary(m));
                }
                if (s.visit()) {
                    change = true;
                }
            }
            if (change) {
                for (int j=0; j<scc.prevLength(); ++j) {
                    SCComponent prev = scc.prev(j);
                    worklist.add(prev);
                }
            }
        }
        
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
    
    public class BDDMethodSummary {
        
        MethodSummary ms;
        
        // instructions contained within this summary
        BDD stores;       // V1 x (V2 x FD) 
        BDD loads;        // (V2 x FD) x V1
        
        // mapping between our nodes and callee nodes
        BDD callMappings; // V2 x V4
        
        // relations
        BDD pointsTo;     // V1 x H1
        BDD edgeSet;      // V1 x V2
        BDD typeFilter;   // V1 x H1
        
        BDD allocsAndLoads; // V2 x V4

        public BDDMethodSummary(jq_Method m) {
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                ms = MethodSummary.getSummary(cfg);
                for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
                    Node n = (Node) i.next();
                    handleNode(n);
                }
            }
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

        public boolean visit() {
            boolean change = false;
            // find all methods that we call.
            for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
                ProgramLocation mc = (ProgramLocation) i.next();
                Collection targets = initial_cg.getTargetMethods(mc);
                for (Iterator j=targets.iterator(); j.hasNext(); ) {
                    jq_Method target = (jq_Method) j.next();
                    BDDMethodSummary s = (BDDMethodSummary) summaries.get(target);
                    if (s != null)
                        handleMethodCall(s);
                }
            }
            return change;
        }
        
        public void init() {
            stores = bdd.zero();
            loads = bdd.zero();
            allocsAndLoads = bdd.zero();
            
            callMappings = bdd.zero();
            
            pointsTo = bdd.zero();
            edgeSet = bdd.zero();
            typeFilter = bdd.one();
        }

        public void handleMethodCall(BDDMethodSummary that) {
            
            BDD tmpRel1, tmpRel2;
            BDD that_stores, that_loads;
            BDD V2set = V2.set(), V3set = V3.set(), V4set = V4.set(), FDset = FD.set();
            BDD V2andFDset = V2set.and(FDset);
            
            tmpRel1 = that.stores.replace(V1ToV3);
            that_stores = tmpRel1.replace(V2ToV4);
            tmpRel1.free();
            
            tmpRel1 = that.loads.replace(V1ToV3);
            that_loads = tmpRel1.replace(V2ToV4);
            tmpRel1.free();
            
            callMappings.orWith(that.allocsAndLoads.id());
            allocsAndLoads.orWith(that.allocsAndLoads.id());
            
            for (;;) {
                BDD newMappings;
                
                BDD oldMappings = callMappings.id();
                
                if (false) {
                    // match callee stores to caller loads.
                    // (V3x(V4xFD)) x (V2xV4) = (V3x(V2xFD))
                    tmpRel1 = that_stores.relprod(callMappings, V4set);
                    // (V3x(V2xFD)) x ((V2xFD)xV1) = (V1xV3)
                    newMappings = tmpRel1.relprod(tmpRel1, V2andFDset);
                    tmpRel1.free();
                } else {
                    newMappings = bdd.zero();
                }
                
                // match callee loads to caller stores.
                // ((V4xFD)xV3) x (V2xV4) = ((V2xFD)xV3)
                tmpRel1 = that_loads.relprod(callMappings, V4set);
                // ((V2xFD)xV3) x (V1x(V2xFD)) = (V1xV3)
                tmpRel2 = tmpRel1.relprod(this.stores, V2andFDset);
                tmpRel1.free();
                newMappings.orWith(tmpRel2);
                
                tmpRel1 = newMappings.replace(V1ToV2);
                newMappings = tmpRel1.replace(V3ToV4);
                tmpRel1.free();
                
                callMappings.orWith(newMappings);
                
                boolean exit = oldMappings.equals(callMappings);
                oldMappings.free();
                if (exit) break;
            }
            
            BDD callMappings2;
            
            tmpRel1 = callMappings.replace(V2ToV1);
            callMappings2 = tmpRel1.replace(V4ToV3);
            tmpRel1.free();
            
            // add callee stores to caller stores.
            // (V3x(V4xFD)) x (V2xV4) = (V3x(V2xFD))
            tmpRel1 = that_stores.relprod(callMappings, V4set);
            // (V3x(V2xFD)) x (V1xV3) = (V1x(V2xFD))
            tmpRel2 = tmpRel1.relprod(callMappings2, V3set);
            this.stores.orWith(tmpRel2);
            
            // add callee loads to caller loads.
            // ((V4xFD)xV3) x (V2xV4) = ((V2xFD)xV3)
            tmpRel1 = that_loads.relprod(callMappings, V4set);
            // ((V2xFD)xV3) x (V1xV3) = ((V2xFD)xV1)
            tmpRel2 = tmpRel1.relprod(callMappings2, V3set);
            this.loads.orWith(tmpRel2);
            
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
            
            BDD tmp = V2.ithVar(dest_i);
            tmp.andWith(V4.ithVar(dest_i));
            allocsAndLoads.orWith(tmp);
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
            }
            loads.orWith(dest_bdd);
            if (TRACE) {
                printSet("Loads-set is now", loads, "V1xV2xFD");
            }
            
            BDD tmp = V2.ithVar(dest_i);
            tmp.andWith(V4.ithVar(dest_i));
            allocsAndLoads.orWith(tmp);
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
                if (TRACE) {
                    printSet("Loads-set is now", loads, "V1xV2xFD");
                }
            
                BDD tmp = V2.ithVar(dest_i);
                tmp.andWith(V4.ithVar(dest_i));
                allocsAndLoads.orWith(tmp);
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
                }
                stores.orWith(src_bdd);
                if (TRACE) {
                    printSet("Stores-set is now", stores, "V1xV2xFD");
                }
            }
            base_bdd.free(); f_bdd.free();
        }
        
        public void addClassInit(jq_Type t) {
        }
        
        public void addAllocType(Node site, jq_Reference type) {
        }

        public void addVarType(Node var, jq_Reference type) {
        }

    }
}

// ContextSensitiveBDD.java, created Thu Mar  6  1:01:05 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
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
import Clazz.jq_Field;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.MethodSummary.ReturnedNode;
import Compil3r.Quad.MethodSummary.UnknownTypeNode;
import Main.HostedVM;
import Util.Assert;
import Util.Collections.HashWorklist;
import Util.Collections.IndexMap;
import Util.Graphs.Navigator;
import Util.Graphs.SCCTopSortedGraph;
import Util.Graphs.SCComponent;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ContextSensitiveBDD {
    
    public static final boolean TRACE = true;
    public static final boolean TRACE_INST = true;

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

    /** BDD domains that we use in the analysis. **/
    BDDDomain      V1, V2, V3, V4, FD, H1, H2, CS;
    /** Default sizes of the domains defined above. **/
    int dBits[] = {18, 18, 18, 18, 13, 14, 14,  8};
    /** Temporary domains that overlap with other domains. **/
    BDDDomain      T2, T1;

    /** Cached domain pairings for replace() operations. **/
    BDDPairing V1ToV2;
    BDDPairing V2ToV1;
    BDDPairing V1ToV3;
    BDDPairing V2ToV4;
    BDDPairing V3ToV1;
    BDDPairing V3ToV2;
    BDDPairing V3ToV4;
    BDDPairing V4ToV3;
    BDDPairing H1ToH2;
    BDDPairing H2ToH1;
    BDDPairing T2ToT1;

    /** Default constructor and initialization. **/
    public ContextSensitiveBDD() {
        this(DEFAULT_NODE_COUNT, DEFAULT_CACHE_SIZE);
    }

    /** Constructor and initialization. **/
    public ContextSensitiveBDD(int nodeCount, int cacheSize) {
        bdd = BuDDyFactory.init(nodeCount, cacheSize);
    
        bdd.setCacheRatio(4);
        bdd.setMaxIncrease(cacheSize);
    
        int[] domains = new int[dBits.length];
        for (int i=0; i<dBits.length; ++i) {
            domains[i] = (1 << dBits[i]);
        }
        BDDDomain[] bdd_domains = bdd.extDomain(domains);
        V1 = bdd_domains[0];
        V2 = bdd_domains[1];
        V3 = bdd_domains[2];
        V4 = bdd_domains[3];
        FD = bdd_domains[4];
        H1 = bdd_domains[5];
        H2 = bdd_domains[6];
        CS = bdd_domains[7];
        T1 = V2;
        T2 = V1;
    
        if (false) {
            int varnum = bdd.varNum();
            int[] varorder = new int[varnum];
            //makeVarOrdering(varorder);
            for (int i=0; i<varorder.length; ++i) {
                //System.out.println("varorder["+i+"]="+varorder[i]);
            }
            bdd.setVarOrder(varorder);
            bdd.enableReorder();
        }
    
        V1ToV2 = bdd.makePair(V1, V2);
        V2ToV1 = bdd.makePair(V2, V1);
        V1ToV3 = bdd.makePair(V1, V3);
        V2ToV4 = bdd.makePair(V2, V4);
        V3ToV1 = bdd.makePair(V3, V1);
        V3ToV2 = bdd.makePair(V3, V2);
        V3ToV4 = bdd.makePair(V3, V4);
        V4ToV3 = bdd.makePair(V4, V3);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
        T2ToT1 = bdd.makePair(T2, T1);
    }
    
    /** Returns the named domain. **/
    BDDDomain getDomain(String s) {
        try {
            java.lang.reflect.Field f = ContextSensitiveBDD.class.getDeclaredField(s);
            return (BDDDomain) f.get(this);
        } catch (NoSuchFieldException _) {
            Assert.UNREACHABLE("No such domain "+s);
            return null;
        } catch (IllegalAccessException _) {
            Assert.UNREACHABLE("Cannot access domain "+s);
            return null;
        }
    }

    public static void main(String[] args) {
        HostedVM.initialize();
        
        boolean DUMP = System.getProperty("bdddump") != null;
        
        ContextSensitiveBDD dis = new ContextSensitiveBDD();
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.prepare();
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        
        dis.go(roots);
        
        if (DUMP)
            dis.dumpResults();
            
    }
    
    public void dumpResults() {
    }
    
    LinkedHashMap summaries = new LinkedHashMap();
    HashWorklist worklist = new HashWorklist(false);
    CallGraph initial_cg;
    
    public CallGraph go(Collection roots) {
        long time = System.currentTimeMillis();
        
        /* Build the initial call graph using CHA. */
        initial_cg = new RootedCHACallGraph();
        initial_cg.setRoots(roots);
        /* Calculate the reachable methods once to touch each method,
           so that the set of types are stable. */
        initial_cg.calculateReachableMethods(roots);
        
        /* Build SCCs. */
        Navigator navigator = initial_cg.getNavigator();
        Set sccs = SCComponent.buildSCC(roots, navigator);
        SCCTopSortedGraph graph = SCCTopSortedGraph.topSort(sccs);
        
        /* Put SCCs on worklist in reverse order. */
        SCComponent scc = graph.getLast();
        while (scc != null) {
            worklist.push(scc);
            scc = scc.prevTopSort();
        }
        
        System.out.println("Initial setup:\t\t"+(System.currentTimeMillis()-time)/1000.+" seconds.");
        
        /* Iterate through worklist. */
        while (!worklist.isEmpty()) {
            scc = (SCComponent) worklist.pull();
            Object[] nodes = scc.nodes();
            boolean change = false;
            for (int i=0; i<nodes.length; ++i) {
                jq_Method m = (jq_Method) nodes[i];
                /* Get the cached summary for this method. */
                BDDMethodSummary s = (BDDMethodSummary) summaries.get(m);
                if (s == null) {
                    /* Not yet visited, build a new summary. */
                    summaries.put(m, s = new BDDMethodSummary(m));
                }
                if (s.visit()) {
                    change = true;
                }
            }
            if (change) {
                if (TRACE) System.out.println("Changed, adding predecessors to worklist.");
                if (scc.isLoop()) {
                    //if (TRACE) System.out.println("Adding self-loop to worklist: "+scc);
                    worklist.push(scc);
                }
                for (int j=0; j<scc.prevLength(); ++j) {
                    SCComponent prev = scc.prev(j);
                    //if (TRACE) System.out.println("Adding to worklist: "+prev);
                    worklist.push(prev);
                }
            }
        }
        
        return null;
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
        
        /** The method summary graph that this BDD summary represents.
         */
        MethodSummary ms;
        
        /** Store instructions in this summary, along with those propagated from
         * callees. V1 x (V2 x FD)   (v2.fd = v1;)
         */
        BDD stores;
        
        /** Load instructions in this summary, along with those propagated from
         * callees. V1 x (V2 x FD)   (v1 = v2.fd;)
         */
        BDD loads;
        
        /** Collection of the targets of all allocations and loads, along with
         * those propagated from callees.  This is stored as V2xV4 for convenient
         * combination with callMappings.
         */
        BDD allocsAndLoads; // V2 x V4
        
        /** Mapping between our nodes and callee nodes.
         */
        BDD allCallMappings; // V2 x CS x V4
        
        // relations
        BDD pointsTo;     // V1 x H1
        BDD edgeSet;      // V1 x V2   (v2 = v1;)
        BDD typeFilter;   // V1 x H1
        
        /** Indices for the call sites in this method.
         */
        IndexMap/* ProgramLocation->index */ callSiteIndexMap = new IndexMap("CallSite");
        
        int getCallSite(ProgramLocation mc) {
            return callSiteIndexMap.get(mc);
        }
        
        public BDDMethodSummary(jq_Method m) {
            this.reset();
            if (m.getBytecode() != null) {
                if (TRACE) System.out.println("Generating summary for method "+m);
                ControlFlowGraph cfg = CodeCache.getCode(m);
                ms = MethodSummary.getSummary(cfg);
                for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
                    Node n = (Node) i.next();
                    handleNode(n);
                }
                for (Iterator i=UnknownTypeNode.getAll().iterator(); i.hasNext(); ) {
                    Node n = (Node) i.next();
                    handleNode(n);
                }
            } else {
                if (TRACE) System.out.println("Skipping native method "+m);
            }
        }
        
        void reset() {
            stores = bdd.zero();
            loads = bdd.zero();
            allCallMappings = bdd.zero();
            pointsTo = bdd.zero();
            edgeSet = bdd.zero();
            typeFilter = bdd.zero();
            allocsAndLoads = bdd.zero();
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
            if (this.ms == null) {
                return false;
            }
            if (TRACE) System.out.println("Visiting "+this.ms.getMethod());
            boolean change = false;
            // find all methods that we call.
            for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
                ProgramLocation mc = (ProgramLocation) i.next();
                Collection targets = initial_cg.getTargetMethods(mc);
                for (Iterator j=targets.iterator(); j.hasNext(); ) {
                    jq_Method target = (jq_Method) j.next();
                    BDDMethodSummary s = (BDDMethodSummary) summaries.get(target);
                    if (s != null) {
                        if (handleMethodCall(mc, s)) {
                            if (TRACE) System.out.println("Method call "+mc+"->"+s.ms.getMethod()+" changed");
                            change = true;
                        }
                    }
                }
            }
            return change;
        }
        
        public boolean handleMethodCall(ProgramLocation mc, BDDMethodSummary that) {
            
            if (that.ms == null) {
                if (TRACE_INST) System.out.println("Skipping call: "+mc);
                return false;
            }
            
            // cache to check for changes.
            BDD oldLoads = this.loads.id();
            BDD oldStores = this.stores.id();
            BDD oldAllCallMappings = this.allCallMappings.id();
            
            if (TRACE_INST) System.out.println("Handling call graph edge: "+mc+"->"+that.ms.getMethod());
            
            BDD callSite = CS.ithVar(getCallSite(mc));
            BDD myCallMappings = allCallMappings.restrict(callSite);
            
            /* Add mapping between actual and formal parameters to myCallMappings. */
            for (int i=0; i<mc.getNumParams(); ++i) {
                PassedParameter pp = new PassedParameter(mc, i);
                Set s = this.ms.getNodesThatCall(pp);
                ParamNode pn = that.ms.getParamNode(i);
                addParameterPass(myCallMappings, pn, s);
            }
            
            BDD tmpRel1, tmpRel2;
            BDD that_stores, that_loads;
            BDD V1set = V1.set(), V2set = V2.set(), V3set = V3.set(), V4set = V4.set(), FDset = FD.set();
            BDD V2andFDset = V2set.and(FDset);
            
            /* Replace domains in callee "stores" BDD to become:
               (V3x(V4xFD))   (v4.fd = v3;)
             */
            tmpRel1 = that.stores.replace(V1ToV3);
            that_stores = tmpRel1.replace(V2ToV4);
            tmpRel1.free();
            
            if (TRACE) {
                printSet("that stores", that_stores, "V3xV4xFD");
            }
            
            /* Replace domains in callee "loads" BDD to become:
               (V3x(V4xFD))   (v3 = v4.fd;)
             */
            tmpRel1 = that.loads.replace(V1ToV3);
            that_loads = tmpRel1.replace(V2ToV4);
            tmpRel1.free();
            
            if (TRACE) {
                printSet("that loads", that_loads, "V3xV4xFD");
            }
            
            /* Map all allocs and loads in callee directly into caller. */
            if (TRACE) {
                printSet("that allocs and loads", that.allocsAndLoads, "V2xV4");
            }
            myCallMappings.orWith(that.allocsAndLoads.id());
            
            /* Include all callee allocs and loads in caller allocs and loads set. */
            allocsAndLoads.orWith(that.allocsAndLoads.id());
            
            /* Loop to match read/write edges, until there is no change to the mapping. */
            for (;;) {
                BDD newMappings;
                
                /* Make a copy of the mappings for comparison later, for termination condition. */
                BDD oldMappings = myCallMappings.id();
                
                if (true) {
                    /* match callee stores to caller loads. */
                    // (V3x(V4xFD)) x (V2xCSxV4) = (V3x(V2xFD)xCS)
                    tmpRel1 = that_stores.relprod(myCallMappings, V4set);
                    // (V3x(V2xFD)) x (V1x(V2xFD)) = (V1xV3)
                    newMappings = tmpRel1.relprod(this.loads, V2andFDset);
                    tmpRel1.free();
                } else {
                    newMappings = bdd.zero();
                }
                
                /* match callee loads to caller stores. */
                // (V3x(V4xFD)) x (V2xV4) = (V3x(V2xFD))
                tmpRel1 = that_loads.relprod(oldMappings, V4set);
                // (V3x(V2xFD)) x (V1x(V2xFD)) = (V1xV3)
                tmpRel2 = tmpRel1.relprod(this.stores, V2andFDset);
                tmpRel1.free();
                newMappings.orWith(tmpRel2);
                
                /* replace domains in newMappings: (V1xV3) -> (V2xV4) */
                tmpRel1 = newMappings.replace(V1ToV2);
                newMappings = tmpRel1.replace(V3ToV4);
                tmpRel1.free();
                
                /* add to myCallMappings */
                myCallMappings.orWith(newMappings);
                
                if (TRACE) {
                    printSet("my call mappings", myCallMappings, "V2xV4");
                }
                
                /* exit if callMappings changed */
                boolean exit = oldMappings.equals(myCallMappings);
                tmpRel1.free();
                oldMappings.free();
                if (exit) break;
            }
            
            /* Now that mapping is done, add mapping for return nodes. */
            ReturnedNode rn = (ReturnedNode) this.ms.callToRVN.get(mc);
            if (rn != null) {
                addReturnValue(myCallMappings, rn, that.ms.returned);
            }
            rn = (ReturnedNode) this.ms.callToTEN.get(mc);
            if (rn != null && !that.ms.thrown.isEmpty())
                addReturnValue(myCallMappings, rn, that.ms.thrown);
            
            if (TRACE) {
                printSet("final call mappings", myCallMappings, "V2xV4");
            }
            
            /* Build myCallMappings2, which is (V1xV3) */
            BDD myCallMappings2;
            tmpRel1 = myCallMappings.restrict(callSite);
            tmpRel2 = tmpRel1.replace(V2ToV1);
            myCallMappings2 = tmpRel2.replace(V4ToV3);
            tmpRel1.free();
            
            /* add callee stores to caller stores. */
            // (V3x(V4xFD)) x (V2xV4) = (V3x(V2xFD))
            tmpRel1 = that_stores.relprod(myCallMappings, V4set);
            // (V3x(V2xFD)) x (V1xV3) = (V1x(V2xFD))
            tmpRel2 = tmpRel1.relprod(myCallMappings2, V3set);
            this.stores.orWith(tmpRel2);
            
            if (TRACE) {
                printSet("this stores now", this.stores, "V1xV2xFD");
            }
            
            /* Add edges to/from return nodes to the mapped return nodes. */
            tmpRel1 = this.edgeSet.replace(V1ToV3);
            // (v2=v3;) (v2.fd=v1;) -> (v3.fd=v1;)
            // (V2xV3) x (V1xV2xFD) = (V1xV3xFD)
            tmpRel2 = tmpRel1.relprod(this.stores, V2set);
            tmpRel1.free();
            tmpRel1 = tmpRel2.replace(V3ToV2);
            tmpRel2.free();
            this.stores.orWith(tmpRel1);
            
            tmpRel2 = this.edgeSet.replace(V1ToV3);
            tmpRel1 = tmpRel2.replace(V2ToV1);
            tmpRel2.free();
            // (v1=v3;) (v2.fd=v1;) -> (v2.fd=v3;)
            // (V1xV3) x (V1xV2xFD) = (V2xV3xFD)
            tmpRel2 = tmpRel1.relprod(this.stores, V1set);
            tmpRel1.free();
            tmpRel1 = tmpRel2.replace(V3ToV1);
            tmpRel2.free();
            this.stores.orWith(tmpRel1);
            
            /* Add callee loads to caller loads. */
            // (V3x(V4xFD)) x (V2xV4) = (V3x(V2xFD))
            tmpRel1 = that_loads.relprod(myCallMappings, V4set);
            // (V3x(V2xFD)) x (V1xV3) = (V1x(V2xFD))
            tmpRel2 = tmpRel1.relprod(myCallMappings2, V3set);
            myCallMappings2.free();
            this.loads.orWith(tmpRel2);
            
            if (TRACE) {
                printSet("this loads now", this.loads, "V1xV2xFD");
            }
            
            /* Add the current call mappings to the set of all call mappings. */
            myCallMappings.andWith(callSite);
            callSite.free();
            allCallMappings.orWith(myCallMappings);
            
            /* Check if anything changed. */
            boolean changed;
            changed = !oldLoads.equals(this.loads) || !oldStores.equals(this.stores) ||
                      !oldAllCallMappings.equals(this.allCallMappings);
            oldLoads.free(); oldStores.free(); oldAllCallMappings.free();
            return changed;
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
        
        public void addParameterPass(BDD callMappings, ParamNode dest, Set srcs) {
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V4.ithVar(dest_i);
            for (Iterator i=srcs.iterator(); i.hasNext(); ) {
                Node src = (Node) i.next();
                int src_i = getVariableIndex(src);
                BDD src_bdd = V2.ithVar(src_i);
                src_bdd.andWith(dest_bdd.id());
                if (TRACE_INST) {
                    System.out.println("Adding parameter pass dest="+dest_i+" src="+src_i);
                }
                callMappings.orWith(src_bdd);
                if (TRACE) {
                    printSet("callMappings is now", callMappings, "V2xV4");
                }
            }
            dest_bdd.free();
        }
    
        public void addReturnValue(BDD callMappings, ReturnedNode dest, Set srcs) {
            int dest_i = getVariableIndex(dest);
            BDD dest_bdd = V2.ithVar(dest_i);
            BDD V4set = V4.set();
            for (Iterator i=srcs.iterator(); i.hasNext(); ) {
                Node src = (Node) i.next();
                int src_i = getVariableIndex(src);
                BDD src_bdd = V4.ithVar(src_i);
                if (TRACE_INST) {
                    printSet("Return value in callee", src_bdd, "V4");
                }
                // V4 x (V2 x V4) = V2
                BDD src_bdd2 = src_bdd.relprod(callMappings, V4set);
                src_bdd.free();
                if (TRACE_INST) {
                    printSet("Return value corresponds to", src_bdd2, "V2");
                }
                src_bdd = src_bdd2.replace(V2ToV1);
                src_bdd2.free();
                src_bdd.andWith(dest_bdd.id());
                if (TRACE_INST) {
                    printSet("Adding return value", src_bdd, "V1xV2");
                }
                edgeSet.orWith(src_bdd);
                if (TRACE) {
                    printSet("edgeSet is now", edgeSet, "V1xV2");
                }
            }
            V4set.free();
            dest_bdd.free();
        }
            
        public void addClassInit(jq_Type t) {
        }
        
        public void addAllocType(Node site, jq_Reference type) {
        }

        public void addVarType(Node var, jq_Reference type) {
        }

    }
    
    String getElementName(String desc, int index) {
        if ("V1".equals(desc) ||
            "V2".equals(desc) ||
            "V3".equals(desc) ||
            "V4".equals(desc)) return this.variableIndexMap.get(index).toString();
        if ("H1".equals(desc) ||
            "H2".equals(desc)) return this.heapobjIndexMap.get(index).toString();
        if ("FD".equals(desc)) return this.fieldIndexMap.get(index).toString();
        return null;
    }
    
    void printSet(String desc, BDD b, String type) {
        java.io.PrintStream ofile = System.out;
        
        ofile.print(desc);
        ofile.print(": ");
        
        if (b.equals(bdd.zero())) {
            ofile.print("F");
        } else if (b.equals(bdd.one())) {
            ofile.print("T");
        } else {
            int[] set = new int[bdd.varNum()];
            fdd_printset_rec(type, ofile, b, set);
            //free(set);
        }
        
        System.out.println();
    }
    
    void fdd_printset_rec(String type, java.io.PrintStream ofile, BDD r, int[] set) {
        int n, m, i;
        boolean used = false;
        int[] var;
        boolean[] binval;
        boolean ok, first;
   
        if (r.equals(bdd.zero()))
            return;
        else if (r.equals(bdd.one())) {
            ofile.print("<");
            first = true;

            for (StringTokenizer st = new StringTokenizer(type, "x_", false);
                st.hasMoreTokens(); ) {
                String domainName = st.nextToken();
                BDDDomain domain_n = getDomain(domainName);
                n = domain_n.getIndex();
                
                boolean firstval = true;
                used = false;

                int[] domain_n_ivar = domain_n.vars();
                int domain_n_varnum = domain_n.varNum();
                for (m=0 ; m<domain_n_varnum ; m++)
                    if (set[domain_n_ivar[m]] != 0)
                        used = true;
     
                if (used) {
                    if (!first)
                        ofile.print(", ");
                    first = false;
                    ofile.print(domainName);
                    ofile.print(":");

                    var = domain_n_ivar;
                    
                    for (m=0 ; m<(1<<domain_n_varnum) ; m++) {
                        binval = fdddec2bin(n, m);
                        ok = true;
           
                        for (i=0 ; i<domain_n_varnum && ok ; i++)
                            if (set[var[i]] == 1  &&  binval[i] != false)
                                ok = false;
                            else if (set[var[i]] == 2  &&  binval[i] != true)
                                ok = false;

                        if (ok) {
                            if (!firstval)
                                ofile.print("/");
                            ofile.print(getElementName(domainName, m));
                            firstval = false;
                        }

                        //free(binval);
                    }
                }
            }

            ofile.print(">");
        } else {
            set[r.var()] = 1;
            fdd_printset_rec(type, ofile, r.low(), set);
      
            set[r.var()] = 2;
            fdd_printset_rec(type, ofile, r.high(), set);
      
            set[r.var()] = 0;
        }
    }
    
    void fdd_printset_rec(java.io.PrintStream ofile, BDD r, int[] set) {
        int fdvarnum = bdd.numberOfDomains();
        
        int n, m, i;
        boolean used = false;
        int[] var;
        boolean[] binval;
        boolean ok, first;
   
        if (r.equals(bdd.zero()))
            return;
        else if (r.equals(bdd.one())) {
            ofile.print("<");
            first = true;

            for (n=0 ; n<fdvarnum ; n++) {
                boolean firstval = true;
                used = false;

                BDDDomain domain_n = bdd.getDomain(n);

                int[] domain_n_ivar = domain_n.vars();
                int domain_n_varnum = domain_n.varNum();
                for (m=0 ; m<domain_n_varnum ; m++)
                    if (set[domain_n_ivar[m]] != 0)
                        used = true;
     
                if (used) {
                    if (!first)
                        ofile.print(", ");
                    first = false;
                    ofile.print(n);
                    ofile.print(":");

                    var = domain_n_ivar;
        
                    for (m=0 ; m<(1<<domain_n_varnum) ; m++) {
                        binval = fdddec2bin(n, m);
                        ok = true;
           
                        for (i=0 ; i<domain_n_varnum && ok ; i++)
                            if (set[var[i]] == 1  &&  binval[i] != false)
                                ok = false;
                            else if (set[var[i]] == 2  &&  binval[i] != true)
                                ok = false;

                        if (ok) {
                            if (!firstval)
                                ofile.print("/");
                            ofile.print(m);
                            firstval = false;
                        }

                        //free(binval);
                    }
                }
            }

            ofile.print(">");
        } else {
            set[r.var()] = 1;
            fdd_printset_rec(ofile, r.low(), set);
      
            set[r.var()] = 2;
            fdd_printset_rec(ofile, r.high(), set);
      
            set[r.var()] = 0;
        }
    }
    
    boolean[] fdddec2bin(int var, int val) {
        boolean[] res;
        int n = 0;

        res = new boolean[bdd.getDomain(var).varNum()];

        while (val > 0) {
            if ((val & 0x1) != 0)
                res[n] = true;
            val >>= 1;
            n++;
        }

        return res;
    }
    
}

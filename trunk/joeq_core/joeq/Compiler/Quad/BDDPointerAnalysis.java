
package Compil3r.Quad;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Compil3r.BytecodeAnalysis.CallTargets;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.OutsideNode;
import Main.HostedVM;
import Main.jq;
import Run_Time.TypeCheck;

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
    BDDDomain T1, T2; 

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
        
        int varnum = bdd.varNum();
        int[] varorder = new int[varnum];
        //makeVarOrdering(varorder);
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
        
        V1ToV2 = bdd.makePair(V1, V2);
        V2ToV1 = bdd.makePair(V2, V1);
        H1ToH2 = bdd.makePair(H1, H2);
        H2ToH1 = bdd.makePair(H2, H1);
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

    public static void main(String[] args) {
        HostedVM.initialize();
        
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
        dis.calculateTypeFilter();
        dis.solveIncremental();
        //dis.pointsTo.printSetWithDomains();
    }

    public void handleMethodSummary(MethodSummary ms) {
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
                }
            }
            if (n instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) n;
                addObjectAllocation(ctn, ctn);
                addAllocType(ctn, (jq_Reference) ctn.getDeclaredType());
            }
            addVarType(n, (jq_Reference) n.getDeclaredType());
        }
        
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation)i.next();
        }
    }

    Map/* Node->index */ variableIndexMap = new HashMap();
    Map/* ConcreteTypeNode->index */ heapobjIndexMap = new HashMap();
    Map/* jq_Field->index */ fieldIndexMap = new HashMap();
    Map/* jq_Reference->index */ typeIndexMap = new HashMap();

    private int getIndex(Map m, Object o) {
        Integer i = (Integer) m.get(o);
        int j;
        if (i == null) {
            j = m.size();
            m.put(o, new Integer(j));
        } else {
            j = i.intValue();
        }
        return j;
    }

    int getVariableIndex(Node dest) {
        return getIndex(variableIndexMap, dest);
    }
    int getHeapobjIndex(ConcreteTypeNode site) {
        return getIndex(heapobjIndexMap, site);
    }
    int getFieldIndex(jq_Field f) {
        return getIndex(fieldIndexMap, f);
    }
    int getTypeIndex(jq_Reference f) {
        return getIndex(typeIndexMap, f);
    }

    public void addObjectAllocation(Node dest, ConcreteTypeNode site) {
        int dest_i = getVariableIndex(dest);
        int site_i = getHeapobjIndex(site);
        BDD dest_bdd = V1.ithVar(dest_i);
        BDD site_bdd = H1.ithVar(site_i);
        dest_bdd.andWith(site_bdd);
        pointsTo.orWith(dest_bdd);
    }

    public void addDirectAssignment(Node dest, Node src) {
        int dest_i = getVariableIndex(dest);
        int src_i = getVariableIndex(src);
        BDD dest_bdd = V2.ithVar(dest_i);
        BDD src_bdd = V1.ithVar(src_i);
        dest_bdd.andWith(src_bdd);
        edgeSet.orWith(dest_bdd);
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
        loads.orWith(dest_bdd);
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
            loads.orWith(dest_bdd);
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
        stores.orWith(base_bdd);
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
            stores.orWith(base_bdd);
        }
        base_bdd.free(); f_bdd.free();
    }
    
    public void addClassType(jq_Reference type) {
        if (type == null) return;
        if (typeIndexMap.containsKey(type)) return;
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
        aC.orWith(type_bdd);
    }

    public void addVarType(Node var, jq_Reference type) {
        addClassType(type);
        int var_i = getVariableIndex(var);
        int type_i = getTypeIndex(type);
        BDD var_bdd = V1.ithVar(var_i);
        BDD type_bdd = T1.ithVar(type_i);
        type_bdd.andWith(var_bdd);
        vC.orWith(type_bdd);
    }
    
    void calculateTypeHierarchy() {
        // not very efficient.
        for (Iterator i=typeIndexMap.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e1 = (Map.Entry) i.next();
            jq_Type t1 = (jq_Type) e1.getKey();
            t1.prepare();
            int i1 = ((Integer) e1.getValue()).intValue();
            for (Iterator j=typeIndexMap.entrySet().iterator(); j.hasNext(); ) {
                Map.Entry e2 = (Map.Entry) j.next();
                jq_Type t2 = (jq_Type) e2.getKey();
                t2.prepare();
                int i2 = ((Integer) e2.getValue()).intValue();
                if (TypeCheck.isAssignable(t1, t2)) {
                    BDD type1_bdd = T1.ithVar(i1);
                    BDD type2_bdd = T1.ithVar(i2);
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
        cC.free(); aC.free(); vC.free();
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
                BDD newPt1 = edgeSet.relprod(pointsTo, V1.set());
                BDD newPt2 = newPt1.replace(V2ToV1);

                /* --- apply type filtering and merge into pointsTo relation --- */
                BDD newPt3 = newPt2.and(typeFilter);
                pointsTo = pointsTo.or(newPt3);
                
            } while (!oldPt2.equals(pointsTo));

            // propagate points-to set over field loads and stores
            /* --- rule (2) --- */
            //
            //   o2 \in pt(l)   l -> q.f   o1 \in pt(q)
            // -----------------------------------------
            //                  o2 \in pt(o1.f) 
            BDD tmpRel1 = stores.relprod(pointsTo, V1.set());
            // (V2xFD)xH1
            BDD tmpRel2 = tmpRel1.replace(V2ToV1).replace(H1ToH2);
            // (V1xFD)xH2
            fieldPt = tmpRel2.relprod(pointsTo, V1.set());
            // (H1xFD)xH2

            /* --- rule (3) --- */
            //
            //   p.f -> l   o1 \in pt(p)   o2 \in pt(o1)
            // -----------------------------------------
            //                 o2 \in pt(l)
            BDD tmpRel3 = loads.relprod(pointsTo, V1.set());
            // (H1xFD)xV2
            BDD newPt4 = tmpRel3.relprod(fieldPt, H1.set().and(FD.set()));
            // V2xH2
            BDD newPt5 = newPt4.replace(V2ToV1).replace(H2ToH1);
            // V1xH2

            /* --- apply type filtering and merge into pointsTo relation --- */
            BDD newPt6 = newPt5.and(typeFilter);
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

}

/*
 * AndersenPointerAnalysis.java
 *
 * Created on April 24, 2002, 3:24 PM
 */

package Compil3r.Quad;

import Clazz.*;
import Compil3r.BytecodeAnalysis.CallTargets;
import MethodSummary.MethodCall;
import MethodSummary.PassedParameter;
import MethodSummary.Node;
import MethodSummary.ConcreteTypeNode;
import MethodSummary.OutsideNode;
import MethodSummary.FieldNode;
import MethodSummary.ParamNode;
import MethodSummary.ReturnValueNode;
import MethodSummary.ThrownExceptionNode;
import Operator.Invoke;
import Operand.ParamListOperand;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedList;
import Util.LinkedHashSet;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class AndersenPointerAnalysis {

    public static java.io.PrintStream out = System.out;
    public static final boolean TRACE = true;

    public static final class AndersenVisitor implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            INSTANCE.visitMethod(cfg);
            INSTANCE.doWorklist();
        }
    }
    
    /** Maps nodes to their set of corresponding nodes. */
    HashMap nodesToCorrespondingNodes;
    
    /** Worklist of operations to perform. */
    LinkedHashSet worklist;
    
    HashSet visitedMethods;
    
    /** Creates new AndersenPointerAnalysis */
    public AndersenPointerAnalysis() {
        nodesToCorrespondingNodes = new HashMap();
        worklist = new LinkedHashSet();
        visitedMethods = new HashSet();
    }

    public static AndersenPointerAnalysis INSTANCE = new AndersenPointerAnalysis();
    
    void visitMethod(ControlFlowGraph cfg) {
        if (TRACE) out.println("Visiting method: "+cfg.getMethod());
        visitedMethods.add(cfg);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            MethodCall mc = (MethodCall)i.next();
            if (TRACE) out.println("Found call: "+mc);
            CallTargets ct = mc.getCallTargets();
            if (TRACE) out.println("Possible targets ignoring type information: "+ct);
            HashSet definite_targets = new HashSet();
            if (ct.size() == 1 && ct.isComplete()) {
                // call can be statically resolved to a single target.
                if (TRACE) out.println("Call is statically resolved to a single target.");
                definite_targets.add(ct.iterator().next());
            } else {
                // use the type information about the receiver object to find targets.
                HashSet set = new HashSet();
                PassedParameter pp = new PassedParameter(mc, 0);
                ms.getNodesThatCall(pp, set);
                if (TRACE) out.println("Possible nodes for receiver object: "+set);
                for (Iterator j=set.iterator(); j.hasNext(); ) {
                    Node base = (Node)j.next();
                    if (TRACE) out.println("Checking base node: "+base);
                    CallTargets ct2 = mc.getCallTargets(base);
                    if (ct2.size() == 1 && ct2.isComplete()) {
                        if (TRACE) out.println("Using node "+base+", call is statically resolved to a single target.");
                        definite_targets.add(ct2.iterator().next());
                    } else {
                        // TODO: if it is an UnknownTypeNode, need to be notified when
                        // a new subclass is loaded.
                        SetOfNodes son = (SetOfNodes)nodesToCorrespondingNodes.get(base);
                        if (TRACE) out.println("Node "+base+" corresponds to set "+son);
                        if (son != null) {
                            Set klasses = son.getNodeTypes();
                            if (TRACE) out.println("Types for node "+base+": "+klasses);
                            CallTargets ct3 = mc.getCallTargets(klasses, true);
                            if (TRACE) out.println("Targets given those types: "+ct3);
                            definite_targets.addAll(ct3);
                            addToWorklist(base);
                        } else {
                            son = new SetOfNodes((HashSet)null);
                            addMapping(base, son); // automatically adds to worklist
                        }
                        CallTargetListener ctl = new CallTargetListener(ms, mc, definite_targets);
                        son.addTypeListener(ctl);
                        if (base instanceof FieldNode) {
                            if (TRACE) out.println("Node "+base+" is a field node, adding its predecessors to the worklist.");
                            addPredecessorsToWorklist((FieldNode)base, new HashSet());
                        }
                    }
                }
            }
            if (TRACE) out.println("Set of definite targets of "+mc+": "+definite_targets);
            for (Iterator j=definite_targets.iterator(); j.hasNext(); ) {
                jq_Method callee = (jq_Method)j.next();
                callee.getDeclaringClass().load();
                if (callee.getBytecode() == null) {
                    out.println(callee+" is a native method, skipping analysis...");
                    continue;
                }
                ControlFlowGraph callee_cfg = CodeCache.getCode(callee);
                MethodSummary callee_summary = MethodSummary.getSummary(callee_cfg);
                addParameterAndReturnMappings(ms, mc, callee_summary);
                if (!visitedMethods.contains(callee_cfg))
                    visitMethod(callee_cfg);
            }
        }
    }

    public static class CallTargetListener {
        MethodSummary ms; MethodCall mc; Set currentResult;
        CallTargetListener(MethodSummary ms, MethodCall mc, Set currentResult) {
            this.ms = ms; this.mc = mc; this.currentResult = currentResult;
        }
        void addType(jq_Reference type) {
            if (TRACE) out.println("Checking if type "+type+" adds a new target for "+mc);
            CallTargets ct = mc.getCallTargets(type, true);
            if (TRACE) out.println("Targets: "+ct);
            Iterator i = ct.iterator();
            while (i.hasNext()) {
                jq_Method callee = (jq_Method)i.next();
                if (currentResult.contains(callee)) continue;
                if (TRACE) out.println(callee+" is a new target");
                currentResult.add(callee);
                callee.getDeclaringClass().load();
                if (callee.getBytecode() == null) {
                    out.println(callee+" is a native method, skipping analysis...");
                    continue;
                }
                ControlFlowGraph callee_cfg = CodeCache.getCode(callee);
                MethodSummary callee_summary = MethodSummary.getSummary(callee_cfg);
                INSTANCE.addParameterAndReturnMappings(ms, mc, callee_summary);
                if (!INSTANCE.visitedMethods.contains(callee_cfg))
                    INSTANCE.visitMethod(callee_cfg);
            }
        }
    }
    
    void addPredecessorsToWorklist(FieldNode node, HashSet visited) {
        if (visited.contains(node)) return;
        visited.add(node);
        for (Iterator i=node.field_predecessors.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if (TRACE) out.println("Adding predecessor node "+n);
            SetOfNodes son = (SetOfNodes)nodesToCorrespondingNodes.get(n);
            if (son != null) {
                addToWorklist(n);
            } else {
                son = new SetOfNodes((HashSet)null);
                addMapping(n, son); // automatically adds to worklist
            }
            if (n instanceof FieldNode)
                addPredecessorsToWorklist((FieldNode)n, visited);
        }
    }
    
    void doWorklist() {
        while (!worklist.isEmpty()) {
            Iterator i = worklist.iterator();
            Node n = (Node)i.next(); i.remove();
            SetOfNodes son = (SetOfNodes)nodesToCorrespondingNodes.get(n);
            if (TRACE) out.println("Worklist: matching edges of node "+n+" to "+son);
            matchEdges(n, son);
        }
    }
    
    void addToWorklist(Node n) {
        boolean b = worklist.add(n);
        if (TRACE && b) out.println("Added node "+n+" to worklist.");
    }
    
    boolean addMapping(Node n, HashSet s) {
        boolean change;
        SetOfNodes son = (SetOfNodes)nodesToCorrespondingNodes.get(n);
        if (son != null) {
            change = son.addAll(s);
        } else {
            nodesToCorrespondingNodes.put(n, son = new SetOfNodes(s));
            change = true;
        }
        if (TRACE && change) out.println("Node "+n+": Added nodes "+s);
        if (change) addToWorklist(n);
        return change;
    }
    
    boolean addMapping(Node n, SetOfNodes s) {
        boolean change;
        SetOfNodes son = (SetOfNodes)nodesToCorrespondingNodes.get(n);
        if (son != null) {
            change = son.addAll(s);
        } else {
            nodesToCorrespondingNodes.put(n, s);
            change = true;
        }
        if (change) addToWorklist(n);
        if (TRACE && change) out.println("Node "+n+": Added set of nodes "+s);
        return change;
    }
    
    void addParameterAndReturnMappings(MethodSummary caller, MethodCall mc, MethodSummary callee) {
        ParamListOperand plo = Invoke.getParamList(mc.q);
        for (int i=0; i<plo.length(); ++i) {
            jq_Type t = plo.get(i).getType();
            if (!(t instanceof jq_Reference)) continue;
            ParamNode pn = callee.getParamNode(i);
            PassedParameter pp = new PassedParameter(mc, i);
            HashSet s = new HashSet();
            caller.getNodesThatCall(pp, s);
            //s.add(pn);
            addMapping(pn, s);
        }
        ReturnValueNode rvn = (ReturnValueNode)caller.nodes.get(new ReturnValueNode(mc));
        if (rvn != null) {
            HashSet s = (HashSet)callee.returned.clone();
            //s.add(rvn);
            addMapping(rvn, s);
        }
        ThrownExceptionNode ten = (ThrownExceptionNode)caller.nodes.get(new ThrownExceptionNode(mc));
        if (ten != null) {
            HashSet s = (HashSet)callee.thrown.clone();
            //s.add(ten);
            addMapping(ten, s);
        }
    }
    
    void matchEdges(Node node, SetOfNodes nodes) {
        for (Iterator i=node.getEdges().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_Field f = (jq_Field)e.getKey();
            SetOfNodes ap_result = nodes.getAccessPathEdges(f, null);
            Object o = e.getValue();
            if (o instanceof HashSet) {
                HashSet s = (HashSet)o;
                for (Iterator j=s.iterator(); j.hasNext(); ) {
                    Node node2 = (Node)j.next();
                    addMapping(node2, ap_result);
                }
            } else {
                Node node2 = (Node)o;
                addMapping(node2, ap_result);
            }
        }
        for (Iterator i=node.getAccessPathEdges().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_Field f = (jq_Field)e.getKey();
            SetOfNodes result = nodes.getAllEdges(f, null);
            HashSet s = new HashSet();
            node.getEdges(f, s);
            result = new SetOfNodes(s, result);
            Node node2 = (Node)e.getValue();
            addMapping(node2, result);
        }
    }

    public static class Path {
        private final SetOfNodes s;
        private final Path next;
        Path(SetOfNodes s, Path n) { this.s = s; this.next = n; }
        SetOfNodes car() { return s; }
        Path cdr() { return next; }
    }
    
    public static class SetOfNodes {
        HashSet set;
        LinkedHashSet contains;
        HashMap ap_cache;
        HashMap all_cache;
        
        HashSet types;
        
        HashSet listeners;
        
        HashSet dirty_fields;
        
        HashSet back_pointers;
        
        SetOfNodes skip;
        boolean onPath;

        static boolean getTypes(HashSet s, HashSet result) {
            boolean change = false;
            if (s != null) {
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    if (n instanceof ConcreteTypeNode) {
                        if (result.add(n.getDeclaredType())) change = true;
                    }
                }
            }
            return change;
        }
        
        SetOfNodes(HashSet set) { this(set, (LinkedHashSet)null); }
        SetOfNodes(LinkedHashSet contains) { this((HashSet)null, contains); }
        SetOfNodes(HashSet set, SetOfNodes next) {
            this.set = set; this.contains = new LinkedHashSet();
            getTypes(set, this.types = new HashSet());
            this.contains.add(next);
            next.addBackPointer(this); this.addTypes(next.types);
        }
        SetOfNodes(HashSet set, LinkedHashSet contains) {
            this.set = set; this.contains = contains;
            getTypes(set, this.types = new HashSet());
            if (contains != null) {
                for (Iterator i=contains.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    son.addBackPointer(this); this.addTypes(son.types);
                }
            }
        }

        public boolean equals(Object o) {
            if (o instanceof SetOfNodes) return equals((SetOfNodes)o);
            return false;
        }
        
        public boolean equals(SetOfNodes that) {
            if (this == that) return true;
            if (this.skip != null) return that.equals(this.skip);
            if (that.skip != null) return that.skip.equals(this);
            return false;
        }
        
        public int hashCode() {
            if (this.skip != null) return this.skip.hashCode();
            return System.identityHashCode(this);
        }
        
        Set getNodeTypes() { return types; }
        
        void addBackPointer(SetOfNodes that) {
            if (back_pointers == null) back_pointers = new HashSet();
            back_pointers.add(that);
        }
        
        private static void getTouchedFields(HashSet nodes, HashSet result) {
            for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                result.addAll(n.getEdgeFields());
                result.addAll(n.getAccessPathEdgeFields());
            }
        }

        void getTouchedFields(HashSet result) {
            if (this.set != null) {
                getTouchedFields(this.set, result);
            }
            if (this.contains != null) {
                for (Iterator i=this.contains.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    son.getTouchedFields(result);
                }
            }
        }
        
        boolean setSkip(SetOfNodes s, HashSet bad_fields) {
            boolean change = false;
            this.ap_cache = null; this.all_cache = null;
            if (this.set != null) {
                for (Iterator i=this.set.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    boolean c = s.set.add(o);
                    if (c) {
                        Node n = (Node)o;
                        bad_fields.addAll(n.getEdgeFields());
                        bad_fields.addAll(n.getAccessPathEdgeFields());
                        change = true;
                    }
                }
            }
            if (this.contains != null) {
                for (Iterator i=this.contains.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    boolean c = s.contains.add(son);
                    if (c) {
                        son.getTouchedFields(bad_fields);
                        change = true;
                    }
                    son.back_pointers.remove(this);
                    son.back_pointers.add(s);
                }
            }
            s.addTypes(this.types);
            this.skip = s;
            return change;
        }

        void backPropagateDirtyFields(HashSet bad_fields) {
            // only need to do ap_cache because it subsumes all_cache
            if (ap_cache == null) return;
            for (Iterator i=bad_fields.iterator(); i.hasNext(); ) {
                if (ap_cache.containsKey(i.next()))
                    i.remove();
            }
            if (this.dirty_fields == null)
                this.dirty_fields = bad_fields;
            else {
                bad_fields.removeAll(this.dirty_fields);
                this.dirty_fields.addAll(bad_fields);
            }
            if (bad_fields.isEmpty()) return;
            if (back_pointers != null) {
                for (Iterator i=back_pointers.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    son.backPropagateDirtyFields((HashSet)bad_fields.clone());
                }
            }
        }
        
        void addTypes(HashSet types) {
            for (Iterator i=types.iterator(); i.hasNext(); ) {
                Object o = i.next();
                jq.assert(o instanceof jq_Reference);
                if (this.types.contains(o)) {
                    i.remove();
                } else {
                    checkTypeListeners((jq_Reference)o);
                    this.types.add(o);
                }
            }
            if (types.isEmpty()) return;
            if (back_pointers != null) {
                for (Iterator i=back_pointers.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    son.addTypes((HashSet)types.clone());
                }
            }
        }
        
        void addTypeListener(CallTargetListener ctl) {
            if (listeners == null)
                listeners = new HashSet();
            listeners.add(ctl);
        }
        
        void checkTypeListeners(jq_Reference new_type) {
            if (listeners == null) return;
            for (Iterator i=listeners.iterator(); i.hasNext(); ) {
                CallTargetListener cs = (CallTargetListener)i.next();
                // check if the new type gives us a new call target.
                cs.addType(new_type);
            }
        }
        
        SetOfNodes getAccessPathEdges(jq_Field f, Path p) {
            if (this.skip != null) return this.skip.getAccessPathEdges(f, p);
            if (this.onPath) {
                SetOfNodes son;
                boolean change = false;
                HashSet bad_fields = new HashSet();
                while (p != null) {
                    son = p.car();
                    if (son != this)
                        if (son.setSkip(this, bad_fields)) change = true;
                    p = p.cdr();
                }
                // change flag is redundant: if change is false, bad_fields will always be empty.
                if (change && !bad_fields.isEmpty()) {
                    this.backPropagateDirtyFields(bad_fields);
                }
                return null;
            }
            java.lang.ref.SoftReference sr;
            SetOfNodes son = null;
            if (ap_cache != null) {
                sr = (java.lang.ref.SoftReference)ap_cache.get(f);
                if (sr != null) {
                    son = (SetOfNodes)sr.get();
                    if (son != null) {
                        if (dirty_fields != null) {
                            if (!dirty_fields.contains(f))
                                return son;
                        }
                    }
                }
            } else {
                ap_cache = new HashMap();
            }
            if (dirty_fields != null) dirty_fields.remove(f);
            HashSet my_result;
            if (set != null) {
                my_result = new HashSet();
                for (Iterator i=set.iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    n.getAccessPathEdges(f, my_result);
                }
            } else {
                my_result = null;
            }
            LinkedHashSet child_results;
            if (contains != null) {
                child_results = new LinkedHashSet();
                int j=0;
                p = new Path(this, p);
                LinkedHashSet my_contains = contains;
                for (Iterator i=my_contains.iterator(); i.hasNext(); ) {
                    SetOfNodes n = (SetOfNodes)i.next(); ++j;
                    int size = my_contains.size();
                    this.onPath = true;
                    SetOfNodes o = n.getAccessPathEdges(f, p);
                    this.onPath = false;
                    if (size != my_contains.size()) {
                        i = my_contains.iterator();
                        for (int k=0; k<j; ++k) i.next();
                    }
                    if (o != null && !o.isEmpty()) child_results.add(o);
                }
                if (child_results.isEmpty()) child_results = null;
            } else {
                child_results = null;
            }
            SetOfNodes son2 = new SetOfNodes(my_result, child_results);
            if (son == null) {
                son = son2;
            } else {
                son.addAll(son2);
            }
            if (ap_cache != null) {
                sr = new java.lang.ref.SoftReference(son);
                ap_cache.put(f, sr);
            }
            return son;
        }

        SetOfNodes getAllEdges(jq_Field f, Path p) {
            if (this.skip != null) return this.skip.getAllEdges(f, p);
            if (this.onPath) {
                SetOfNodes son;
                boolean change = false;
                HashSet bad_fields = new HashSet();
                while (p != null) {
                    son = p.car();
                    if (son != this)
                        if (son.setSkip(this, bad_fields)) change = true;
                    p = p.cdr();
                }
                // change flag is redundant: if change is false, bad_fields will always be empty.
                if (change && !bad_fields.isEmpty()) {
                    this.backPropagateDirtyFields(bad_fields);
                }
                return null;
            }
            java.lang.ref.SoftReference sr;
            SetOfNodes son = null;
            if (all_cache != null) {
                sr = (java.lang.ref.SoftReference)all_cache.get(f);
                if (sr != null) {
                    son = (SetOfNodes)sr.get();
                    if (son != null) {
                        if (dirty_fields != null) {
                            if (!dirty_fields.contains(f))
                                return son;
                        }
                    }
                }
            } else {
                all_cache = new HashMap();
            }
            if (dirty_fields != null) dirty_fields.remove(f);
            HashSet my_result;
            if (set != null) {
                my_result = new HashSet();
                for (Iterator i=set.iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    n.getEdges(f, my_result);
                }
            } else {
                my_result = null;
            }
            LinkedHashSet child_results = new LinkedHashSet();
            child_results.add(this.getAccessPathEdges(f, null));
            if (contains != null) {
                int j=0;
                p = new Path(this, p);
                LinkedHashSet my_contains = contains;
                for (Iterator i=my_contains.iterator(); i.hasNext(); ) {
                    SetOfNodes n = (SetOfNodes)i.next();
                    int size = my_contains.size();
                    this.onPath = true;
                    SetOfNodes o = n.getAllEdges(f, p);
                    this.onPath = false;
                    if (size != my_contains.size()) {
                        i = my_contains.iterator();
                        for (int k=0; k<j; ++k) i.next();
                    }
                    if (o != null && !o.isEmpty()) child_results.add(o);
                }
            }
            SetOfNodes son2 = new SetOfNodes(my_result, child_results);
            if (son == null) {
                son = son2;
            } else {
                son.addAll(son2);
            }
            if (all_cache != null) {
                sr = new java.lang.ref.SoftReference(son);
                all_cache.put(f, sr);
            }
            return son;
        }
        
        boolean isEmpty() {
            if (this.skip != null) return this.skip.isEmpty();
            if (set != null && !set.isEmpty()) return false;
            if (contains != null) {
                for (Iterator i=contains.iterator(); i.hasNext(); ) {
                    if (!((SetOfNodes)i.next()).isEmpty()) return false;
                }
            }
            return true;
        }
        
        boolean add(Node that) {
            if (this.skip != null) return this.skip.add(that);
            boolean b;
            if (this.set != null) {
                b = this.set.add(that);
            } else {
                this.set = new HashSet(); this.set.add(that);
                b = true;
            }
            if (b) {
                if (that instanceof ConcreteTypeNode) {
                    if (!this.types.contains(that.getDeclaredType())) {
                        HashSet s = new HashSet(); s.add(that.getDeclaredType());
                        this.addTypes(s);
                    }
                }
                if (back_pointers != null || ap_cache != null) {
                    HashSet bad_fields = new HashSet();
                    bad_fields.addAll(that.getEdgeFields());
                    bad_fields.addAll(that.getAccessPathEdgeFields());
                    this.backPropagateDirtyFields(bad_fields);
                }
            }
            return b;
        }
        boolean addAll(HashSet that) {
            if (this.skip != null) return this.skip.addAll(that);
            boolean change = false;
            HashSet bad_fields = null;
            HashSet new_types = null;
            if (this.set != null) {
                for (Iterator i=that.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    boolean c = this.set.add(o);
                    if (c) {
                        Node n = (Node)o;
                        change = true;
                        if (bad_fields == null) bad_fields = new HashSet();
                        bad_fields.addAll(n.getEdgeFields());
                        bad_fields.addAll(n.getAccessPathEdgeFields());
                        if (n instanceof ConcreteTypeNode) {
                            if (!this.types.contains(n.getDeclaredType())) {
                                if (new_types == null) new_types = new HashSet();
                                new_types.add(n.getDeclaredType());
                            }
                        }
                    }
                }
            } else {
                this.set = that; change = true;
                getTouchedFields(that, bad_fields = new HashSet());
            }
            if (bad_fields != null && !bad_fields.isEmpty())
                this.backPropagateDirtyFields(bad_fields);
            if (new_types != null)
                this.addTypes(new_types);
            return change;
        }
        boolean addAll(SetOfNodes that) {
            if (this.skip != null) return this.skip.addAll(that);
            while (that.skip != null) that = that.skip;
            boolean change = false;
            HashSet bad_fields = null;
            HashSet new_types = (HashSet)that.types.clone();
            if (that.set != null) {
                if (this.set != null) {
                    for (Iterator i=that.set.iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        boolean c = this.set.add(o);
                        if (c) {
                            Node n = (Node)o;
                            change = true;
                            if (bad_fields == null) bad_fields = new HashSet();
                            bad_fields.addAll(n.getEdgeFields());
                            bad_fields.addAll(n.getAccessPathEdgeFields());
                            if (n instanceof ConcreteTypeNode) {
                                new_types.add(n.getDeclaredType());
                            }
                        }
                    }
                } else {
                    this.set = that.set; change = true;
                    getTouchedFields(that.set, bad_fields = new HashSet());
                }
            }
            if (that.contains != null) {
                if (this.contains != null) {
                    for (Iterator i=that.contains.iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        boolean c = this.contains.add(o);
                        if (c) {
                            SetOfNodes son = (SetOfNodes)o;
                            if (bad_fields == null) bad_fields = new HashSet();
                            son.getTouchedFields(bad_fields);
                            son.back_pointers.add(this);
                            change = true;
                        }
                    }
                } else {
                    this.contains = that.contains; change = true;
                    for (Iterator i=that.contains.iterator(); i.hasNext(); ) {
                        SetOfNodes son = (SetOfNodes)i.next();
                        if (bad_fields == null) bad_fields = new HashSet();
                        son.getTouchedFields(bad_fields);
                        son.back_pointers.add(this);
                    }
                }
            }
            if (bad_fields != null && !bad_fields.isEmpty())
                this.backPropagateDirtyFields(bad_fields);
            this.addTypes(new_types);
            return change;
        }
        static SetOfNodes union(SetOfNodes dis, SetOfNodes dat) {
            while (dis.skip != null) dis = dis.skip;
            while (dat.skip != null) dat = dat.skip;
            LinkedHashSet list = new LinkedHashSet();
            list.add(dis);
            list.add(dat);
            return new SetOfNodes(list);
        }
        
        public Iterator iterator() { return new Itr(this); }
        
        public String toString() {
            Itr i = new Itr(this);
            StringBuffer sb = new StringBuffer();
            sb.append('{');
            if (i.hasNext()) {
                sb.append(i.next());
                while (i.hasNext()) {
                    sb.append(',');
                    sb.append(i.next());
                }
            }
            sb.append('}');
            return sb.toString();
        }
        
        static class Itr implements Iterator {
            LinkedList stack;   // stack of Iterators
            Iterator it;        // current Iterator
            Itr(SetOfNodes sn) {
                while (sn.skip != null) sn = sn.skip;
                this.stack = new LinkedList();
                if (sn.contains != null && !sn.contains.isEmpty()) {
                    this.stack.addLast(sn.contains.iterator());
                }
                if (sn.set != null && !sn.set.isEmpty()) {
                    this.it = sn.set.iterator();
                } else {
                    nextIterator();
                }
            }
            private void nextIterator() {
                for (;;) {
                    if (stack.isEmpty()) {
                        this.it = Collections.EMPTY_SET.iterator();
                        return;
                    }
                    Iterator i = (Iterator)stack.getLast();
                    if (!i.hasNext()) {
                        stack.removeLast();
                        continue;
                    }
                    SetOfNodes sn = (SetOfNodes)i.next();
                    while (sn.skip != null) sn = sn.skip;
                    if (sn.set != null && !sn.set.isEmpty()) {
                        this.it = sn.set.iterator();
                        if (sn.contains != null && !sn.contains.isEmpty()) {
                            stack.addLast(i = sn.contains.iterator());
                        }
                        return;
                    }
                    if (sn.contains != null && !sn.contains.isEmpty()) {
                        stack.addLast(i = sn.contains.iterator());
                    }
                }
            }
            public boolean hasNext() {
                return it.hasNext();
            }
            public Object next() {
                Object o = it.next();
                if (!it.hasNext()) nextIterator();
                return o;
            }
            
            public void remove() {
                throw new UnsupportedOperationException();
            }
            
        }
        
    }
}

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
import MethodSummary.CallSite;
import MethodSummary.Node;
import MethodSummary.ConcreteTypeNode;
import MethodSummary.OutsideNode;
import MethodSummary.GlobalNode;
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
    public static final boolean TRACE_SETS = true;

    public static final class AndersenVisitor implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            INSTANCE.visitMethod(cfg);
            INSTANCE.doWorklist();
            System.out.println("Result after analyzing "+cfg.getMethod()+":");
            System.out.println(INSTANCE.dumpResults());
        }
    }
    
    /** Maps nodes to their set of corresponding nodes. */
    HashMap nodesToCorrespondingNodes;
    
    /** Worklist of operations to perform. */
    LinkedHashSet worklist;
    
    HashSet visitedMethods;
    
    HashMap callSitesToTargets;
    
    /** Creates new AndersenPointerAnalysis */
    public AndersenPointerAnalysis() {
        nodesToCorrespondingNodes = new HashMap();
        worklist = new LinkedHashSet();
        visitedMethods = new HashSet();
        callSitesToTargets = new HashMap();
    }

    public static AndersenPointerAnalysis INSTANCE = new AndersenPointerAnalysis();
    
    public static final String lineSep = System.getProperty("line.separator");
    
    public String dumpResults() {
        StringBuffer sb = new StringBuffer();
        for (Iterator i=callSitesToTargets.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            CallSite cs = (CallSite)e.getKey();
            Set s = (Set)e.getValue();
            sb.append(cs.toString());
            sb.append(": ");
            sb.append(s.toString());
            sb.append(lineSep);
        }
        return sb.toString();
    }
    
    void visitMethod(ControlFlowGraph cfg) {
        if (visitedMethods.contains(cfg)) return;
        if (TRACE) out.println("Visiting method: "+cfg.getMethod());
        visitedMethods.add(cfg);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            MethodCall mc = (MethodCall)i.next();
            CallSite cs = new CallSite(ms, mc);
            if (TRACE) out.println("Found call: "+cs);
            CallTargets ct = mc.getCallTargets();
            if (TRACE) out.println("Possible targets ignoring type information: "+ct);
            HashSet definite_targets = new HashSet();
            jq.assert(!callSitesToTargets.containsKey(cs));
            callSitesToTargets.put(cs, definite_targets);
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
                        CallTargetListener ctl = new CallTargetListener(cs, definite_targets);
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
        CallSite cs; Set currentResult;
        CallTargetListener(CallSite cs, Set currentResult) {
            this.cs = cs; this.currentResult = currentResult;
            jq.assert(INSTANCE.callSitesToTargets.get(cs) == currentResult);
        }
        CallTargetListener(MethodSummary ms, MethodCall mc, Set currentResult) {
            this.cs = new CallSite(ms, mc); this.currentResult = currentResult;
            jq.assert(INSTANCE.callSitesToTargets.get(cs) == currentResult);
        }
        void addType(jq_Reference type) {
            if (TRACE) out.println("Checking if type "+type+" adds a new target for "+cs);
            MethodSummary ms = cs.caller;
            MethodCall mc = cs.m;
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
        if (TRACE) out.println("Adding mapping from "+n+" to set of nodes "+s.toString_addr());
        if (son != null) {
            if (TRACE) out.println("Mapping for "+n+" already exists: "+son.toString_addr()+", merging");
            change = son.addAll(s);
        } else {
            nodesToCorrespondingNodes.put(n, s);
            change = true;
        }
        if (TRACE && change) out.println("Mapping for "+n+" changed, adding to worklist");
        if (change) addToWorklist(n);
        return change;
    }
    
    void addParameterAndReturnMappings(MethodSummary caller, MethodCall mc, MethodSummary callee) {
        if (TRACE) out.println("Adding parameter and return mappings for "+mc+" from "+jq.hex(System.identityHashCode(caller))+" to "+jq.hex(System.identityHashCode(callee)));
        ParamListOperand plo = Invoke.getParamList(mc.q);
        for (int i=0; i<plo.length(); ++i) {
            jq_Type t = plo.get(i).getType();
            if (!(t instanceof jq_Reference)) continue;
            ParamNode pn = callee.getParamNode(i);
            PassedParameter pp = new PassedParameter(mc, i);
            HashSet s = new HashSet();
            caller.getNodesThatCall(pp, s);
            //s.add(pn);
            if (TRACE) out.println("Adding parameter mapping "+pn+" to set "+s);
            addMapping(pn, s);
        }
        ReturnValueNode rvn = (ReturnValueNode)caller.nodes.get(new ReturnValueNode(mc));
        if (rvn != null) {
            HashSet s = (HashSet)callee.returned.clone();
            //s.add(rvn);
            if (TRACE) out.println("Adding return mapping "+rvn+" to set "+s);
            addMapping(rvn, s);
        }
        ThrownExceptionNode ten = (ThrownExceptionNode)caller.nodes.get(new ThrownExceptionNode(mc));
        if (ten != null) {
            HashSet s = (HashSet)callee.thrown.clone();
            //s.add(ten);
            if (TRACE) out.println("Adding thrown mapping "+ten+" to set "+s);
            addMapping(ten, s);
        }
    }
    
    void matchEdges(Node node, SetOfNodes nodes) {
        for (Iterator i=node.getEdges().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_Field f = (jq_Field)e.getKey();
            SetOfNodes ap_result = nodes.getAccessPathEdges(f, null);
            Object o = e.getValue();
            if (TRACE) out.println("Node "+node+" inside edge to field "+f+": "+o+" matches outside edges "+ap_result);
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
            Object o = e.getValue();
            if (TRACE) out.println("Node "+node+" outside edge to field "+f+": "+o+" matches edges "+result);
            if (o instanceof HashSet) {
                HashSet set = (HashSet)o;
                if (TRACE) out.println("Adding nodes to "+result.toString_addr()+": "+set);
                s.addAll(set);
                for (Iterator j=set.iterator(); j.hasNext(); ) {
                    Node node2 = (Node)j.next();
                    addMapping(node2, result);
                }
            } else {
                Node node2 = (Node)o;
                if (TRACE) out.println("Adding node to "+result.toString_addr()+": "+node2);
                s.add(node2);
                addMapping(node2, result);
            }
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

        private String toString_addr() { return jq.hex(System.identityHashCode(this)); }
            
        private static String dumpSetContents(Set s) {
            if (s == null) return "{}";
            StringBuffer sb = new StringBuffer();
            sb.append('{');
            Iterator i=s.iterator();
            if (i.hasNext()) {
                sb.append(jq.hex(System.identityHashCode(i.next())));
                while (i.hasNext()) {
                    sb.append(',');
                    sb.append(jq.hex(System.identityHashCode(i.next())));
                }
            }
            sb.append('}');
            return sb.toString();
        }

        private String toString_full() {
            StringBuffer sb = new StringBuffer();
            sb.append("Set ");
            sb.append(this.toString_addr());
            if (skip != null) {
                sb.append(" skip=");
                sb.append(this.skip.toString_addr());
                return sb.toString();
            }
            sb.append(" nodes ");
            sb.append(dumpSetContents(set));
            sb.append(" contains ");
            sb.append(dumpSetContents(contains));
            sb.append(" types ");
            sb.append(types.toString());
            if (listeners != null) {
                sb.append(" listeners ");
                sb.append(listeners.toString());
            }
            if (dirty_fields != null) {
                sb.append(" dirty_fields ");
                sb.append(dirty_fields.toString());
            }
            if (back_pointers != null) {
                sb.append(" back_pointers ");
                sb.append(back_pointers.toString());
            }
            return sb.toString();
        }
        
        SetOfNodes(HashSet set) { this(set, (LinkedHashSet)null); }
        SetOfNodes(LinkedHashSet contains) { this((HashSet)null, contains); }
        SetOfNodes(HashSet set, SetOfNodes next) {
            this.set = set; this.contains = new LinkedHashSet();
            getTypes(set, this.types = new HashSet());
            this.contains.add(next);
            next.addBackPointer(this); this.addTypes(next.types);
            if (TRACE_SETS) out.println("Constructed "+this.toString_full());
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
            if (TRACE_SETS) out.println("Constructed "+this.toString_full());
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
            if (TRACE_SETS) out.println("Adding back pointer from "+this.toString_addr()+" to "+that.toString_addr());
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

        void _getTouchedFields(HashSet visited, HashSet result) {
            if (this.skip != null) { this.skip._getTouchedFields(visited, result); }
            if (visited.contains(this)) {
                if (TRACE_SETS) out.println("already visited set when getting touched fields! (cycle?) "+this.toString_addr());
                return;
            }
            visited.add(this);
            if (this.set != null) {
                getTouchedFields(this.set, result);
            }
            if (TRACE_SETS) out.println(this.toString_addr()+": getting the touched fields, current is "+result);
            if (this.contains != null) {
                for (Iterator i=this.contains.iterator(); i.hasNext(); ) {
                    SetOfNodes son = (SetOfNodes)i.next();
                    son._getTouchedFields(visited, result);
                }
            }
        }
        void getTouchedFields(HashSet result) {
            _getTouchedFields(new HashSet(), result);
        }
        
        boolean setSkip(SetOfNodes s, HashSet bad_fields) {
            if (TRACE_SETS) out.println(this.toString_addr()+": setting the skip to "+s.toString_addr());
            boolean change = false;
            this.ap_cache = null; this.all_cache = null;
            if (this.set != null) {
                for (Iterator i=this.set.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    boolean c = s.set.add(o);
                    if (c) {
                        Node n = (Node)o;
                        if (TRACE_SETS) out.println("added new node "+n+" to the local set of "+s.toString_addr()+", marking the fields as bad: "+n.getEdgeFields()+", "+n.getAccessPathEdgeFields());
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
                        if (TRACE_SETS) out.println("added new set "+son.toString_addr()+" to the contains set of "+s.toString_addr()+", marking the fields as bad");
                        son.getTouchedFields(bad_fields);
                        change = true;
                    }
                    if (TRACE_SETS) out.println(son.toString_addr()+": replacing back pointer "+this.toString_addr()+" with "+s.toString_addr());
                    son.back_pointers.remove(this);
                    son.back_pointers.add(s);
                }
            }
            if (TRACE_SETS) out.println("adding types of "+this.toString_addr()+"="+this.types+" to "+s.toString_addr());
            s.addTypes(this.types);
            this.skip = s;
            return change;
        }

        void backPropagateDirtyFields(HashSet bad_fields) {
            if (this.skip != null) { this.skip.backPropagateDirtyFields(bad_fields); return; }
            if (TRACE_SETS) out.println("back propagating dirty fields "+bad_fields+" to "+this.toString_addr());
            if (ap_cache == null) return;
            for (Iterator i=bad_fields.iterator(); i.hasNext(); ) {
                Object o = i.next();
                if (ap_cache.containsKey(o)) {
                    if (TRACE_SETS) out.println(this.toString_addr()+": removing entry for field "+o+" from ap_cache");
                    i.remove();
                }
                if (all_cache != null && all_cache.containsKey(o)) {
                    if (TRACE_SETS) out.println(this.toString_addr()+": removing entry for field "+o+" from all_cache");
                    all_cache.remove(o);
                }
            }
            if (this.dirty_fields == null) {
                this.dirty_fields = bad_fields;
            } else {
                bad_fields.removeAll(this.dirty_fields);
                if (TRACE_SETS) out.println(this.toString_addr()+": new dirty fields: "+bad_fields);
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
            if (this.skip != null) { this.skip.addTypes(types); return; }
            if (TRACE_SETS) out.println(this.toString_addr()+": adding types "+types);
            for (Iterator i=types.iterator(); i.hasNext(); ) {
                Object o = i.next();
                jq_Reference r = (jq_Reference)o;
                if (this.types.contains(o)) {
                    if (TRACE_SETS) out.println(this.toString_addr()+": already contains type "+r);
                    i.remove();
                } else {
                    if (TRACE_SETS) out.println(this.toString_addr()+": new type "+r);
                    checkTypeListeners(r);
                    this.types.add(r);
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
            if (this.skip != null) { this.skip.addTypeListener(ctl); return; }
            if (listeners == null)
                listeners = new HashSet();
            if (TRACE_SETS) out.println(this.toString_addr()+": adding listener "+ctl);
            listeners.add(ctl);
        }
        
        void checkTypeListeners(jq_Reference new_type) {
            if (this.skip != null) { this.skip.checkTypeListeners(new_type); return; }
            if (listeners == null) return;
            if (TRACE_SETS) out.println(this.toString_addr()+": checking for listener for new class "+new_type);
            for (Iterator i=listeners.iterator(); i.hasNext(); ) {
                CallTargetListener cs = (CallTargetListener)i.next();
                // check if the new type gives us a new call target.
                cs.addType(new_type);
            }
        }
        
        SetOfNodes getAccessPathEdges(jq_Field f, Path p) {
            if (this.skip != null) return this.skip.getAccessPathEdges(f, p);
            if (TRACE_SETS) out.println(this.toString_addr()+": getting ap edges "+f);
            if (this.onPath) {
                if (TRACE_SETS) out.println(this.toString_addr()+": cycle detected! path="+p);
                SetOfNodes son;
                boolean change = false;
                HashSet bad_fields = new HashSet();
                while (p != null) {
                    son = p.car();
                    if (TRACE_SETS) out.println("next in path: "+son.toString_addr());
                    if (son != this) {
                        if (son.setSkip(this, bad_fields)) {
                            if (TRACE_SETS) out.println("change when setting skip on "+son.toString_addr()+", bad fields="+bad_fields);
                            change = true;
                        }
                    }
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
                        if (TRACE_SETS) out.println(this.toString_addr()+": ap_cache contains entry for "+f+": "+son.toString_addr());
                        if (dirty_fields != null) {
                            if (!dirty_fields.contains(f)) {
                                return son;
                            } else {
                                if (TRACE_SETS) out.println(this.toString_addr()+": field "+f+" is dirty! recalculating.");
                            }
                        }
                    }
                }
            } else {
                if (TRACE_SETS) out.println(this.toString_addr()+": allocating ap_cache");
                ap_cache = new HashMap();
            }
            if (dirty_fields != null) {
                if (TRACE_SETS) out.println(this.toString_addr()+": removing field "+f+" from the dirty set");
                dirty_fields.remove(f);
            }
            HashSet my_result;
            if (set != null) {
                my_result = new HashSet();
                if (TRACE_SETS) out.println(this.toString_addr()+": getting the ap edges on field "+f+" of local nodes "+set);
                for (Iterator i=set.iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    n.getAccessPathEdges(f, my_result);
                }
                if (TRACE_SETS) out.println(this.toString_addr()+": result on local nodes: "+my_result);
            } else {
                my_result = null;
            }
            LinkedHashSet child_results;
            if (contains != null) {
                child_results = new LinkedHashSet();
                int j=0;
                p = new Path(this, p);
                LinkedHashSet my_contains = contains;
                if (TRACE_SETS) out.println(this.toString_addr()+": getting the ap edges on field "+f+" of children "+dumpSetContents(my_contains));
                for (Iterator i=my_contains.iterator(); i.hasNext(); ) {
                    SetOfNodes n = (SetOfNodes)i.next(); ++j;
                    int size = my_contains.size();
                    this.onPath = true;
                    SetOfNodes o = n.getAccessPathEdges(f, p);
                    this.onPath = false;
                    if (size != my_contains.size()) {
                        if (TRACE_SETS) out.println(this.toString_addr()+": size of contains set changed ("+size+" to "+my_contains.size()+"), redoing iterator");
                        i = my_contains.iterator();
                        for (int k=0; k<j; ++k) i.next();
                    }
                    if (o != null && !o.isEmpty()) {
                        if (TRACE_SETS) out.println(this.toString_addr()+": adding "+o.toString_addr()+" to child results");
                        child_results.add(o);
                    }
                }
                if (child_results.isEmpty()) child_results = null;
            } else {
                child_results = null;
            }
            SetOfNodes son2 = new SetOfNodes(my_result, child_results);
            if (son == null) {
                son = son2;
            } else {
                if (TRACE_SETS) out.println("adding new results to existing results "+son.toString_addr());
                son.addAll(son2);
            }
            if (ap_cache != null) {
                sr = new java.lang.ref.SoftReference(son);
                ap_cache.put(f, sr);
                if (TRACE_SETS) out.println("putting results "+son.toString_addr()+" into ap cache for field "+f);
            }
            return son;
        }

        SetOfNodes getAllEdges(jq_Field f, Path p) {
            if (this.skip != null) return this.skip.getAllEdges(f, p);
            if (TRACE_SETS) out.println(this.toString_addr()+": getting all edges "+f);
            if (this.onPath) {
                if (TRACE_SETS) out.println(this.toString_addr()+": cycle detected! path="+p);
                SetOfNodes son;
                boolean change = false;
                HashSet bad_fields = new HashSet();
                while (p != null) {
                    son = p.car();
                    if (TRACE_SETS) out.println("next in path: "+son.toString_addr());
                    if (son != this) {
                        if (son.setSkip(this, bad_fields)) {
                            if (TRACE_SETS) out.println("change when setting skip on "+son.toString_addr()+", bad fields="+bad_fields);
                            change = true;
                        }
                    }
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
                        if (TRACE_SETS) out.println(this.toString_addr()+": all_cache contains entry for "+f+": "+son.toString_addr());
                        if (dirty_fields != null) {
                            if (!dirty_fields.contains(f)) {
                                return son;
                            } else {
                                if (TRACE_SETS) out.println(this.toString_addr()+": field "+f+" is dirty! recalculating.");
                            }
                        }
                    }
                }
            } else {
                if (TRACE_SETS) out.println(this.toString_addr()+": allocating all_cache");
                all_cache = new HashMap();
            }
            if (dirty_fields != null) {
                if (TRACE_SETS) out.println(this.toString_addr()+": removing field "+f+" from the dirty set");
                dirty_fields.remove(f);
            }
            HashSet my_result;
            if (set != null) {
                my_result = new HashSet();
                if (TRACE_SETS) out.println(this.toString_addr()+": getting write edges on field "+f+" of local nodes "+set);
                for (Iterator i=set.iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    n.getEdges(f, my_result);
                }
                if (TRACE_SETS) out.println(this.toString_addr()+": result on local nodes: "+my_result);
            } else {
                my_result = null;
            }
            LinkedHashSet child_results = new LinkedHashSet();
            child_results.add(this.getAccessPathEdges(f, null));
            if (contains != null) {
                int j=0;
                p = new Path(this, p);
                LinkedHashSet my_contains = contains;
                if (TRACE_SETS) out.println(this.toString_addr()+": getting all edges on field "+f+" of children "+dumpSetContents(my_contains));
                for (Iterator i=my_contains.iterator(); i.hasNext(); ) {
                    SetOfNodes n = (SetOfNodes)i.next();
                    int size = my_contains.size();
                    this.onPath = true;
                    SetOfNodes o = n.getAllEdges(f, p);
                    this.onPath = false;
                    if (size != my_contains.size()) {
                        if (TRACE_SETS) out.println(this.toString_addr()+": size of contains set changed ("+size+" to "+my_contains.size()+"), redoing iterator");
                        i = my_contains.iterator();
                        for (int k=0; k<j; ++k) i.next();
                    }
                    if (o != null && !o.isEmpty()) {
                        if (TRACE_SETS) out.println(this.toString_addr()+": adding "+o.toString_addr()+" to child results");
                        child_results.add(o);
                    }
                }
            }
            SetOfNodes son2 = new SetOfNodes(my_result, child_results);
            if (son == null) {
                son = son2;
            } else {
                if (TRACE_SETS) out.println("adding new results to existing results "+son.toString_addr());
                son.addAll(son2);
            }
            if (all_cache != null) {
                sr = new java.lang.ref.SoftReference(son);
                all_cache.put(f, sr);
                if (TRACE_SETS) out.println("putting results "+son.toString_addr()+" into all cache for field "+f);
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
            if (TRACE_SETS) out.println("adding node "+that+" to set "+this.toString_addr());
            if (this.set != null) {
                if (TRACE_SETS) out.println(this.toString_addr()+": local set already exists, adding to it");
                b = this.set.add(that);
            } else {
                if (TRACE_SETS) out.println(this.toString_addr()+": allocating new local set");
                this.set = new HashSet(); this.set.add(that);
                b = true;
            }
            if (b) {
                if (TRACE_SETS) out.println(this.toString_addr()+": change occurred from adding node "+that);
                if (that instanceof ConcreteTypeNode) {
                    if (!this.types.contains(that.getDeclaredType())) {
                        if (TRACE_SETS) out.println(this.toString_addr()+": new node has a new type: "+that.getDeclaredType());
                        HashSet s = new HashSet(); s.add(that.getDeclaredType());
                        this.addTypes(s);
                    }
                }
                if (back_pointers != null || ap_cache != null) {
                    HashSet bad_fields = new HashSet();
                    bad_fields.addAll(that.getEdgeFields());
                    bad_fields.addAll(that.getAccessPathEdgeFields());
                    if (TRACE_SETS) out.println(this.toString_addr()+": propagating bad fields from new node: "+bad_fields);
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
            if (TRACE_SETS) out.println("adding nodes "+that+" to set "+this.toString_addr());
            if (this.set != null) {
                for (Iterator i=that.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    boolean c = this.set.add(o);
                    if (c) {
                        Node n = (Node)o;
                        if (TRACE_SETS) out.println(this.toString_addr()+": change occurred from adding node "+n);
                        change = true;
                        if (bad_fields == null) bad_fields = new HashSet();
                        bad_fields.addAll(n.getEdgeFields());
                        bad_fields.addAll(n.getAccessPathEdgeFields());
                        if (TRACE_SETS) out.println(this.toString_addr()+": bad fields: "+bad_fields);
                        if (n instanceof ConcreteTypeNode) {
                            if (!this.types.contains(n.getDeclaredType())) {
                                if (TRACE_SETS) out.println(this.toString_addr()+": new node has a new type: "+n.getDeclaredType());
                                if (new_types == null) new_types = new HashSet();
                                new_types.add(n.getDeclaredType());
                            }
                        }
                    }
                }
            } else {
                this.set = that; change = true;
                getTouchedFields(that, bad_fields = new HashSet());
                if (TRACE_SETS) out.println(this.toString_addr()+": using new local set: "+that+", touched fields: "+bad_fields);
            }
            if (bad_fields != null && !bad_fields.isEmpty()) {
                if (TRACE_SETS) out.println(this.toString_addr()+": propagating bad fields from new nodes: "+bad_fields);
                this.backPropagateDirtyFields(bad_fields);
            }
            if (new_types != null) {
                if (TRACE_SETS) out.println(this.toString_addr()+": propagating new types from new nodes: "+new_types);
                this.addTypes(new_types);
            }
            return change;
        }
        boolean addAll(SetOfNodes that) {
            if (this.skip != null) return this.skip.addAll(that);
            while (that.skip != null) that = that.skip;
            boolean change = false;
            HashSet bad_fields = null;
            HashSet new_types = (HashSet)that.types.clone();
            if (TRACE_SETS) out.println("adding contents of set "+that.toString_addr()+" to set "+this.toString_addr());
            if (that.set != null) {
                if (TRACE_SETS) out.println("adding local nodes "+that.set+" to set "+this.toString_addr());
                if (this.set != null) {
                    for (Iterator i=that.set.iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        boolean c = this.set.add(o);
                        if (c) {
                            Node n = (Node)o;
                            if (TRACE_SETS) out.println(this.toString_addr()+": change occurred from adding node "+n);
                            change = true;
                            if (bad_fields == null) bad_fields = new HashSet();
                            bad_fields.addAll(n.getEdgeFields());
                            bad_fields.addAll(n.getAccessPathEdgeFields());
                            if (TRACE_SETS) out.println(this.toString_addr()+": bad fields: "+bad_fields);
                            if (n instanceof ConcreteTypeNode) {
                                //new_types.add(n.getDeclaredType());
                                jq.assert(new_types.contains(n.getDeclaredType()));
                            }
                        }
                    }
                } else {
                    this.set = that.set; change = true;
                    getTouchedFields(that.set, bad_fields = new HashSet());
                    if (TRACE_SETS) out.println(this.toString_addr()+": using new local set: "+that.set+", touched fields: "+bad_fields);
                }
            }
            if (that.contains != null) {
                if (TRACE_SETS) out.println("adding contains set of set "+that.toString_addr()+" to set "+this.toString_addr());
                if (this.contains != null) {
                    for (Iterator i=that.contains.iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        boolean c = this.contains.add(o);
                        if (c) {
                            SetOfNodes son = (SetOfNodes)o;
                            if (TRACE_SETS) out.println(this.toString_addr()+": change occurred from adding contains set "+son.toString_addr());
                            if (bad_fields == null) bad_fields = new HashSet();
                            son.getTouchedFields(bad_fields);
                            if (TRACE_SETS) out.println(this.toString_addr()+": change occurred from adding contains set "+son.toString_addr());
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
                        if (TRACE_SETS) out.println(this.toString_addr()+": bad fields: "+bad_fields);
                        son.back_pointers.add(this);
                        if (TRACE_SETS) out.println("adding a back pointer from "+son.toString_addr()+" to "+this.toString_addr());
                    }
                }
            }
            if (bad_fields != null && !bad_fields.isEmpty()) {
                if (TRACE_SETS) out.println(this.toString_addr()+": propagating bad fields from new set: "+bad_fields);
                this.backPropagateDirtyFields(bad_fields);
            }
            if (TRACE_SETS) out.println(this.toString_addr()+": propagating types from new set: "+new_types);
            this.addTypes(new_types);
            return change;
        }
        static SetOfNodes union(SetOfNodes dis, SetOfNodes dat) {
            while (dis.skip != null) dis = dis.skip;
            while (dat.skip != null) dat = dat.skip;
            LinkedHashSet list = new LinkedHashSet();
            list.add(dis);
            list.add(dat);
            if (TRACE_SETS) out.println("making a new set, which is the union of "+dis.toString_addr()+" and "+dat.toString_addr());
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
            HashSet visited;    // visited sets
            LinkedList stack;   // stack of Iterators
            Iterator it;        // current Iterator
            Itr(SetOfNodes sn) {
                while (sn.skip != null) sn = sn.skip;
                this.visited = new HashSet();
                this.stack = new LinkedList();
                if (sn.contains != null && !sn.contains.isEmpty()) {
                    this.stack.addLast(sn.contains.iterator());
                }
                this.visited.add(sn);
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
                    if (visited.contains(sn)) continue;
                    visited.add(sn);
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

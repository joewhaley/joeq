/*
 * ContextSensitiveCallGraph.java
 *
 * Created on March 7, 2002, 3:03 PM
 */

package Compil3r.Quad;
import Bootstrap.PrimordialClassLoader;
import Util.Templates.List;
import Util.Templates.ListIterator;
import Util.AppendIterator;
import Util.SingletonIterator;
import Util.NullIterator;
import Util.FilterIterator;
import Util.IdentityHashCodeWrapper;
import Clazz.*;
import java.util.HashSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Iterator;
import RegisterFactory.Register;
import Operand.AConstOperand;
import Operand.ParamListOperand;
import Operand.RegisterOperand;
import Operator.ALoad;
import Operator.AStore;
import Operator.CheckCast;
import Operator.Getfield;
import Operator.Getstatic;
import Operator.Invoke;
import Operator.Move;
import Operator.New;
import Operator.NewArray;
import Operator.Putfield;
import Operator.Putstatic;
import Operator.Return;
import Operator.Special;
import Operator.Unary;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class ContextSensitiveCallGraph {

    public static java.io.PrintStream out = System.out;
    public static final boolean TRACE_INTRA = false;
    public static final boolean TRACE_INTER = true;

    public static final class MethodSummaryBuilder implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            AnalysisSummary s = getSummary(cfg);
        }
    }
    
    public static final class PrintCallTargets implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            AnalysisSummary s = getSummary(cfg);
            QuadIterator qi = new QuadIterator(cfg);
            MethodCall mc = new MethodCall(cfg.getMethod(), null);
            CallingContext cc = new CallingContext(mc, s, TopCallingContext.TOP);
            if (TRACE_INTER) out.println("Created first calling context: "+cc);
            while (qi.hasNext()) {
                Quad q = qi.nextQuad();
                if (q.getOperator() instanceof Invoke) {
                    HashSet targets = new HashSet();
                    System.out.println("Targets of "+q);
                    CallTargetsQuery ctq = new CallTargetsQuery(cc, q);
                    QuerySolver.GLOBAL.doQuery(ctq);
                    System.out.println(ctq.getResult(null));
                }
            }
        }
    }
    
    public static HashMap summary_cache = new HashMap();
    public static AnalysisSummary getSummary(ControlFlowGraph cfg) {
        AnalysisSummary s = (AnalysisSummary)summary_cache.get(cfg);
        if (s == null) {
            if (TRACE_INTER) out.println("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv");
            if (TRACE_INTER) out.println("Building summary for "+cfg.getMethod());
            BuildMethodSummary b = new BuildMethodSummary(cfg);
            s = b.getSummary();
            summary_cache.put(cfg, s);
            if (TRACE_INTER) out.println("Summary for "+cfg.getMethod()+":");
            if (TRACE_INTER) out.println(s);
            if (TRACE_INTER) out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
        }
        return s;
    }

    public static class ReturnValue {
        Quad q;
        ReturnValue(Quad q) { this.q = q; }
        public int hashCode() { return q.hashCode(); }
        public boolean equals(ReturnValue that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof ReturnValue) return equals((ReturnValue)o);
            return false;
        }
        public String toString() { return "Return value for quad "+q.getID(); }
    }

    public static class ThrownValue {
        Quad q;
        ThrownValue(Quad q) { this.q = q; }
        public int hashCode() { return q.hashCode(); }
        public boolean equals(ThrownValue that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof ThrownValue) return equals((ThrownValue)o);
            return false;
        }
        public String toString() { return "Thrown value for quad "+q.getID(); }
    }
        
    /*
    public static class TopCallingContext extends CallingContext {
        public static final TopCallingContext TOP = new TopCallingContext();
        
        public String toString() { return "TOP:"; }
        
        TopCallingContext() {
            super(null, null);
        }
        public void getNodes(PassedParameter pp, HashSet result) {
            out.println(this+"Trying to get nodes of "+pp+" from top calling context");
            jq_Reference r = (jq_Reference)pp.m.m.getParamTypes()[pp.paramNum];
            result.add(UnknownTypeNode.get(r));
        }
        static void helper(HashSet memo, jq_Class k) {
            k.load();
            jq_Class[] cl = k.getSubClasses();
            for (int i=0; i<cl.length; ++i) {
                if (memo.contains(cl[i])) continue;
                memo.add(cl[i]);
                helper(memo, cl[i]);
            }
        }
        
        public void getTypesOfParameter(PassedParameter pp, HashSet result, QueryResult qwery) {
            if (TRACE_INTER) out.println(this+"Getting possible types for param "+pp+" from top calling context");
            jq_Method m = pp.m.m;
            int n = pp.paramNum;
            jq_Reference r = (jq_Reference)m.getParamTypes()[n];
            result.add(r);
            if (r instanceof jq_Class) {
                helper(result, (jq_Class)r);
            }
        }
    }
     */
    
    public static class SpecificMethodTarget {
        final MethodCall mc;
        final jq_Method m;
        SpecificMethodTarget(MethodCall mc, jq_Method m) {
            this.mc = mc; this.m = m;
        }
        public boolean equals(SpecificMethodTarget that) {
            return this.m == that.m && this.mc.equals(that.mc);
        }
        public boolean equals(Object o) {
            if (o instanceof SpecificMethodTarget) return equals((SpecificMethodTarget)o);
            return false;
        }
        public int hashCode() { return mc.hashCode() ^ m.hashCode(); }
        public String toString() { return mc+" actual target: "+m; }
    }
    
    public static class NodeAndAccessPath {
        final Node n;
        final AccessPath ap;
        NodeAndAccessPath(Node n, AccessPath ap) {
            this.n = n; this.ap = ap;
        }
        public boolean equals(NodeAndAccessPath that) {
            return this.n == that.n && this.ap.equals(that.ap);
        }
        public boolean equals(Object o) {
            if (o instanceof NodeAndAccessPath) return equals((NodeAndAccessPath)o);
            return false;
        }
        public int hashCode() { return n.hashCode() ^ ap.hashCode(); }
        public String toString() { return n+" access path "+ap; }
    }
    
    /*
    public static class CallingContext {
        final MethodCall call_site;
        final AnalysisSummary s;
        final CallingContext parent;
        final HashMap children;

        public CallingContext getParentContext() { return this.parent; }
        
        public CallingContext getChildCallingContext(MethodCall mc, jq_Method m, AnalysisSummary s) {
            SpecificMethodTarget smt = new SpecificMethodTarget(mc, m);
            CallingContext context2 = (CallingContext)children.get(smt);
            if (context2 == null) {
                children.put(smt, context2 = new CallingContext(mc, s, this));
                if (TRACE_INTER) out.println("Created new calling context: "+context2);
            }
            return context2;
        }
        
        public String toString() {
            int depth = 0;
            CallingContext p = parent;
            while (p != TopCallingContext.TOP) { ++depth; p = p.parent; }
            return depth+" "+call_site.m.getName()+"():";
        }
        
        protected CallingContext(MethodCall call_site, AnalysisSummary s) {
            this.call_site = call_site; this.s = s; this.parent = this;
            this.children = new HashMap();
        }
        
        CallingContext(MethodCall call_site, AnalysisSummary s, CallingContext parent) {
            this.call_site = call_site; this.s = s; this.parent = parent;
            this.children = new HashMap();
        }
        
        public void getNodes(PassedParameter pp, HashSet result) {
            for (Iterator i = s.nodeIterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                if ((n.passedParameters != null) && n.passedParameters.contains(pp))
                    result.add(n);
            }
        }
        
        HashMap local_memoizer = new HashMap();
        
        // Get the set of types that can be passed as the given parameter.
        public void getTypesOfParameter(PassedParameter pp, HashSet result, QueryResult qwery) {
            HashSet memo = (HashSet)local_memoizer.get(pp);
            if (memo != null) {
                if (TRACE_INTER) out.println(this+"Memoizer succeeded for "+pp+":"+memo);
                result.addAll(memo); return;
            }
            //if (TRACE_INTER) out.println(this+"Memoizer contains: "+local_memoizer);
            if (TRACE_INTER) out.println(this+"Getting types for "+pp);
            local_memoizer.put(pp, memo = new HashSet());
            for (Iterator i = s.nodeIterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                if (n.passedParameters != null && n.passedParameters.contains(pp)) {
                    getTypesOfNode(n, memo, qwery);
                }
            }
            result.addAll(memo);
            //if (TRACE_INTER) out.println(this+"Memoizer now contains "+local_memoizer);
        }
        
        // Get the set of types that the given node can be.
        public void getTypesOfNode(Node n, HashSet result, QueryResult qwery) {
            if (TRACE_INTER) out.println(this+"Getting types for "+n);
            if (n instanceof ConcreteTypeNode) {
                // n is a concrete type, so we know the target exactly.
                jq_Reference r = ((ConcreteTypeNode)n).type;
                if (TRACE_INTER) out.println(this+"Concrete : adding "+r);
                result.add(r);
            } else if (n instanceof UnknownTypeNode) {
                // n is an unknown type!
                // we need to add ALL possible subtypes.
                // TODO.
            } else if (n instanceof ReturnValueNode) {
                // n is the return value of a method call.
                // we need to analyze that method call to find the types.
                MethodCall that_mc = ((ReturnValueNode)n).m;
                getReturnedTypes(that_mc.q, result, qwery);
            } else if (n instanceof ThrownExceptionNode) {
                // n is an exception thrown by a method call.
                // we need to analyze that method call to find the types.
                MethodCall that_mc = ((ThrownExceptionNode)n).m;
                getThrownTypes(that_mc.q, result, qwery);
            } else if (n instanceof CaughtExceptionNode) {
                // n is a caught exception.
                // TODO.
            } else if (n instanceof ParamNode) {
                // n is one of OUR parameters!
                // we need to look at our calling context to find the types.
                Check check = null;
                if (qwery != null) check = new ParamCheck((ParamNode)n);
                HashSet memo2 = new HashSet();
                PassedParameter pp = new PassedParameter(call_site, ((ParamNode)n).n);
                this.parent.getTypesOfParameter(pp, memo2, null);
                if (qwery != null) {
                    check.setExpectedResult(memo2);
                    qwery.addCheck(check);
                }
                result.addAll(memo2);
            } else if (n instanceof FieldNode) {
                // n is from a field dereference!
                // we need to find all possible writes into this field node.
                // (i.e. we need the pointer information for all predecessors)
                getPointerWrites((FieldNode)n, result, qwery);
            } else {
                jq.UNREACHABLE(n.toString());
            }
        }

        public void getPointerWrites(FieldNode n, HashSet result, QueryResult qwery) {
            HashSet memo = (HashSet)local_memoizer.get(n);
            if (memo != null) {
                if (TRACE_INTER) out.println(this+"Memoizer succeeded for "+n+":"+memo);
                result.addAll(memo); return;
            }
            //if (TRACE_INTER) out.println(this+"Memoizer contains: "+local_memoizer);
            local_memoizer.put(n, memo = new HashSet());
            if (TRACE_INTER) out.println(this+"Getting pointer writes into "+n);
            if (n.f instanceof jq_StaticField) {
                // double-check the static initializer
            }
            AccessPath ap = new AccessPath(null, n.f);
            // find all possible writes into nodes that were used as a base object.
            for (Iterator i=n.field_predecessors.iterator(); i.hasNext(); ) {
                Node base = (Node)i.next();
                getPointerWrites(base, ap, memo, qwery);
            }
            result.addAll(memo);
        }

        public boolean canCallWriteToAccessPath(Node n, AccessPath ap) {
            // todo: do we need to worry about returned/thrown?
            if (n.passedParameters == null) return false;
            for (Iterator i=n.passedParameters.iterator(); i.hasNext(); ) {
                PassedParameter pp = (PassedParameter)i.next();
                if (canWriteToAccessPath(pp, ap)) return true;
            }
            return false;
        }
        public boolean canWriteToAccessPath(HashSet fns, AccessPath ap) {
            if (ap.parent == null) {
                for (Iterator i=fns.iterator(); i.hasNext(); ) {
                    FieldNode fn = (FieldNode)i.next();
                    if ((fn.addedEdges != null) &&
                       (fn.addedEdges.get(ap.field) != null)) return true;
                    if (canCallWriteToAccessPath(fn, ap)) return true;
                }
                return false;
            } else {
                HashSet nodes = new HashSet();
                for (Iterator i=fns.iterator(); i.hasNext(); ) {
                    FieldNode fn = (FieldNode)i.next();
                    if (canCallWriteToAccessPath(fn, ap)) return true;
                    fn.getAccessPathEdges(ap.field, nodes);
                }
                return canWriteToAccessPath(nodes, ap.parent);
            }
        }
        public boolean canWriteToAccessPath(PassedParameter pp, AccessPath p) {
            if (TRACE_INTER) out.println(this+"Checking if call "+pp+" can write into access path "+p);
            MethodCall mc = pp.m;
            int n = pp.paramNum;
            Quad q = mc.q;
            HashSet targets = new HashSet();
            getCallTargets(q, targets, null);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                ParamNode pn = s_callee.params[n];
                if (context2.canCallWriteToAccessPath(pn, p)) return true;
                if (p.parent != null) {
                    HashSet nodes = new HashSet();
                    pn.getAccessPathEdges(p.field, nodes);
                    if (context2.canWriteToAccessPath(nodes, p.parent)) return true;
                } else {
                    if ((pn.addedEdges != null) &&
                       (pn.addedEdges.get(p.field) != null)) return true;
                }
            }
            return false;
        }
        
        public void getPointerWrites(PassedParameter pp, AccessPath p, HashSet result, QueryResult qwery) {
            if (TRACE_INTER) out.println(this+"Getting pointer writes for call "+pp+" access path "+p);
            MethodCall mc = pp.m;
            int n = pp.paramNum;
            Quad q = mc.q;
            HashSet targets = new HashSet();
            getCallTargets(q, targets, qwery);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                ParamNode pn = s_callee.params[n];
                NodeAndAccessPath nap = new NodeAndAccessPath(pn, p);
                HashSet res = s_callee.getQuery(nap, this, mc);
                if (res == null) {
                    res = new HashSet();
                    QueryResult query = new QueryResult(nap);
                    CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                    context2.getPointerWrites(pn, p, res, query);
                    query.result = res;
                    s_callee.registerQuery(nap, query);
                }
                result.addAll(res);
            }
        }
        
        public void getPointerWrites(Node n, AccessPath p, HashSet result, QueryResult qwery) {
            NodeAndAccessPath rv = new NodeAndAccessPath(n, p);
            HashSet memo = (HashSet)local_memoizer.get(rv);
            if (memo != null) {
                if (TRACE_INTER) out.println(this+"Memoizer succeeded for "+rv+":"+memo);
                result.addAll(memo); return;
            }
            //if (TRACE_INTER) out.println(this+"Memoizer contains "+local_memoizer);
            local_memoizer.put(rv, memo = new HashSet());
            if (TRACE_INTER) out.println(this+"Getting pointer writes for node "+n+" access path "+p);
            if (p.parent == null) {
                // single level access path.
                // get all writes into this field in our context.
                HashSet nodes = new HashSet();
                n.getEdges(p.field, nodes);
                for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                    Node v = (Node)i.next();
                    if (TRACE_INTER) out.println(this+"Node "+v+" is written into node "+n+" field "+p.field);
                    this.getTypesOfNode(v, memo, qwery);
                }
            } else {
                // multi level access path.
            }
            
            // look at all calls on this node in our context
            if (n.passedParameters != null) {
                for (Iterator i=n.passedParameters.iterator(); i.hasNext(); ) {
                    PassedParameter pp = (PassedParameter)i.next();
                    if (this.canWriteToAccessPath(pp, p)) {
                        if (TRACE_INTER) out.println(this+"Call "+pp+" CAN write into access path "+p);
                        this.getPointerWrites(pp, p, memo, qwery);
                    }
                }
            }
            
            if (n instanceof ParamNode) {
                // n was a parameter passed in from the caller.
                // need to check the caller for writes/calls.
                PassedParameter pp = new PassedParameter(call_site, ((ParamNode)n).n);
                Check check = null;
                if (qwery != null) check = new ParamPathCheck((ParamNode)n, p);
                HashSet nodes = new HashSet();
                this.parent.getNodes(pp, nodes);
                if (TRACE_INTER) out.println(this+""+n+" is from caller, so checking caller nodes "+nodes+" for writes/calls into access path "+p);
                HashSet memo2 = new HashSet();
                for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                    this.parent.getPointerWrites((Node)i.next(), p, memo2, null);
                }
                if (qwery != null) {
                    check.setExpectedResult(memo2);
                    qwery.addCheck(check);
                }
                memo.addAll(memo2);
            } else if (n instanceof ReturnValueNode) {
                // n is the return value of a method call.
                // need to check the callee for writes/calls.
                MethodCall that_mc = ((ReturnValueNode)n).m;
                getPointerWritesIntoReturned(that_mc.q, p, memo, qwery);
            } else if (n instanceof ThrownExceptionNode) {
                // n is an exception thrown by a method call.
                // need to check the callee for writes/calls.
                MethodCall that_mc = ((ThrownExceptionNode)n).m;
                getPointerWritesIntoThrown(that_mc.q, p, memo, qwery);
            } else if (n instanceof FieldNode) {
                // n is from a field dereference.
                // see below.
            } else if (n instanceof GlobalNode) {
                // n is the global node.
                // no need to do anything.
            } else if (n instanceof ConcreteTypeNode) {
                // n is a concrete type.
                // no need to do anything.
            } else if (n instanceof UnknownTypeNode) {
                // n is an unknown type.
                // no need to do anything.
            } else if (n instanceof CaughtExceptionNode) {
                // n is a caught exception.
                // TODO.
            } else {
                jq.UNREACHABLE(n.toString());
            }
            
            // find nodes that point to us, and check if there are any writes
            // through the (one) longer access path through those nodes.
            if (n.predecessors != null) {
                for (Iterator i=n.predecessors.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    AccessPath new_ap = new AccessPath(p, f);
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        this.getPointerWrites((Node)o, new_ap, memo, qwery);
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            this.getPointerWrites((Node)j.next(), new_ap, memo, qwery);
                        }
                    }
                }
            }
            if (n instanceof FieldNode) {
                FieldNode fn = (FieldNode)n;
                AccessPath new_ap = new AccessPath(p, fn.f);
                for (Iterator i=fn.field_predecessors.iterator(); i.hasNext(); ) {
                    this.getPointerWrites((Node)i.next(), new_ap, memo, qwery);
                }
            }
            result.addAll(memo);
        }
        
        public void getPointerWritesIntoReturned(Quad q, AccessPath p, HashSet result, QueryResult qwery) {
            if (TRACE_INTER) out.println(this+"finding writes along access path "+p+" into the return value for "+q);
            HashSet targets = new HashSet();
            getCallTargets(q, targets, qwery);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                MethodCall mc = new MethodCall(m, q);
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                if (s_callee.returned == null) continue;
                CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                for (Iterator j=s_callee.returned.iterator(); j.hasNext(); ) {
                    Node n = (Node)j.next();
                    NodeAndAccessPath nap = new NodeAndAccessPath(n, p);
                    HashSet res = s_callee.getQuery(nap, this, mc);
                    if (res == null) {
                        res = new HashSet();
                        QueryResult query = new QueryResult(nap);
                        context2.getPointerWrites(n, p, res, query);
                        query.result = res;
                        s_callee.registerQuery(nap, query);
                    }
                    result.addAll(res);
                }
            }
        }
        
        public void getPointerWritesIntoThrown(Quad q, AccessPath p, HashSet result, QueryResult qwery) {
            if (TRACE_INTER) out.println(this+"finding writes along access path "+p+" into the thrown value for "+q);
            HashSet targets = new HashSet();
            getCallTargets(q, targets, qwery);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                MethodCall mc = new MethodCall(m, q);
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                if (s_callee.returned == null) continue;
                CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                for (Iterator j=s_callee.thrown.iterator(); j.hasNext(); ) {
                    Node n = (Node)j.next();
                    NodeAndAccessPath nap = new NodeAndAccessPath(n, p);
                    HashSet res = s_callee.getQuery(nap, this, mc);
                    if (res == null) {
                        res = new HashSet();
                        QueryResult query = new QueryResult(nap);
                        context2.getPointerWrites(n, p, res, query);
                        query.result = res;
                        s_callee.registerQuery(nap, query);
                    }
                    result.addAll(res);
                }
            }
        }
        
        public void getReturnedTypes(Quad q, HashSet result, QueryResult qwery) {
            ReturnValue rv = new ReturnValue(q);
            HashSet memo = (HashSet)local_memoizer.get(rv);
            if (memo != null) {
                if (TRACE_INTER) out.println(this+"Memoizer succeeded for "+rv+":"+memo);
                result.addAll(memo); return;
            }
            //if (TRACE_INTER) out.println(this+"Memoizer contains "+local_memoizer);
            local_memoizer.put(rv, memo = new HashSet());
            if (TRACE_INTER) out.println(this+"Getting returned types of "+q);
            HashSet targets = new HashSet();
            getCallTargets(q, targets, qwery);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                MethodCall mc = new MethodCall(m, q);
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                if (s_callee.returned == null) continue;
                CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                for (Iterator j=s_callee.returned.iterator(); j.hasNext(); ) {
                    Node n = (Node)j.next();
                    context2.getTypesOfNode(n, memo, null);
                }
            }
            result.addAll(memo);
            //if (TRACE_INTER) out.println(this+"Memoizer now contains "+local_memoizer);
        }
        
        public void getThrownTypes(Quad q, HashSet result, QueryResult qwery) {
            ThrownValue rv = new ThrownValue(q);
            HashSet memo = (HashSet)local_memoizer.get(rv);
            if (memo != null) {
                result.addAll(memo); return;
            }
            local_memoizer.put(rv, memo = new HashSet());
            if (TRACE_INTER) out.println(this+"Getting thrown types of "+q);
            HashSet targets = new HashSet();
            getCallTargets(q, targets, qwery);
            for (Iterator i=targets.iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                MethodCall mc = new MethodCall(m, q);
                m.getDeclaringClass().load();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                if (s_callee.thrown == null) continue;
                CallingContext context2 = getChildCallingContext(mc, m, s_callee);
                for (Iterator j=s_callee.thrown.iterator(); j.hasNext(); ) {
                    Node n = (Node)j.next();
                    context2.getTypesOfNode(n, result, null);
                }
            }
            result.addAll(memo);
        }
        
        public void getCallTargets(Quad q, HashSet result, QueryResult qwery) {
            jq_Method target = Invoke.getMethod(q).getMethod();
            if (!((Invoke)q.getOperator()).isVirtual()) {
                result.add(target);
                return;
            }
            if (TRACE_INTER) out.println(this+"Getting targets of virtual call "+q);
            // find the set of types for the 'this' pointer.
            MethodCall mc = new MethodCall(target, q);
            PassedParameter pp = new PassedParameter(mc, 0);
            HashSet this_types = new HashSet();
            getTypesOfParameter(pp, this_types, qwery);
           // look up the target methods and add them.
            for (Iterator i = this_types.iterator(); i.hasNext(); ) {
                jq_Reference r = (jq_Reference)i.next();
                r.load(); r.verify(); r.prepare();
                jq_Method resolved_target = r.getVirtualMethod(target.getNameAndDesc());
                if (resolved_target != null) {
                    result.add(resolved_target);
                } else {
                    if (TRACE_INTER) out.println("Invalid call! "+r+" on target "+target);
                }
            }
        }
    
    }
         */
    
    public static class AccessPath {
        
        jq_Field _field;
        Node _n;
        boolean _last;
        
        HashSet succ;

        public void reachable(HashSet s) {
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    s.add(ap);
                    ((AccessPath)ap.getObject()).reachable(s);
                }
            }
        }
        public Iterator reachable() {
            HashSet s = new HashSet();
            s.add(IdentityHashCodeWrapper.create(this));
            this.reachable(s);
            return new FilterIterator(s.iterator(), filter);
        }
        
        public void addSuccessor(AccessPath ap) {
            succ.add(IdentityHashCodeWrapper.create(ap));
        }
        
        public static AccessPath create(jq_Field f, Node n, AccessPath p) {
            if (p == null) return new AccessPath(f, n, true);
            AccessPath that = p.findNode(n);
            if (that == null) {
                that = new AccessPath(f, n);
            } else {
                p = p.copy();
                that = p.findNode(n);
            }
            that.addSuccessor(p);
            return that;
        }
        
        public static AccessPath create(AccessPath p, jq_Field f, Node n) {
            if (p == null) return new AccessPath(f, n, true);
            p = p.copy();
            AccessPath that = p.findNode(n);
            if (that == null) {
                that = new AccessPath(f, n);
            }
            that.setLast();
            for (Iterator i = p.findLast(); i.hasNext(); ) {
                AccessPath last = (AccessPath)i.next();
                last.unsetLast();
                last.addSuccessor(that);
            }
            return p;
        }
        
        private void findLast(HashSet s, HashSet last) {
            if (this._last) last.add(IdentityHashCodeWrapper.create(this));
            s.add(IdentityHashCodeWrapper.create(this));
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    ((AccessPath)ap.getObject()).findLast(s, last);
                }
            }
        }
        
        public Iterator findLast() {
            HashSet visited = new HashSet();
            HashSet last = new HashSet();
            findLast(visited, last);
            return new FilterIterator(last.iterator(), filter);
        }
        
        private AccessPath findNode(Node n, HashSet s) {
            if (n == this._n) return this;
            s.add(IdentityHashCodeWrapper.create(this));
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    AccessPath q = ((AccessPath)ap.getObject()).findNode(n, s);
                    if (q != null) return q;
                }
            }
            return null;
        }
            
        public AccessPath findNode(Node n) {
            HashSet visited = new HashSet();
            return findNode(n, visited);
        }
        
        public void setLast() { this._last = true; }
        public void unsetLast() { this._last = false; }
        
        private AccessPath copy(HashMap m) {
            AccessPath that = (AccessPath)m.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                that = new AccessPath(this._field, this._n, this._last);
                m.put(IdentityHashCodeWrapper.create(this), that);
                for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                    IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                    that.addSuccessor(((AccessPath)ap.getObject()).copy(m));
                }
            }
            return that;
        }
        public AccessPath copy() {
            HashMap m = new HashMap();
            return this.copy(m);
        }
        
        private void toString(StringBuffer sb, HashSet set) {
            set.add(IdentityHashCodeWrapper.create(this));
            if (this._field == null) sb.append("[]");
            else sb.append(this._field.getName());
            if (this._last) sb.append("<e>");
            sb.append("->(");
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (set.contains(ap)) {
                    sb.append("<backedge>");
                } else {
                    ((AccessPath)ap.getObject()).toString(sb, set);
                }
            }
            sb.append(')');
        }
        public String toString() {
            StringBuffer sb = new StringBuffer();
            HashSet visited = new HashSet();
            toString(sb, visited);
            return sb.toString();
        }
        
        private AccessPath(jq_Field f, Node n, boolean last) {
            this._field = f; this._n = n; this._last = last;
            this.succ = new HashSet();
        }
        private AccessPath(jq_Field f, Node n) {
            this(f, n, false);
        }
        public jq_Field getField() { return _field; }
        
        private boolean equals(AccessPath that, HashSet s) {
            //if (this._n != that._n) return false;
            if (this._field != that._field) return false;
            if (this._last != that._last) return false;
            if (this.succ.size() != that.succ.size()) return false;
            s.add(IdentityHashCodeWrapper.create(this));
            for (Iterator i = this.succ.iterator(), j = that.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper a = (IdentityHashCodeWrapper)i.next();
                IdentityHashCodeWrapper b = (IdentityHashCodeWrapper)j.next();
                if (s.contains(a)) continue;
                if (!((AccessPath)a.getObject()).equals(((AccessPath)b.getObject()), s)) return false;
            }
            return true;
        }
        public boolean equals(AccessPath that) {
            HashSet s = new HashSet();
            return this.equals(that, s);
        }
        public boolean equals(Object o) {
            if (o instanceof AccessPath) return equals((AccessPath)o);
            return false;
        }
        public int hashCode() {
            int x = this.local_hashCode();
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper a = (IdentityHashCodeWrapper)i.next();
                x ^= (((AccessPath)a.getObject()).local_hashCode() << 1);
            }
            return x;
        }
        private int local_hashCode() {
            return _field != null ? _field.hashCode() : 0x31337;
        }
        public jq_Field first() { return _field; }
        public Iterator next() {
            return new FilterIterator(succ.iterator(), filter);
        }
        public static final FilterIterator.Filter filter = new FilterIterator.Filter() {
            public Object map(Object o) { return ((IdentityHashCodeWrapper)o).getObject(); }
        };
    }
    
    /** Creates new ContextSensitiveCallGraph */
    public ContextSensitiveCallGraph() {
    }

    public static class MethodCall {
        final jq_Method m; final Quad q;
        public MethodCall(jq_Method m, Quad q) {
            this.m = m; this.q = q;
        }
        public int hashCode() { return (q==null)?-1:q.hashCode(); }
        public boolean equals(MethodCall that) { return this.q == that.q; }
        public boolean equals(Object o) { if (o instanceof MethodCall) return equals((MethodCall)o); return false; }
        public String toString() { return "quad "+((q==null)?-1:q.getID())+" "+m.getName()+"()"; }
    }
    
    public static class PassedParameter {
        final MethodCall m; final int paramNum;
        public PassedParameter(MethodCall m, int paramNum) {
            this.m = m; this.paramNum = paramNum;
        }
        public int hashCode() { return m.hashCode() ^ paramNum; }
        public boolean equals(PassedParameter that) { return this.m.equals(that.m) && this.paramNum == that.paramNum; }
        public boolean equals(Object o) { if (o instanceof PassedParameter) return equals((PassedParameter)o); return false; }
        public String toString() { return "Param "+paramNum+" for "+m; }
    }
    
    public abstract static class Node implements Cloneable {
        HashMap predecessors;
        HashSet passedParameters;
        HashMap addedEdges;
        HashMap accessPathEdges;
        
        Node() {}
        Node(Node n) {
            this.predecessors = n.predecessors;
            this.passedParameters = n.passedParameters;
            this.addedEdges = n.addedEdges;
            this.accessPathEdges = n.accessPathEdges;
        }
        
        void replaceBy(HashSet set) {
            if (this.predecessors != null) {
                for (Iterator i=this.predecessors.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        Node that = (Node)o;
                        if (that == this) {
                            // add self-cycles on f to all nodes in set.
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                Node k = (Node)j.next();
                                k.addEdge(f, k);
                            }
                            i.remove();
                            continue;
                        }
                        that.removeEdge(f, this);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            that.addEdge(f, (Node)j.next());
                        }
                    } else {
                        for (Iterator k=((HashSet)o).iterator(); k.hasNext(); ) {
                            Node that = (Node)k.next();
                            if (that == this) {
                                // add self-cycles on f to all mapped nodes.
                                for (Iterator j=set.iterator(); j.hasNext(); ) {
                                    Node k2 = (Node)j.next();
                                    k2.addEdge(f, k2);
                                }
                                k.remove();
                                continue;
                            }
                            that.removeEdge(f, this);
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                that.addEdge(f, (Node)j.next());
                            }
                        }
                    }
                }
            }
            if (this.addedEdges != null) {
                for (Iterator i=this.addedEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        Node that = (Node)o;
                        jq.assert(that != this); // cyclic edges handled above.
                        this.removeEdge(f, that);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            that.addEdge(f, (Node)j.next());
                        }
                    } else {
                        for (Iterator k=((HashSet)o).iterator(); k.hasNext(); ) {
                            Node that = (Node)k.next();
                            jq.assert(that != this); // cyclic edges handled above.
                            this.removeEdge(f, that);
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                that.addEdge(f, (Node)j.next());
                            }
                        }
                    }
                }
            }
            if (this.accessPathEdges != null) {
                for (Iterator i=this.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        Node that = (Node)o;
                        jq.assert(that != this); // cyclic edges handled above.
                        this.removeEdge(f, that);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            that.addAccessPathEdge(f, (FieldNode)j.next());
                        }
                    } else {
                        for (Iterator k=((HashSet)o).iterator(); k.hasNext(); ) {
                            Node that = (Node)k.next();
                            jq.assert(that != this); // cyclic edges handled above.
                            this.removeEdge(f, that);
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                that.addAccessPathEdge(f, (FieldNode)j.next());
                            }
                        }
                    }
                }
            }
            if (this.passedParameters != null) {
                for (Iterator i=this.passedParameters.iterator(); i.hasNext(); ) {
                    PassedParameter pp = (PassedParameter)i.next();
                    for (Iterator j=set.iterator(); j.hasNext(); ) {
                        ((Node)j.next()).recordPassedParameter(pp);
                    }
                }
            }
        }
        
        static void updateMap(HashMap um, Iterator i, HashMap m) {
            while (i.hasNext()) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                if (o instanceof Node) {
                    m.put(f, um.get(o));
                } else {
                    for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                        m.put(f, um.get(j.next()));
                    }
                }
            }
        }
        
        public void update(HashMap um) {
            HashMap m = this.predecessors;
            if (m != null) {
                this.predecessors = new HashMap();
                updateMap(um, m.entrySet().iterator(), this.predecessors);
            }
            m = this.addedEdges;
            if (m != null) {
                this.addedEdges = new HashMap();
                updateMap(um, m.entrySet().iterator(), this.addedEdges);
            }
            m = this.accessPathEdges;
            if (m != null) {
                this.accessPathEdges = new HashMap();
                updateMap(um, m.entrySet().iterator(), this.accessPathEdges);
            }
            if (this.passedParameters != null) {
                this.passedParameters = (HashSet)this.passedParameters.clone();
            }
        }
        
        public abstract jq_Reference getDeclaredType();
        
        public boolean equals(Node that) {
            if (this.predecessors != that.predecessors) {
                if ((this.predecessors == null) || (that.predecessors == null)) return false;
                if (!this.predecessors.equals(that.predecessors)) return false;
            }
            if (this.passedParameters != that.passedParameters) {
                if ((this.passedParameters == null) || (that.passedParameters == null)) return false;
                if (!this.passedParameters.equals(that.passedParameters)) return false;
            }
            if (this.addedEdges != that.addedEdges) {
                if ((this.addedEdges == null) || (that.addedEdges == null)) return false;
                if (!this.addedEdges.equals(that.addedEdges)) return false;
            }
            if (this.accessPathEdges != that.accessPathEdges) {
                if ((this.accessPathEdges == null) || (that.accessPathEdges == null)) return false;
                if (!this.accessPathEdges.equals(that.accessPathEdges)) return false;
            }
            return true;
        }
        public boolean equals(Object o) {
            if (o instanceof Node) return equals((Node)o);
            return false;
        }
        public Object clone() { return this.copy(); }
        
        abstract Node copy();

        boolean removePredecessor(jq_Field m, Node n) {
            if (predecessors == null) return false;
            Object o = predecessors.get(m);
            if (o instanceof HashSet) return ((HashSet)o).remove(n);
            else if (o == n) { predecessors.remove(m); return true; }
            else return false;
        }
        boolean addPredecessor(jq_Field m, Node n) {
            if (predecessors == null) predecessors = new HashMap();
            Object o = predecessors.get(m);
            if (o == null) {
                predecessors.put(m, n);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).add(n);
            if (o == n) return false;
            HashSet s = new HashSet(); s.add(o); s.add(n);
            predecessors.put(m, s);
            return true;
        }
        
        boolean recordPassedParameter(PassedParameter cm) {
            if (passedParameters == null) passedParameters = new HashSet();
            return passedParameters.add(cm);
        }
        boolean recordPassedParameter(MethodCall m, int paramNum) {
            if (passedParameters == null) passedParameters = new HashSet();
            PassedParameter cm = new PassedParameter(m, paramNum);
            return passedParameters.add(cm);
        }
        boolean removeEdge(jq_Field m, Node n) {
            if (addedEdges == null) return false;
            n.removePredecessor(m, this);
            Object o = addedEdges.get(m);
            if (o instanceof HashSet) return ((HashSet)o).remove(n);
            else if (o == n) { addedEdges.remove(m); return true; }
            else return false;
        }
        boolean addEdge(jq_Field m, Node n) {
            n.addPredecessor(m, this);
            if (addedEdges == null) addedEdges = new HashMap();
            Object o = addedEdges.get(m);
            if (o == null) {
                addedEdges.put(m, n);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).add(n);
            if (o == n) return false;
            HashSet s = new HashSet(); s.add(o); s.add(n);
            addedEdges.put(m, s);
            return true;
        }
        boolean addEdges(jq_Field m, HashSet s) {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                n.addPredecessor(m, this);
            }
            if (addedEdges == null) addedEdges = new HashMap();
            Object o = addedEdges.get(m);
            if (o == null) {
                addedEdges.put(m, s);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).addAll(s);
            addedEdges.put(m, s); return s.add(o); 
        }
        boolean removeAccessPathEdge(jq_Field m, FieldNode n) {
            if (accessPathEdges == null) return false;
            if (n.field_predecessors != null) n.field_predecessors.remove(this);
            Object o = accessPathEdges.get(m);
            if (o instanceof HashSet) return ((HashSet)o).remove(n);
            else if (o == n) { accessPathEdges.remove(m); return true; }
            else return false;
        }
        boolean addAccessPathEdge(jq_Field m, FieldNode n) {
            if (n.field_predecessors == null) n.field_predecessors = new HashSet();
            n.field_predecessors.add(this);
            if (accessPathEdges == null) accessPathEdges = new HashMap();
            Object o = accessPathEdges.get(m);
            if (o == null) {
                accessPathEdges.put(m, n);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).add(n);
            if (o == n) return false;
            HashSet s = new HashSet(); s.add(o); s.add(n);
            accessPathEdges.put(m, s);
            return true;
        }
        boolean addAccessPathEdges(jq_Field m, HashSet s) {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                FieldNode n = (FieldNode)i.next();
                if (n.field_predecessors == null) n.field_predecessors = new HashSet();
                n.field_predecessors.add(this);
            }
            if (accessPathEdges == null) accessPathEdges = new HashMap();
            Object o = accessPathEdges.get(m);
            if (o == null) {
                accessPathEdges.put(m, s);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).addAll(s);
            accessPathEdges.put(m, s); return s.add(o); 
        }
        
        void getEdges(jq_Field m, HashSet result) {
            if (addedEdges == null) return;
            Object o = addedEdges.get(m);
            if (o == null) return;
            if (o instanceof HashSet) {
                result.addAll((HashSet)o);
            } else {
                result.add(o);
            }
        }

        void getAccessPathEdges(jq_Field m, HashSet result) {
            if (accessPathEdges == null) return;
            Object o = accessPathEdges.get(m);
            if (o == null) return;
            if (o instanceof HashSet) {
                result.addAll((HashSet)o);
            } else {
                result.add(o);
            }
        }
        
        public abstract String toString_short();
        public String toString() { return toString_short(); }
        public String toString_long() {
            StringBuffer sb = new StringBuffer();
            if (addedEdges != null) {
                sb.append(" writes: ");
                for (Iterator i=addedEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    sb.append(f);
                    sb.append("={");
                    if (o instanceof Node)
                        sb.append(((Node)o).toString_short());
                    else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                           sb.append(((Node)j.next()).toString_short());
                           if (j.hasNext()) sb.append(", ");
                        }
                    sb.append("}");
                    }
                }
            }
            if (accessPathEdges != null) {
                sb.append(" reads: ");
                sb.append(accessPathEdges);
            }
            if (passedParameters != null) {
                sb.append(" called: ");
                sb.append(passedParameters);
            }
            return sb.toString();
        }
    }
    
    public static final class ConcreteTypeNode extends Node {
        jq_Reference type; Quad q;
        
        ConcreteTypeNode(jq_Reference type, Quad q) { this.type = type; this.q = q; }
        ConcreteTypeNode(ConcreteTypeNode that) {
            super(that); this.type = that.type; this.q = that.q;
        }
        
        public jq_Reference getDeclaredType() { return type; }
        
        public boolean equals(ConcreteTypeNode that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof ConcreteTypeNode) return equals((ConcreteTypeNode)o);
            else return false;
        }
        public int hashCode() { return q.hashCode(); }
        
        Node copy() { return new ConcreteTypeNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "Concrete: "+type+" q: "+q.getID(); }
    }
    
    public static final class UnknownTypeNode extends Node {
        static final HashMap FACTORY = new HashMap();
        public static UnknownTypeNode get(jq_Reference type) {
            UnknownTypeNode n = (UnknownTypeNode)FACTORY.get(type);
            if (n == null) FACTORY.put(type, n = new UnknownTypeNode(type));
            return n;
        }
        
        jq_Reference type;
        
        UnknownTypeNode(jq_Reference type, Quad q) { this.type = type; }
        UnknownTypeNode(jq_Reference type) { this.type = type; }
        private UnknownTypeNode(UnknownTypeNode that) { super(that); this.type = that.type; }
        
        public jq_Reference getDeclaredType() { return type; }
        
        public boolean equals(UnknownTypeNode that) { return this.type == that.type; }
        public boolean equals(Object o) {
            if (o instanceof UnknownTypeNode) return equals((UnknownTypeNode)o);
            else return false;
        }
        public int hashCode() { return type.hashCode(); }
        
        Node copy() { return new UnknownTypeNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "Unknown: "+type; }
    }
    
    public abstract static class OutsideNode extends Node {
        OutsideNode() {}
        OutsideNode(Node n) { super(n); }
    }
    
    public static final class GlobalNode extends OutsideNode {
        GlobalNode() {}
        public jq_Reference getDeclaredType() { jq.UNREACHABLE(); return null; }
        Node copy() { return this; }
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "global"; }
        public static final GlobalNode GLOBAL = new GlobalNode();
    }
    
    public static final class ReturnValueNode extends OutsideNode {
        MethodCall m;
        ReturnValueNode(MethodCall m) { this.m = m; }
        ReturnValueNode(ReturnValueNode that) {
            super(that); this.m = that.m;
        }
        public boolean equals(ReturnValueNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ReturnValueNode) return equals((ReturnValueNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        public jq_Reference getDeclaredType() { return (jq_Reference)m.m.getReturnType(); }
        
        Node copy() { return new ReturnValueNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "Return value of "+m; }
    }
    
    public static final class CaughtExceptionNode extends OutsideNode {
        ExceptionHandler eh;
        CaughtExceptionNode(ExceptionHandler eh) { this.eh = eh; }
        CaughtExceptionNode(CaughtExceptionNode that) {
            super(that); this.eh = that.eh;
        }
        public boolean equals(CaughtExceptionNode that) { return this.eh.equals(that.eh); }
        public boolean equals(Object o) {
            if (o instanceof CaughtExceptionNode) return equals((CaughtExceptionNode)o);
            else return false;
        }
        public int hashCode() { return eh.hashCode(); }
        
        public jq_Reference getDeclaredType() { return (jq_Reference)eh.getExceptionType(); }
        
        Node copy() { return new CaughtExceptionNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "Caught exception: "+eh; }
    }
    
    public static final class ThrownExceptionNode extends OutsideNode {
        MethodCall m;
        ThrownExceptionNode(MethodCall m) { this.m = m; }
        ThrownExceptionNode(ThrownExceptionNode that) {
            super(that); this.m = that.m;
        }
        public boolean equals(ThrownExceptionNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ThrownExceptionNode) return equals((ThrownExceptionNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        public jq_Reference getDeclaredType() { return Bootstrap.PrimordialClassLoader.getJavaLangObject(); }
        
        Node copy() { return new ThrownExceptionNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return "Thrown exception of "+m; }
    }
    
    public static final class ParamNode extends OutsideNode {
        jq_Method m; int n; jq_Reference declaredType;
        
        ParamNode(jq_Method m, int n, jq_Reference declaredType) { this.m = m; this.n = n; this.declaredType = declaredType; }
        ParamNode(ParamNode that) {
            super(that); this.m = that.m; this.n = that.n; this.declaredType = that.declaredType;
        }

        public boolean equals(ParamNode that) { return this.n == that.n && this.m == that.m; }
        public boolean equals(Object o) {
            if (o instanceof ParamNode) return equals((ParamNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode() ^ n; }
        
        public jq_Reference getDeclaredType() { return declaredType; }
        
        Node copy() { return new ParamNode(this); }
        
        public String toString_long() { return this.toString_short()+super.toString_long(); }
        public String toString_short() { return "Param#"+n+" method "+m.getName(); }
    }
    
    public static final class FieldNode extends OutsideNode {
        jq_Field f; Quad q;
        HashSet field_predecessors;
        
        FieldNode(jq_Field f, Quad q) { this.f = f; this.q = q; }
        FieldNode(FieldNode that) {
            super(that); this.f = that.f; this.q = that.q; this.field_predecessors = that.field_predecessors;
        }

        void unify(FieldNode that) {
            jq.assert(this.f == that.f);
            this.q = null; that.q = null;
            HashSet s = new HashSet(); s.add(this);
            that.replaceBy(s);
        }
        
        void replaceBy(HashSet set) {
            if (this.field_predecessors != null) {
                for (Iterator i=this.field_predecessors.iterator(); i.hasNext(); ) {
                    Node that = (Node)i.next();
                    if (that == this) {
                        // add self-cycles on f to all nodes in set.
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            FieldNode k = (FieldNode)j.next();
                            k.addAccessPathEdge(f, k);
                        }
                        i.remove();
                        continue;
                    }
                    that.removeAccessPathEdge(f, this);
                    for (Iterator j=set.iterator(); j.hasNext(); ) {
                        that.addAccessPathEdge(f, (FieldNode)j.next());
                    }
                }
            }
            super.replaceBy(set);
        }
        
        public void update(HashMap um) {
            super.update(um);
            HashSet m = this.field_predecessors;
            if (m != null) {
                this.field_predecessors = new HashSet();
                for (Iterator j=m.iterator(); j.hasNext(); ) {
                    this.field_predecessors.add(um.get(j.next()));
                }
            }
        }
        
        public boolean equals(FieldNode that) { return this.q == that.q && this.f == that.f; }
        public boolean equals(Object o) {
            if (o instanceof FieldNode) return equals((FieldNode)o);
            else return false;
        }
        public int hashCode() { return f.hashCode(); }
        
        public String fieldName() {
            if (f != null) return f.getName().toString();
            return getDeclaredType()+"[]";
        }
        
        public jq_Reference getDeclaredType() {
            if (f != null) {
                return (jq_Reference)f.getType();
            }
            RegisterOperand r = ALoad.getDest(q);
            return (jq_Reference)r.getType();
        }
        
        Node copy() { return new FieldNode(this); }
        
        public String toString_long() { return this.toString_short()+super.toString_long(); }
        public String toString_short() { return "FieldLoad "+fieldName()+" quad "+q.getID(); }
    }
    
    public static final class State implements Cloneable {
        final Object[] registers;
        State(int nRegisters) {
            this.registers = new Object[nRegisters];
        }
        public Object clone() { return this.copy(); }
        State copy() {
            State that = new State(this.registers.length);
            for (int i=0; i<this.registers.length; ++i) {
                Object a = this.registers[i];
                if (a instanceof Node)
                    //that.registers[i] = ((Node)a).copy();
                    that.registers[i] = a;
                else if (a instanceof HashSet)
                    that.registers[i] = ((HashSet)a).clone();
            }
            return that;
        }
        boolean merge(State that) {
            boolean change = false;
            for (int i=0; i<this.registers.length; ++i) {
                Object b = that.registers[i];
                if (b == null) continue;
                Object a = this.registers[i];
                if (b.equals(a)) continue;
                HashSet q;
                if (!(a instanceof HashSet)) {
                    this.registers[i] = q = new HashSet();
                    q.add(a);
                } else {
                    q = (HashSet)a;
                }
                if (b instanceof HashSet) {
                    if (q.addAll((HashSet)b)) {
                        if (TRACE_INTRA) out.println("change in register "+i+" from adding set");
                        change = true;
                    }
                } else {
                    if (q.add(b)) {
                        if (TRACE_INTRA) out.println("change in register "+i+" from adding "+b);
                        change = true;
                    }
                }
            }
            return change;
        }
        void dump(java.io.PrintStream out) {
            for (int i=0; i<registers.length; ++i) {
                if (registers[i] == null) continue;
                out.print(i+": "+registers[i]+" ");
            }
            out.println();
        }
    }
    
    public static final class BuildMethodSummary extends QuadVisitor.EmptyVisitor {
        
        final jq_Method method;
        final int nLocals, nRegisters;
        final ParamNode[] param_nodes;
        final GlobalNode my_global;
        final State[] start_states;
        final HashSet returned, thrown;
        boolean change;
        BasicBlock bb;
        State s;
        final HashSet methodCalls;
        final HashSet passedAsParameter;
        
        AnalysisSummary getSummary() {
            AnalysisSummary s = new AnalysisSummary(param_nodes, my_global, returned, thrown, passedAsParameter);
            // merge global nodes.
            if (my_global.accessPathEdges != null) {
                for (Iterator i=my_global.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof FieldNode)
                        GlobalNode.GLOBAL.addAccessPathEdge(f, (FieldNode)o);
                    else
                        GlobalNode.GLOBAL.addAccessPathEdges(f, (HashSet)o);
                }
            }
            return s;
        }
        
        void setLocal(int i, Node n) { s.registers[i] = n; }
        void setRegister(Register r, Node n) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            s.registers[i] = n;
        }
        void setRegister(Register r, Object n) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            if (n instanceof HashSet) n = ((HashSet)n).clone();
            s.registers[i] = n;
        }
        Object getRegister(Register r) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            return s.registers[i];
        }

        BuildMethodSummary(ControlFlowGraph cfg) {
            RegisterFactory rf = cfg.getRegisterFactory();
            this.nLocals = rf.getLocalSize(PrimordialClassLoader.getJavaLangObject());
            this.nRegisters = this.nLocals + rf.getStackSize(PrimordialClassLoader.getJavaLangObject());
            this.method = cfg.getMethod();
            this.start_states = new State[cfg.getNumberOfBasicBlocks()];
            this.methodCalls = new HashSet();
            this.passedAsParameter = new HashSet();
            this.s = this.start_states[0] = new State(this.nRegisters);
            jq_Type[] params = this.method.getParamTypes();
            this.param_nodes = new ParamNode[params.length];
            for (int i=0, j=0; i<params.length; ++i, ++j) {
                if (params[i].isReferenceType()) {
                    setLocal(i, param_nodes[i] = new ParamNode(method, j, (jq_Reference)params[i]));
                } else if (params[i].getReferenceSize() == 8) ++j;
            }
            this.my_global = new GlobalNode();
            this.returned = new HashSet(); this.thrown = new HashSet();
            
            if (TRACE_INTRA) out.println("Building summary for "+this.method);
            
            // iterate until convergence.
            List.BasicBlock rpo_list = cfg.reversePostOrder(cfg.entry());
            for (;;) {
                ListIterator.BasicBlock rpo = rpo_list.basicBlockIterator();
                this.change = false;
                while (rpo.hasNext()) {
                    this.bb = rpo.nextBasicBlock();
                    this.s = start_states[bb.getID()];
                    if (this.s == null) {
                        continue;
                    }
                    this.s = this.s.copy();
                    if (this.bb.isExceptionHandlerEntry()) {
                        java.util.Iterator i = cfg.getExceptionHandlersMatchingEntry(this.bb);
                        jq.assert(i.hasNext());
                        ExceptionHandler eh = (ExceptionHandler)i.next();
                        CaughtExceptionNode n = new CaughtExceptionNode(eh);
                        if (i.hasNext()) {
                            HashSet set = new HashSet(); set.add(n);
                            while (i.hasNext()) {
                                eh = (ExceptionHandler)i.next();
                                n = new CaughtExceptionNode(eh);
                                set.add(n);
                            }
                            s.registers[nLocals] = set;
                        } else {
                            s.registers[nLocals] = n;
                        }
                    }
                    if (TRACE_INTRA) {
                        out.println("State at beginning of "+this.bb+":");
                        this.s.dump(out);
                    }
                    this.bb.visitQuads(this);
                    ListIterator.BasicBlock succs = this.bb.getSuccessors().basicBlockIterator();
                    while (succs.hasNext()) {
                        BasicBlock succ = succs.nextBasicBlock();
                        mergeWith(succ);
                    }
                }
                if (!this.change) break;
            }
        }

        void mergeWith(BasicBlock succ) {
            if (this.start_states[succ.getID()] == null) {
                if (TRACE_INTRA) out.println(succ+" not yet visited.");
                this.start_states[succ.getID()] = this.s.copy();
                this.change = true;
            } else {
                if (TRACE_INTRA) out.println("merging out set of "+bb+" "+jq.hex8(this.s.hashCode())+" into in set of "+succ+" "+jq.hex8(this.start_states[succ.getID()].hashCode()));
                if (this.start_states[succ.getID()].merge(this.s)) {
                    if (TRACE_INTRA) out.println(succ+" in set changed");
                    this.change = true;
                }
            }
        }
        
        public static final boolean INSIDE_EDGES = false;
        
        void heapLoad(HashSet result, Node base, jq_Field f, FieldNode fn) {
            base.addAccessPathEdge(f, fn);
            if (INSIDE_EDGES)
                base.getEdges(f, result);
        }
        void heapLoad(Quad obj, Register dest_r, HashSet base_s, jq_Field f) {
            FieldNode fn = new FieldNode(f, obj);
            HashSet result = new HashSet();
            for (Iterator i=base_s.iterator(); i.hasNext(); ) {
                heapLoad(result, (Node)i.next(), f, fn);
            }
            if (result.isEmpty()) {
                setRegister(dest_r, fn);
            } else {
                result.add(fn);
                setRegister(dest_r, result);
            }
        }
        void heapLoad(Quad obj, Register dest_r, Node base_n, jq_Field f) {
            FieldNode fn = new FieldNode(f, obj);
            HashSet result = new HashSet();
            heapLoad(result, base_n, f, fn);
            if (result.isEmpty()) {
                setRegister(dest_r, fn);
            } else {
                result.add(fn);
                setRegister(dest_r, result);
            }
        }
        void heapLoad(Quad obj, Register dest_r, Register base_r, jq_Field f) {
            Object o = getRegister(base_r);
            if (o instanceof HashSet) {
                heapLoad(obj, dest_r, (HashSet)o, f);
            } else {
                heapLoad(obj, dest_r, (Node)o, f);
            }
        }
        
        void heapStore(Node base, Node src, jq_Field f) {
            base.addEdge(f, src);
        }
        void heapStore(Node base, HashSet src, jq_Field f) {
            base.addEdges(f, (HashSet)src.clone());
        }
        void heapStore(Register base_r, Node src_n, jq_Field f) {
            Object base = getRegister(base_r);
            if (base instanceof HashSet) {
                for (Iterator i = ((HashSet)base).iterator(); i.hasNext(); ) {
                    heapStore((Node)i.next(), src_n, f);
                }
            } else {
                heapStore((Node)base, src_n, f);
            }
        }
        void heapStore(Node base, Register src_r, jq_Field f) {
            Object src = getRegister(src_r);
            if (src instanceof Node) {
                heapStore(base, (Node)src, f);
            } else {
                heapStore(base, (HashSet)src, f);
            }
        }
        void heapStore(Register base_r, Register src_r, jq_Field f) {
            Object base = getRegister(base_r);
            Object src = getRegister(src_r);
            if (src instanceof Node) {
                heapStore(base_r, (Node)src, f);
                return;
            }
            HashSet src_h = (HashSet)src;
            if (base instanceof HashSet) {
                for (Iterator i = ((HashSet)base).iterator(); i.hasNext(); ) {
                    heapStore((Node)i.next(), src_h, f);
                }
            } else {
                heapStore((Node)base, src_h, f);
            }
        }

        void passParameter(Register r, MethodCall m, int p) {
            Object v = getRegister(r);
            if (TRACE_INTRA) out.println("Passing "+r+" to "+m+" param "+p+": "+v);
            if (v instanceof HashSet) {
                for (Iterator i = ((HashSet)v).iterator(); i.hasNext(); ) {
                    Node n = (Node)i.next();
                    n.recordPassedParameter(m, p);
                    passedAsParameter.add(n);
                }
            } else {
                Node n = (Node)v;
                n.recordPassedParameter(m, p);
                passedAsParameter.add(n);
            }
        }
        
        /** An array load instruction. */
        public void visitALoad(Quad obj) {
            if (obj.getOperator() instanceof Operator.ALoad.ALOAD_A) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register r = ALoad.getDest(obj).getRegister();
                Operand o = ALoad.getBase(obj);
                if (o instanceof RegisterOperand) {
                    Register b = ((RegisterOperand)o).getRegister();
                    heapLoad(obj, r, b, null);
                } else {
                    // base is not a register?!
                }
            }
        }
        /** An array store instruction. */
        public void visitAStore(Quad obj) {
            if (obj.getOperator() instanceof Operator.AStore.ASTORE_A) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Operand base = AStore.getBase(obj);
                Operand val = AStore.getValue(obj);
                if (base instanceof RegisterOperand) {
                    Register base_r = ((RegisterOperand)base).getRegister();
                    if (val instanceof RegisterOperand) {
                        Register src_r = ((RegisterOperand)val).getRegister();
                        heapStore(base_r, src_r, null);
                    } else {
                        jq.assert(val instanceof AConstOperand);
                        jq_Reference type = ((AConstOperand)val).getType();
                        ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                        heapStore(base_r, n, null);
                    }
                } else {
                    // base is not a register?!
                }
            }
        }
        /** A type cast check instruction. */
        public void visitCheckCast(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            Register dest_r = CheckCast.getDest(obj).getRegister();
            Operand src = CheckCast.getSrc(obj);
            // TODO: treat it like a move for now.
            if (src instanceof RegisterOperand) {
                Register src_r = ((RegisterOperand)src).getRegister();
                setRegister(dest_r, getRegister(src_r));
            } else {
                jq.assert(src instanceof AConstOperand);
                jq_Reference type = ((AConstOperand)src).getType();
                ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                setRegister(dest_r, n);
            }
        }
        /** A get instance field instruction. */
        public void visitGetfield(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Getfield.GETFIELD_A) ||
               (obj.getOperator() instanceof Operator.Getfield.GETFIELD_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register r = Getfield.getDest(obj).getRegister();
                Operand o = Getfield.getBase(obj);
                jq_Field f = Getfield.getField(obj).getField();
                if (o instanceof RegisterOperand) {
                    Register b = ((RegisterOperand)o).getRegister();
                    heapLoad(obj, r, b, f);
                } else {
                    // base is not a register?!
                }
            }
        }
        /** A get static field instruction. */
        public void visitGetstatic(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Getstatic.GETSTATIC_A) ||
               (obj.getOperator() instanceof Operator.Getstatic.GETSTATIC_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register r = Getstatic.getDest(obj).getRegister();
                jq_Field f = Getstatic.getField(obj).getField();
                heapLoad(obj, r, my_global, f);
            }
        }
        /** A type instance of instruction. */
        public void visitInstanceOf(Quad obj) {
            // skip for now.
        }
        /** An invoke instruction. */
        public void visitInvoke(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            jq_Method m = Invoke.getMethod(obj).getMethod();
            MethodCall mc = new MethodCall(m, obj);
            methodCalls.add(mc);
            jq_Type[] params = m.getParamTypes();
            ParamListOperand plo = Invoke.getParamList(obj);
            jq.assert(m == Allocator.HeapAllocator._multinewarray || params.length == plo.length());
            for (int i=0; i<params.length; ++i) {
                if (!params[i].isReferenceType()) continue;
                Register r = plo.get(i).getRegister();
                passParameter(r, mc, i);
            }
            if (m.getReturnType().isReferenceType()) {
                RegisterOperand dest = Invoke.getDest(obj);
                if (dest != null) {
                    Register dest_r = dest.getRegister();
                    ReturnValueNode n = new ReturnValueNode(mc);
                    setRegister(dest_r, n);
                }
            }
            // TODO: thrown exception node at all handlers.
        }
        /** A register move instruction. */
        public void visitMove(Quad obj) {
            if (obj.getOperator() instanceof Operator.Move.MOVE_A) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = Move.getDest(obj).getRegister();
                Operand src = Move.getSrc(obj);
                if (src instanceof RegisterOperand) {
                    Register src_r = ((RegisterOperand)src).getRegister();
                    setRegister(dest_r, getRegister(src_r));
                } else {
                    jq.assert(src instanceof AConstOperand);
                    jq_Reference type = ((AConstOperand)src).getType();
                    ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                    setRegister(dest_r, n);
                }
            }
        }
        /** An object allocation instruction. */
        public void visitNew(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            Register dest_r = New.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)New.getType(obj).getType();
            ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
            setRegister(dest_r, n);
        }
        /** An array allocation instruction. */
        public void visitNewArray(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            Register dest_r = NewArray.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)NewArray.getType(obj).getType();
            ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
            setRegister(dest_r, n);
        }
        /** A put instance field instruction. */
        public void visitPutfield(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A) ||
               (obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Operand base = Putfield.getBase(obj);
                Operand val = Putfield.getSrc(obj);
                jq_Field f = Putfield.getField(obj).getField();
                if (base instanceof RegisterOperand) {
                    Register base_r = ((RegisterOperand)base).getRegister();
                    if (val instanceof RegisterOperand) {
                        Register src_r = ((RegisterOperand)val).getRegister();
                        heapStore(base_r, src_r, f);
                    } else {
                        jq.assert(val instanceof AConstOperand);
                        jq_Reference type = ((AConstOperand)val).getType();
                        ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                        heapStore(base_r, n, f);
                    }
                } else {
                    // base is not a register?!
                }
            }
        }
        /** A put static field instruction. */
        public void visitPutstatic(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Putstatic.PUTSTATIC_A) ||
               (obj.getOperator() instanceof Operator.Putstatic.PUTSTATIC_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Operand val = Putstatic.getSrc(obj);
                jq_Field f = Putstatic.getField(obj).getField();
                if (val instanceof RegisterOperand) {
                    Register src_r = ((RegisterOperand)val).getRegister();
                    heapStore(my_global, src_r, f);
                } else {
                    jq.assert(val instanceof AConstOperand);
                    jq_Reference type = ((AConstOperand)val).getType();
                    ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                    heapStore(my_global, n, f);
                }
            }
        }
        
        static void addToSet(HashSet s, Object o) {
            if (o instanceof HashSet) s.addAll((HashSet)o);
            else s.add(o);
        }
        
        public void visitReturn(Quad obj) {
            Operand src = Return.getSrc(obj);
            HashSet r;
            if (obj.getOperator() == Return.RETURN_A.INSTANCE) r = returned;
            else if (obj.getOperator() == Return.THROW_A.INSTANCE) r = thrown;
            else return;
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            if (src instanceof RegisterOperand) {
                Register src_r = ((RegisterOperand)src).getRegister();
                addToSet(r, getRegister(src_r));
            } else {
                jq.assert(src instanceof AConstOperand);
                jq_Reference type = ((AConstOperand)src).getType();
                ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                r.add(n);
            }
        }
            
        public void visitSpecial(Quad obj) {
            if (obj.getOperator() == Special.GET_THREAD_BLOCK.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                ConcreteTypeNode n = new ConcreteTypeNode(Scheduler.jq_Thread._class, obj);
                setRegister(dest_r, n);
            } else if (obj.getOperator() == Special.GET_TYPE_OF.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                UnknownTypeNode n = new UnknownTypeNode(Clazz.jq_Reference._class, obj);
                setRegister(dest_r, n);
            }
        }
        public void visitUnary(Quad obj) {
            if (obj.getOperator() == Unary.INT_2OBJECT.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = Unary.getDest(obj).getRegister();
                UnknownTypeNode n = new UnknownTypeNode(PrimordialClassLoader.getJavaLangObject(), obj);
                setRegister(dest_r, n);
            }
        }
        public void visitExceptionThrower(Quad obj) {
            ListIterator.jq_Class xs = obj.getThrownExceptions().classIterator();
            while (xs.hasNext()) {
                jq_Class x = xs.nextClass();
                ListIterator.ExceptionHandler eh = bb.getExceptionHandlers().exceptionHandlerIterator();
                while (eh.hasNext()) {
                    ExceptionHandler h = eh.nextExceptionHandler();
                    if (h.mayCatch(x))
                        this.mergeWith(h.getEntry());
                    if (h.mustCatch(x))
                        break;
                }
            }
        }
        
    }
    
    /*
    public static class CallingContextAndNode {
        CallingContext cc; Node n;
        CallingContextAndNode(CallingContext cc, Node n) { this.cc = cc; this.n = n; }
        public boolean equals(CallingContextAndNode that) {
            return this.cc == that.cc && this.n == that.n;
        }
        public boolean equals(Object o) {
            if (o instanceof CallingContextAndNode) return equals((CallingContextAndNode)o);
            return false;
        }
        public int hashCode() { return cc.hashCode() ^ n.hashCode(); }
    }
     */

    // Solves queries using a worklist algorithm.
    public static class QuerySolver {
        HashMap all_queries;
        HashSet open_queries;
        LinkedList worklist;

        QuerySolver() {
            this.all_queries = new HashMap();
            this.open_queries = new HashSet();
            this.worklist = new LinkedList();
        }
        
        void doQuery(Query q) {
            this.all_queries.clear(); Query.current_id = 0;
            if (TRACE_INTER) out.println(" ... Starting query solver ...");
            worklist.add(q); open_queries.add(q);
            performLoop();
        }
        
        void performLoop() {
            while (!worklist.isEmpty()) {
                Object o = worklist.removeFirst();
                open_queries.remove(o);
                if (o instanceof Query) {
                    Query q = (Query)o;
                    if (TRACE_INTER) out.println("-=> Performing query "+q.toString_full());
                    boolean b = q.performQuery();
                    if (b) {
                        if (TRACE_INTER) out.println("-=> Updating successors of "+q);
                        q.updateSuccessors();
                    }
                    continue;
                }
                if (o instanceof Dependence) {
                    Dependence d = (Dependence)o;
                    if (TRACE_INTER) out.println("-=> Updating dependence "+d);
                    boolean b = d.from.propagateResult(d.to);
                    if (b) {
                        if (TRACE_INTER) out.println("-=> Updating successors of "+d.to);
                        d.to.updateSuccessors();
                    }
                    continue;
                }
            }
            jq.assert(open_queries.isEmpty());
        }
        
        Query addQuery(DependentQuery dq, Query q) {
            Query q2 = (Query)all_queries.get(q);
            if (q2 == null) {
                q2 = q;
                if (TRACE_INTER) out.println("-=> New query "+q2.id+": "+q2.toString_full());
                all_queries.put(q2, q2);
                open_queries.add(q2); worklist.add(q2);
            } else {
                if (TRACE_INTER) out.println("-=> Reusing query "+q2.id);
            }
            Dependence d = new Dependence(q2, dq);
            if (open_queries.contains(d)) {
                if (TRACE_INTER) out.println("-=> Query "+q2.id+" already open");
                return q2;
            }
            open_queries.add(d); worklist.add(d);
            return q2;
        }
        
        public static final QuerySolver GLOBAL = new QuerySolver();
    }
    
    // Records a dependence edge.  When the result of the 'from' query changes, the
    // 'to' query must be recalculated.
    public static class Dependence {
        Query from;
        DependentQuery to;
        
        Dependence(Query from, DependentQuery to) {
            this.from = from; this.to = to;
        }
        
        public boolean equals(Dependence that) {
            return this.from == that.from && this.to == that.to;
        }
        public boolean equals(Object o) {
            if (o instanceof Dependence) return equals((Dependence)o);
            return false;
        }
        public int hashCode() { return from.hashCode() ^ to.hashCode(); }
        public String toString() { return "q"+from.id+"->q"+to.id; }
    }
    
    // The base type for all queries.
    public abstract static class Query {
        abstract boolean performQuery();
        
        static int current_id;
        final int id;
        
        Query() { this.id = ++current_id; }
        
        // Set of queries that depend on our result.
        HashSet dependentQueries;
        public final HashSet getResult(DependentQuery who) {
            jq.assert(who == null || dependentQueries == null || dependentQueries.contains(who));
            return _getResult();
        }
        abstract HashSet _getResult();
        // Add a query that depends on our result.
        final void addSuccessor(DependentQuery who) {
            if (dependentQueries == null) dependentQueries = new HashSet();
            dependentQueries.add(who);
        }
        // Add the queries that depend on our result to the worklist.
        final void updateSuccessors() {
            if (dependentQueries == null) return;
            for (Iterator i=dependentQueries.iterator(); i.hasNext(); ) {
                DependentQuery q = (DependentQuery)i.next();
                QuerySolver.GLOBAL.addQuery(q, this);
            }
        }
        abstract boolean propagateResult(DependentQuery q);
        
        public final String toString() { return "q"+id; }
        public abstract String toString_full() ;
    }

    public abstract static class ConstantQuery extends Query {
        final boolean performQuery() { return true; }
    }

    public static class ConcreteTypeQuery extends ConstantQuery implements TypesQuery {
        private HashSet result = new HashSet();
        ConcreteTypeQuery(jq_Reference r) {
            this.result.add(r);
        }
        HashSet _getResult() { return result; }
        boolean propagateResult(DependentQuery q) { return q.receiveResult((TypesQuery)this); }
        public String toString_full() { return "q"+id+": Concrete "+result; }
    }
    public static class UnknownTypeQuery extends ConstantQuery implements TypesQuery {
        private HashSet result;
        UnknownTypeQuery(HashSet r) {
            this.result = r;
        }
        HashSet _getResult() { return result; }
        boolean propagateResult(DependentQuery q) { return q.receiveResult((TypesQuery)this); }
        public String toString_full() { return "q"+id+": Unknown "+result; }
    }
    public static class ParamNodeQuery extends ConstantQuery implements NodesQuery {
        private HashSet result = new HashSet();
        ParamNodeQuery(CallingContext c, ParamNode pn) {
            PassedParameter pp = new PassedParameter(c.call_site, pn.n);
            HashSet nodes = new HashSet();
            c.getParentContext().getNodes(pp, nodes);
            for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                CallingContextAndNode ccn = new CallingContextAndNode(c.getParentContext(), n);
                result.add(ccn);
            }
        }
        HashSet _getResult() { return result; }
        boolean propagateResult(DependentQuery q) { return q.receiveResult((NodesQuery)this); }
        public String toString_full() { return "q"+id+": Param node "+result; }
    }
    
    public abstract static class DependentQuery extends Query {
        HashSet result = new HashSet();
        HashSet _getResult() { return result; }

        void spawnQuery(Query q) {
            q = QuerySolver.GLOBAL.addQuery(this, q);
            if (TRACE_INTER) out.println("q"+id+": spawning query "+q);
            q.addSuccessor(this);
        }
        void spawnSharedQuery(DependentQuery q) {
            q = (DependentQuery)QuerySolver.GLOBAL.addQuery(this, q);
            if (TRACE_INTER) out.println("q"+id+": spawning shared query "+q);
            if (q.result == this.result) {
                q.dependentQueries = this.dependentQueries;
            } else {
                if (TRACE_INTER) out.println("q"+id+": not shared with "+q);
                q.addSuccessor(this);
            }
        }
        
        abstract boolean receiveResult(NodesQuery dq);
        abstract boolean receiveResult(TypesQuery ntq);
        abstract boolean receiveResult(CallTargetsQuery ctq);
        
        // can spawn a NodesQuery.
        final boolean spawnTypesQuery(AnalysisSummary as, Node n) {
            if (n instanceof ConcreteTypeNode) {
                jq_Reference r = ((ConcreteTypeNode)n).type;
                ConcreteTypeQuery ctq = new ConcreteTypeQuery(r);
                ctq.performQuery();
                return this.receiveResult(ctq);
            }
            if (n instanceof UnknownTypeNode) {
                jq_Reference r = ((UnknownTypeNode)n).type;
                HashSet s = new HashSet();
                s.add(r);
                if (r instanceof jq_Class)
                    TopCallingContext.helper(s, (jq_Class)r);
                UnknownTypeQuery utq = new UnknownTypeQuery(s);
                utq.performQuery();
                return this.receiveResult(utq);
            }
            if (n instanceof ParamNode) {
                ParamNodeQuery pnq = new ParamNodeQuery(c, (ParamNode)n);
                pnq.performQuery();
                return this.receiveResult(pnq);
            }
            if (n instanceof ReturnValueNode) {
                ReturnValueNodeQuery rvnq = new ReturnValueNodeQuery(c, (ReturnValueNode)n);
                spawnQuery(rvnq);
                return false;
            }
            if (n instanceof ThrownExceptionNode) {
                ThrownExceptionNodeQuery rvnq = new ThrownExceptionNodeQuery(c, (ThrownExceptionNode)n);
                spawnQuery(rvnq);
                return false;
            }
            if (n instanceof CaughtExceptionNode) {
                // TODO.
                return false;
            }
            if (n instanceof FieldNode) {
                HashSet result = new HashSet();
                NodeAccessPathQuery nq = new NodeAccessPathQuery(c, n, null, result);
                spawnQuery(nq);
                return false;
            }
            jq.UNREACHABLE(n.toString());
            return false;
        }
        
    }

    public static interface NodesQuery {
        HashSet getResult(DependentQuery q);
    }
    public static interface TypesQuery {
        HashSet getResult(DependentQuery q);
    }
    
    // Takes <CallingContext, Node, AccessPath> triple.
    // Returns set of <CallingContext, Node> pairs.
    public static class NodeAccessPathQuery extends DependentQuery implements NodesQuery {
        CallingContext c;
        Node n;
        AccessPath ap;

        NodeAccessPathQuery(CallingContext c, Node n, AccessPath ap, HashSet result) {
            this.c = c; this.n = n; this.ap = ap; this.result = result;
        }
        
        public String toString_full() { return "q"+id+": Node "+n+" path "+ap+" in context "+c; }
        
        public int hashCode() {
            return c.hashCode()^n.hashCode()^((ap==null)?0:ap.hashCode());
        }
        public boolean equals(NodeAccessPathQuery that) {
            if (this == that) return true;
            if (c != that.c) return false;
            if (n != that.n) return false;
            if (ap == that.ap) {
                return true;
            }
            if (ap == null) return false;
            if (that.ap == null) return false;
            if (!ap.equals(that.ap)) return false;
            return true;
        }
        public boolean equals(Object o) {
            if (o instanceof NodeAccessPathQuery) return equals((NodeAccessPathQuery)o);
            return false;
        }
        
        boolean performQuery() {
            if (n instanceof FieldNode) {
                // find predecessor read edges.
                FieldNode fn = (FieldNode)n;
                AccessPath ap2 = AccessPath.create(fn.f, fn, ap);
                for (Iterator i=fn.field_predecessors.iterator(); i.hasNext(); ) {
                    NodeAccessPathQuery dq = new NodeAccessPathQuery(c, (Node)i.next(), ap2, result);
                    spawnSharedQuery(dq);
                }
                if (ap == null) {
                    // don't add field nodes.
                    return false;
                }
            }
            
            if (ap == null) {
                boolean b = result.add(new CallingContextAndNode(c, n));
                return b;
            }
            
            if (n instanceof UnknownTypeNode) {
                boolean b = false;
                for (Iterator i = ap.findLast(); i.hasNext(); ) {
                    AccessPath p = (AccessPath)i.next();
                    Node n = p._n;
                    UnknownTypeNode utn = new UnknownTypeNode(n.getDeclaredType());
                    if (result.add(new CallingContextAndNode(c, utn))) b = true;
                }
                return b;
            }
            
            HashSet r = new HashSet();
            n.getEdges(ap.getField(), r);
            n.getAccessPathEdges(ap.getField(), r);
            
            for (Iterator j=ap.next(); j.hasNext(); ) {
                AccessPath ap2 = (AccessPath)j.next();
                for (Iterator i=r.iterator(); i.hasNext(); ) {
                    Node n2 = (Node)i.next();
                    NodeAccessPathQuery dq = new NodeAccessPathQuery(c, n2, ap2, result);
                    spawnSharedQuery(dq);
                }
            }
            
            if (n.predecessors != null) {
                for (Iterator i=n.predecessors.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    AccessPath ap2 = AccessPath.create(ap, f, n);
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        NodeAccessPathQuery dq = new NodeAccessPathQuery(c, (Node)o, ap2, result);
                        spawnSharedQuery(dq);
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            NodeAccessPathQuery dq = new NodeAccessPathQuery(c, (Node)j.next(), ap2, result);
                            spawnSharedQuery(dq);
                        }
                    }
                }
            }
            
            if (n.passedParameters != null) {
                ctq2pp = new HashMap();
                for (Iterator i=n.passedParameters.iterator(); i.hasNext(); ) {
                    PassedParameter pp = (PassedParameter)i.next();
                    CallTargetsQuery ctq = new CallTargetsQuery(c, pp.m.q);
                    ctq2pp.put(ctq, pp);
                    spawnQuery(ctq);
                }
            }
            
            if (n instanceof ParamNode) {
                PassedParameter pp = new PassedParameter(c.call_site, ((ParamNode)n).n);
                HashSet nodes = new HashSet();
                CallingContext c2 = c.getParentContext();
                c2.getNodes(pp, nodes);
                for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                    NodeAccessPathQuery dq = new NodeAccessPathQuery(c2, (Node)i.next(), ap, result);
                    spawnSharedQuery(dq);
                }
            } else if (n instanceof ReturnValueNode) {
                
            } else if (n instanceof ThrownExceptionNode) {
            }
            return false;
        }

        boolean propagateResult(DependentQuery q) { return q.receiveResult((NodesQuery)this); }
        
        boolean receiveResult(NodesQuery dq) {
            if (this.result == ((DependentQuery)dq).result) {
                return false;
            }
            return this.result.addAll(dq.getResult(this));
        }
        
        HashMap ctq2pp;
        
        boolean receiveResult(CallTargetsQuery ctq) {
            PassedParameter pp = (PassedParameter)ctq2pp.get(ctq);
            for (Iterator i = ctq.getResult(this).iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                m.getDeclaringClass().load();
                //if (!canWriteToAccessPath(m, ap)) continue;
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                if (true) {
                    // instantiate.
                    AnalysisSummary.instantiate(c.s, pp.m, s_callee);
                } else {
                    CallingContext c2 = c.getChildCallingContext(pp.m, m, s_callee);
                    ParamNode pn = s_callee.params[pp.paramNum];
                    NodeAccessPathQuery dq = new NodeAccessPathQuery(c2, pn, ap, result);
                    spawnSharedQuery(dq);
                }
            }
            return false;
        }
        
        boolean receiveResult(TypesQuery ntq) { jq.UNREACHABLE(); return false; }
    }
    
    // Takes <CallingContext, Quad> pair.
    // Returns a set of jq_Methods.
    public static class CallTargetsQuery extends DependentQuery {
        CallingContext c;
        Quad q;
        
        CallTargetsQuery(CallingContext c, Quad q) { this.c = c; this.q = q; }
        
        public String toString_full() { return "q"+id+": Call targets "+q+" in context "+c; }
        
        public int hashCode() { return c.hashCode() ^ q.hashCode(); }
        public boolean equals(CallTargetsQuery that) {
            if (c != that.c) return false;
            if (q != that.q) return false;
            return true;
        }
        public boolean equals(Object o) {
            if (o instanceof CallTargetsQuery) return equals((CallTargetsQuery)o);
            return false;
        }
        
        boolean performQuery() {
            jq_Method target = Invoke.getMethod(q).getMethod();
            if (!((Invoke)q.getOperator()).isVirtual()) {
                boolean b = result.add(target);
                if (TRACE_INTER) out.println("q"+id+": Simple, non-virtual call: "+target);
                return b;
            }
            // find the set of nodes for the 'this' pointer.
            MethodCall mc = new MethodCall(target, q);
            PassedParameter pp = new PassedParameter(mc, 0);
            HashSet nodes = new HashSet();
            if (TRACE_INTER) out.println("q"+id+": Getting nodes for "+pp);
            c.getNodes(pp, nodes);
            // find the set of types for those nodes.
            NodesTypesQuery ntq = new NodesTypesQuery(c, nodes);
            spawnQuery(ntq);
            return false;
        }
        
        boolean receiveResult(TypesQuery ctq) {
            boolean change = false;
            HashSet type_result = ((Query)ctq).getResult(this);
            // look up the target methods and add them.
            jq_Method target = Invoke.getMethod(q).getMethod();
            for (Iterator i = type_result.iterator(); i.hasNext(); ) {
                jq_Reference r = (jq_Reference)i.next();
                r.load(); r.verify(); r.prepare();
                jq_Method resolved_target = r.getVirtualMethod(target.getNameAndDesc());
                if (resolved_target != null) {
                    if (resolved_target.getBytecode() != null) {
                        if (result.add(resolved_target)) change = true;
                    }
                } else {
                    if (TRACE_INTER) out.println("Invalid call! "+r+" on target "+target);
                }
            }
            return change;
        }
        
        boolean propagateResult(DependentQuery q) { return q.receiveResult(this); }
        
        boolean receiveResult(NodesQuery ntq) { jq.UNREACHABLE(); return false; }
        boolean receiveResult(CallTargetsQuery ntq) { jq.UNREACHABLE(); return false; }
    }
    
    public static class NodesTypesQuery extends DependentQuery implements TypesQuery {
        CallingContext c;
        HashSet nodes;
        
        NodesTypesQuery(CallingContext c, HashSet nodes) {
            this.c = c; this.nodes = nodes;
        }
        
        public String toString_full() { return "q"+id+": Types of "+nodes+" in context "+c; }
        
        boolean performQuery() {
            return doNextQuery();
        }
        
        boolean doNextQuery() {
            Iterator i = nodes.iterator();
            if (i.hasNext()) {
                Node n = (Node)i.next();
                nodes.remove(n);
                return spawnTypesQuery(c, n);
            }
            return false;
        }
        
        boolean receiveResult(NodesQuery nq) {
            boolean change = false;
            for (Iterator i=((Query)nq).getResult(this).iterator(); i.hasNext(); ) {
                CallingContextAndNode ccn = (CallingContextAndNode)i.next();
                if (spawnTypesQuery(ccn.cc, ccn.n)) change = true;
            }
            return change;
        }
        boolean receiveResult(TypesQuery ctq) {
            doNextQuery();
            return result.addAll(ctq.getResult(this));
        }
        
        boolean propagateResult(DependentQuery q) { return q.receiveResult((TypesQuery)this); }
        boolean receiveResult(CallTargetsQuery ntq) { jq.UNREACHABLE(); return false; }
    }
    
    public static class ReturnValueNodeQuery extends DependentQuery implements NodesQuery {
        CallingContext c;
        ReturnValueNode n;
        
        ReturnValueNodeQuery(CallingContext c) {
            this.c = c; this.n = null;
        }
        ReturnValueNodeQuery(CallingContext c, ReturnValueNode n) {
            this.c = c; this.n = n;
        }
        
        public String toString_full() { return "q"+id+": Return value of "+n+" in context "+c; }
        
        public int hashCode() { return c.hashCode() ^ n.hashCode(); }
        public boolean equals(ReturnValueNodeQuery that) {
            if (c != that.c) return false;
            if (n != that.n) return false;
            return true;
        }
        public boolean equals(Object o) {
            if (o instanceof ReturnValueNodeQuery) return equals((ReturnValueNodeQuery)o);
            return false;
        }
        
        boolean performQuery() {
            // n is the return value of a method call.
            // we need to analyze that method call to find the return types.
            MethodCall that_mc = n.m;
            CallTargetsQuery ctq = new CallTargetsQuery(c, that_mc.q);
            spawnQuery(ctq);
            return false;
        }
        
        boolean receiveResult(CallTargetsQuery ctq) {
            boolean change = false;
            for (Iterator i = ctq.getResult(this).iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method)i.next();
                ControlFlowGraph cfg = CodeCache.getCode(m);
                AnalysisSummary s_callee = getSummary(cfg);
                HashSet s = getSet(s_callee);
                if (s == null) continue;
                MethodCall that_mc = ((ReturnValueNode)n).m;
                if (true) {
                    // instantiate.
                    AnalysisSummary.instantiate(c.s, that_mc, s_callee);
                } else {
                    CallingContext c2 = c.getChildCallingContext(that_mc, m, s_callee);
                    for (Iterator j = s.iterator(); j.hasNext(); ) {
                        if (result.add(new CallingContextAndNode(c2, (Node)j.next())))
                            change = true;
                    }
                }
            }
            return change;
        }
        
        HashSet getSet(AnalysisSummary s_callee) { return s_callee.returned; }
        
        boolean receiveResult(NodesQuery nq) {
            return result.addAll(nq.getResult(this));
        }
        boolean propagateResult(DependentQuery q) { return q.receiveResult((NodesQuery)this); }
        boolean receiveResult(TypesQuery ctq) { jq.UNREACHABLE(); return false; }
    }
    
    public static class ThrownExceptionNodeQuery extends ReturnValueNodeQuery {
        ThrownExceptionNode n;
        ThrownExceptionNodeQuery(CallingContext c, ThrownExceptionNode n) {
             super(c); this.n = n;
        }
        
        public int hashCode() { return c.hashCode() ^ n.hashCode(); }
        public boolean equals(ThrownExceptionNodeQuery that) {
            if (c != that.c) return false;
            if (n != that.n) return false;
            return true;
        }
        public boolean equals(Object o) {
            if (o instanceof ThrownExceptionNodeQuery) return equals((ThrownExceptionNodeQuery)o);
            return false;
        }
        
        public String toString_full() { return "q"+id+": Thrown exception of "+n+" in context "+c; }
        boolean performQuery() {
            // n is the return value of a method call.
            // we need to analyze that method call to find the return types.
            MethodCall that_mc = n.m;
            CallTargetsQuery ctq = new CallTargetsQuery(c, that_mc.q);
            spawnQuery(ctq);
            return false;
        }
        HashSet getSet(AnalysisSummary s_callee) { return s_callee.thrown; }
    }
    
    /** Intra-method summary graph. */
    public static class AnalysisSummary {
        /** The parameter nodes. */
        final ParamNode[] params;
        /** All nodes in the summary graph. */
        final HashMap nodes;
        /** The returned nodes. */
        final HashSet returned;
        /** The thrown nodes. */
        final HashSet thrown;
        
        final HashMap instantiated_callees;
        
        public static boolean addToMultiMap(HashMap mm, Object from, Object to) {
            HashSet s = (HashSet)mm.get(from);
            if (s == null) {
                mm.put(from, s = new HashSet());
            }
            return s.add(to);
        }
        
        public static boolean addToMultiMap(HashMap mm, Object from, HashSet to) {
            HashSet s = (HashSet)mm.get(from);
            if (s == null) {
                mm.put(from, s = new HashSet());
            }
            return s.addAll(to);
        }
        
        public void getNodesThatCall(PassedParameter pp, HashSet result) {
            for (Iterator i = this.nodeIterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                if ((n.passedParameters != null) && n.passedParameters.contains(pp))
                    result.add(n);
            }
        }
        
        public static void addEdges(HashSet s, jq_Field f, Node n) {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Node a = (Node)i.next();
                a.addEdge(f, n);
            }
        }
        
        public static HashSet get_mapping(HashMap callee_to_caller, Node callee_n) {
            HashSet s = (HashSet)callee_to_caller.get(callee_n);
            if (s != null) return s;
            s = new HashSet(); s.add(callee_n);
            return s;
        }
        
        public AnalysisSummary copy() {
            HashMap m = new HashMap();
            for (Iterator i=nodeIterator(); i.hasNext(); ) {
                Node a = (Node)i.next();
                Node b = a.copy();
                m.put(a, b);
            }
            for (Iterator i=nodeIterator(); i.hasNext(); ) {
                Node a = (Node)i.next();
                Node b = (Node)m.get(a);
                b.update(m);
            }
            HashSet returned = new HashSet();
            for (Iterator i=this.returned.iterator(); i.hasNext(); ) {
                returned.add(m.get(i.next()));
            }
            HashSet thrown = new HashSet();
            for (Iterator i=this.thrown.iterator(); i.hasNext(); ) {
                thrown.add(m.get(i.next()));
            }
            ParamNode[] params = new ParamNode[this.params.length];
            for (int i=0; i<params.length; ++i) {
                if (this.params[i] == null) continue;
                params[i] = (ParamNode)m.get(this.params[i]);
            }
            HashMap nodes = new HashMap();
            for (Iterator i=m.entrySet().iterator(); i.hasNext(); ) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                nodes.put(e.getValue(), e.getValue());
            }
            return new AnalysisSummary(params, returned, thrown, nodes);
        }
        
        public void unifyAccessPaths() {
            HashSet roots = new HashSet();
            for (int i=0; i<params.length; ++i) {
                if (params[i] == null) continue;
                roots.add(params[i]);
            }
            roots.addAll(returned); roots.addAll(thrown);
            unifyAccessPaths(roots);
        }
        
        public void unifyAccessPaths(HashSet roots) {
            LinkedList worklist = new LinkedList();
            for (Iterator i=roots.iterator(); i.hasNext(); ) {
                worklist.add(i.next());
            }
            while (worklist.isEmpty()) {
                Node n = (Node)worklist.removeFirst();
                unifyAccessPathEdges(n);
                if (n.accessPathEdges != null) {
                    for (Iterator i=n.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                        java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                        FieldNode n2 = (FieldNode)e.getValue();
                        if (roots.contains(n2)) continue;
                        worklist.add(n2); roots.add(n2);
                    }
                }
            }
        }
        
        public void unifyAccessPathEdges(Node n) {
            if (n.accessPathEdges != null) {
                for (Iterator i=n.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    FieldNode n2;
                    if (o instanceof HashSet) {
                        Iterator j=((HashSet)o).iterator();
                        n2 = (FieldNode)j.next();
                        while (j.hasNext()) {
                            FieldNode n3 = (FieldNode)j.next();
                            n2.unify(n3);
                            if (returned.contains(n3)) {
                                returned.remove(n3); returned.add(n2);
                            }
                            if (thrown.contains(n3)) {
                                thrown.remove(n3); thrown.add(n2);
                            }
                            nodes.remove(n3);
                        }
                        e.setValue(n2);
                    } else {
                        n2 = (FieldNode)o;
                    }
                }
            }
        }
        
        public static void instantiate(AnalysisSummary caller, MethodCall mc, AnalysisSummary callee) {
            callee = callee.copy();
            //System.out.println("Instantiating "+callee+" into "+caller);
            HashMap callee_to_caller = new HashMap();
            // initialize map with parameters.
            for (int i=0; i<callee.params.length; ++i) {
                ParamNode pn = callee.params[i];
                if (pn == null) continue;
                PassedParameter pp = new PassedParameter(mc, i);
                HashSet s = new HashSet();
                caller.getNodesThatCall(pp, s);
                callee_to_caller.put(pn, s);
            }
            for (int ii=0; ii<callee.params.length; ++ii) {
                ParamNode pn = callee.params[ii];
                if (pn == null) continue;
                HashSet s = (HashSet)callee_to_caller.get(pn);
                pn.replaceBy(s);
                if (callee.returned.contains(pn)) {
                    callee.returned.remove(pn); callee.returned.add(s);
                }
            }
            ReturnValueNode rvn = new ReturnValueNode(mc);
            rvn = (ReturnValueNode)caller.nodes.get(rvn);
            rvn.replaceBy(callee.returned);
            caller.unifyAccessPaths(callee.returned);
            for (int ii=0; ii<callee.params.length; ++ii) {
                ParamNode pn = callee.params[ii];
                if (pn == null) continue;
                HashSet s = (HashSet)callee_to_caller.get(pn);
                caller.unifyAccessPaths(s);
            }
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            for (int i=0; i<params.length; ++i) {
                if (params[i] == null) continue;
                sb.append(params[i].toString_long());
                sb.append('\n');
            }
            if (returned != null && !returned.isEmpty()) {
                sb.append("Returned: ");
                sb.append(returned);
                sb.append('\n');
            }
            if (thrown != null && !thrown.isEmpty()) {
                sb.append("Thrown: ");
                sb.append(thrown);
                sb.append('\n');
            }
            sb.append("All nodes:\n");
            for (Iterator i=nodes.keySet().iterator(); i.hasNext(); ) {
                sb.append(i.next());
                sb.append('\n');
            }
            return sb.toString();
        }
        
        AnalysisSummary(ParamNode[] params, HashSet returned, HashSet thrown, HashMap nodes) {
            this.params = params;
            this.returned = returned;
            this.thrown = thrown;
            this.instantiated_callees = new HashMap();
            this.nodes = nodes;
        }
        AnalysisSummary(ParamNode[] param_nodes, GlobalNode my_global, HashSet returned, HashSet thrown, HashSet passedAsParameters) {
            this.params = param_nodes;
            this.returned = returned;
            this.thrown = thrown;
            this.instantiated_callees = new HashMap();
            this.nodes = new HashMap();
            // build useful node set
            HashSet visited = new HashSet();
            for (int i=0; i<params.length; ++i) {
                if (params[i] == null) continue;
                addAsUseful(visited, params[i]);
            }
            for (Iterator i=returned.iterator(); i.hasNext(); ) {
                addAsUseful(visited, (Node)i.next());
            }
            for (Iterator i=thrown.iterator(); i.hasNext(); ) {
                addAsUseful(visited, (Node)i.next());
            }
            for (Iterator i=passedAsParameters.iterator(); i.hasNext(); ) {
                addAsUseful(visited, (Node)i.next());
            }
            if (my_global.accessPathEdges != null) {
                for (Iterator i=my_global.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        addIfUseful(visited, (Node)o);
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            addIfUseful(visited, (Node)j.next());
                        }
                    }
                }
            }
            unifyAccessPaths();
        }
        boolean addIfUseful(HashSet visited, Node n) {
            if (visited.contains(n)) return nodes.containsKey(n);
            visited.add(n);
            boolean useful = false;
            if (n.addedEdges != null) {
                useful = true;
                for (Iterator i=n.addedEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        addAsUseful(visited, (Node)o);
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            addAsUseful(visited, (Node)j.next());
                        }
                    }
                }
            }
            if (n.accessPathEdges != null) {
                for (Iterator i=n.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        if (addIfUseful(visited, (Node)o)) {
                            useful = true;
                        } else {
                            i.remove();
                        }
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            if (addIfUseful(visited, (Node)j.next())) {
                                useful = true;
                            } else {
                                j.remove();
                            }
                        }
                    }
                }
            }
            if (n.passedParameters != null) useful = true;
            if (useful)
                this.nodes.put(n, n);
            return useful;
        }
        void addAsUseful(HashSet visited, Node n) {
            if (visited.contains(n)) return;
            visited.add(n); this.nodes.put(n, n);
            if (n.addedEdges != null) {
                for (Iterator i=n.addedEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        addAsUseful(visited, (Node)o);
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            addAsUseful(visited, (Node)j.next());
                        }
                    }
                }
            }
            if (n.accessPathEdges != null) {
                for (Iterator i=n.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        if (!addIfUseful(visited, (Node)o)) {
                            i.remove();
                        }
                    } else {
                        for (Iterator j=((HashSet)o).iterator(); j.hasNext(); ) {
                            Node j_n = (Node)j.next();
                            if (!addIfUseful(visited, j_n)) {
                                j.remove();
                            }
                        }
                    }
                }
            }
        }
        
        Iterator nodeIterator() { return nodes.keySet().iterator(); }
    }
    
}

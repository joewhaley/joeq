/*
 * SparsePointerAnalysis.java
 *
 * Created on April 17, 2002, 5:57 PM
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
public class SparsePointerAnalysis {

    /** Creates new SparsePointerAnalysis */
    public SparsePointerAnalysis() {
    }

    public static abstract class NodesQuery {
        protected HashSet/*Node*/ constants;
    }
    
    public static final class ConstantNodesQuery extends NodesQuery {
        public ConstantNodesQuery(HashSet r) { this.constants = r; }
    }
    
    public static abstract class ComposedNodesQuery extends NodesQuery {
        protected LinkedList/*NodesQuery*/ subqueries;
        
        public void addResult(ComposedNodesQuery cnq) {
            this.constants.addAll(cnq.constants);
            this.subqueries.addAll(cnq.subqueries);
        }
        
    }
    
    public static abstract class DependentNodesQuery extends ComposedNodesQuery {
        // maps from a query to the queries that consume the result.
        protected HashMap/*<Query, HashSet<Query>>*/ queryToConsumers;
    }
    
    public static class NonVirtualCallTargetsQuery extends CallTargetsQuery {
        private jq_Method target;
        NonVirtualCallTargetsQuery(jq_Method m) { this.target = m; }
    }
    
    public static class VirtualCallTargetsQuery extends CallTargetsQuery {
        HashSet receiverNodes;
        VirtualCallTargetsQuery(HashSet nodes) { this.receiverNodes = nodes; }
    }
    
    public static abstract class CallTargetsQuery {
        public CallTargetsQuery create(AnalysisSummary s, Quad q) {
            jq_Method target = Invoke.getMethod(q).getMethod();
            if (!((Invoke)q.getOperator()).isVirtual()) {
                return new NonVirtualCallTargetsQuery(target);
            }
            // find the set of nodes for the 'this' pointer.
            MethodCall mc = new MethodCall(target, q);
            PassedParameter pp = new PassedParameter(mc, 0);
            HashSet nodes = new HashSet();
            s.getNodes(pp, nodes);
            return new VirtualCallTargetsQuery(s, nodes);
        }
        
        private CallTargetsQuery(Quad q) {
        }
        public void doIt() {
            // find the set of types for those nodes.
            NodesTypesQuery ntq = new NodesTypesQuery(summary, nodes);
            spawnQuery(ntq);
        }
        
        public void receiveResult(TypesQueryResult types, Node n) {
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
        }
    }
    
    public static class SimplifyNodesQuery extends ComposedNodesQuery {
        
        /** The summary that contains the set of nodes in question. */
        private AnalysisSummary summary;
        /** The set of nodes in question. */
        private HashSet setOfNodes;
        
        public SimplifyNodesQuery create(AnalysisSummary s, HashSet s_nodes) {
            return new SimplifyNodesQuery(s, s_nodes);
        }
        
        public SimplifyNodesQuery createReturned(AnalysisSummary callee, boolean which) {
            HashSet s_nodes = which?callee.returned:callee.thrown;
            return new SimplifyNodesQuery(callee, s_nodes);
        }
        
        private SimplifyNodesQuery(AnalysisSummary s, HashSet s_nodes) {
            this.summary = s;
            this.setOfNodes = s_nodes;
        }
        
        public void simplify() {
            constants = new HashSet();
            for (Iterator i=this.setOfNodes.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                if (n instanceof ParamNode) {
                    ParamNodesQuery pnq = ParamNodesQuery.create(summary, ((ParamNode)n).n);
                    subqueries.add(pnq);
                } else if (n instanceof FieldNode) {
                    FieldNodesQuery fnq = FieldNodesQuery.create(summary, (FieldNode)n);
                    fnq.simplify();
                    this.addResult(fnq);
                } else if (n instanceof ReturnedNode) {
                    MethodCall that_mc = ((ReturnedNode)n).getMethodCall();
                    CallTargetsQuery ctq = CallTargetsQuery.create(summary, that_mc.q);
                    ctq.simplify();
                    ReturnValueNodesQuery rvnq = ReturnValueNodesQuery.create(summary, ctq);
                    rvnq.simplify();
                    if (rvnq.isSimple()) {
                        this.addResult(rvnq);
                    } else {
                        subqueries.add(rvnq);
                    }
                } else {
                    jq.assert(!(n instanceof OutsideNode));
                    constants.add(n);
                }
            }
        }
        
        public void receiveCallTargets(CallTargetsQuery ctqa) {
            HashSet set = new HashSet();
            ctqa.getTerminals(set);
            for (Iterator i=set.iterator(); i.hasNext(); ) {
                jq_Method target = (jq_Method)i.next();
                ControlFlowGraph cfg = CodeCache.getCode(target);
                AnalysisSummary s_callee = getSummary(cfg);
                boolean r = false, t = false;
                HashSet s = queryToNode.get(ctqa);
                MethodCall that_mc = null;
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    jq.assert(that_mc == null || that_mc == ((ReturnedNode)o).getMethodCall());
                    that_mc = ((ReturnedNode)o).getMethodCall();
                    if (n instanceof ReturnValueNode) r = true;
                    else if (n instanceof ThrownExceptionNode) t = true;
                    else jq.UNREACHABLE();
                }
                if (r) {
                    ReturnValueNodesQuery rvnq = new ReturnValueNodesQuery(s_callee, true, null);
                    spawnOrLookupQuery(rvnq);
                    HashSet s = null;
                    if (queryToNode == null) queryToNode = new HashMap();
                    else s = (HashSet)queryToNode.get(rvnq);
                    if (s == null) queryToNode.put(rvnq, s = new HashSet());
                    s.add(that_mc);
                }
                if (t) {
                    ReturnValueNodesQuery rvnq = new ReturnValueNodesQuery(s_callee, false, null);
                    spawnOrLookupQuery(rvnq);
                    HashSet s = null;
                    if (queryToNode == null) queryToNode = new HashMap();
                    else s = (HashSet)queryToNode.get(rvnq);
                    if (s == null) queryToNode.put(rvnq, s = new HashSet());
                    s.add(that_mc);
                }
            }
        }
        
        public void receiveNodes(NodeQuery fnq) {
            Object o = queryToNode.get(fnq);
            if (o == null) {
                qa = new WrappedQueryResult(fnq, qa);
            } else {
                HashSet s = (HashSet)o;
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    MethodCall that_mc = (MethodCall)i.next();
                    Context c = new Context(callee, that_mc);
                    QueryResult qa2 = rvnq.applyContext(c);
                    qa = new WrappedQueryResult(qa2, qa);
                }
            } else {
                jq.UNREACHABLE();
            }
        }
        
        protected NodesQuery _applyContext(Context c, NodesQuery qa2) {
            NodesQuery qa = this.qa;
            while (qa != null) {
                qa2 = qa._applyContext(c, qa2);
                qa = qa.next;
            }
            for (Iterator i=spawnedQueries.iterator(); i.hasNext(); ) {
                
            }
            return qa2;
            
            return new ConstantNodesQuery(result, qa2);
        }
        protected void _getTerminals(HashSet r) {
            r.addAll(result);
        }
        
    }
    
    
    
    /** NodeQuery objects form a linked list. */
    public static abstract class NodesQuery {
        protected NodesQuery next;
        protected NodesQuery(NodesQuery n) { this.next = n; }
        
        /** Combines all of the terminals into a single set, returning a new
         *  list of NodesQuery objects sans those terminals. */
        public final void getTerminals(HashSet result) {
            NodesQuery qa = this;
            while (qa != null) {
                qa._getTerminals(result);
                qa = qa.next;
            }
        }
        
        /** Returns a new list of NodesQuery objects that correspond to applying
         *  the given context to this list. */
        public NodesQuery applyContext(Context c) {
            NodesQuery qa = this;
            NodesQuery qa2 = null;
            while (qa != null) {
                qa2 = qa._applyContext(c, qa2);
                qa = qa.next;
            }
            return qa2;
        }
        protected abstract NodesQuery _applyContext(Context c, NodesQuery qa2);
        protected abstract void _getTerminals(HashSet result);
    }
    
    
    public static abstract class NodesQueryResult {
        protected NodesQueryResult next;

        protected NodesQueryResult(NodesQueryResult n) { this.next = n; }
        
        public final NodesQueryResult getResult(Context c) {
            NodesQueryResult qa = this;
            NodesQueryResult qa2 = null;
            while (qa != null) {
                qa2 = qa._getResult(c, qa2);
                qa = qa.next;
            }
            return qa2;
        }
        protected abstract NodesQueryResult _getResult(Context c, NodesQueryResult qa2);
        
    }
    
    public static class WrappedNodesQueryResult extends NodesQueryResult {
        private final NodesQueryResult nqa;
        public ConstantNodesQueryResult(NodesQueryResult a, NodesQueryResult nq) {
            super(nq);
            this.nqa = a;
        }
        protected NodesQueryResult _getResult(Context c, NodesQueryResult qa2) {
            return nqa.getResult(c, qa2);
        }
    }
    
    public static class ParamNodesQueryResult extends QueryResult {
        private final int paramNum;
        public ParamNodesQueryResult(int n, QueryResult nq) {
            super(nq);
            this.paramNum = n;
        }
        protected NodesQueryResult _getResult(Context c, NodesQueryResult qa2) {
            result.addAll(c.getParamNodes(this.paramNum));
        }
    }
    
    public static class CallTargetsQuery extends Query {
        final AnalysisSummary summary;
        final Quad q;
        public CallTargetsQuery(AnalysisSummary s, Quad quad) {
            this.summary = s; this.q = quad;
        }
        public void doIt() {
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
            NodesTypesQuery ntq = new NodesTypesQuery(summary, nodes);
            spawnQuery(ntq);
        }
        
        public void receiveResult(TypesQueryResult types, Node n) {
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
        }
    }
    
    public static abstract class CallTargetsQueryResult {
    }
    
    public static class ConstantCallTargetsQueryResult extends CallTargetsQueryResult {
        private final HashSet result;
        public ConstantNodesQueryResult(HashSet s, NodesQueryResult nq) {
            super(nq);
            this.result = s;
        }
        protected NodesQueryResult _getResult(Context c, NodesQueryResult qa2) {
            return new ConstantCallTargetsQueryResult(result, qa2);
        }
    }
    
    // Solves constraints using a worklist algorithm.
    public static class ConstraintSolver {
        HashMap all_constraints;
        HashSet open_constraints;
        LinkedList worklist;

        ConstraintSolver() {
            this.all_constraints = new HashMap();
            this.open_constraints = new HashSet();
            this.worklist = new LinkedList();
        }
        
        void solveConstraint(Constraint q) {
            this.all_constraints.clear(); Constraint.current_id = 0;
            if (TRACE_INTER) out.println(" ... Starting constraint solver ...");
            worklist.add(q); open_constraints.add(q);
            performLoop();
        }
        
        void performLoop() {
            while (!worklist.isEmpty()) {
                Object o = worklist.removeFirst();
                open_constraints.remove(o);
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
                if (o instanceof Constraint) {
                    Constraint d = (Constraint)o;
                    if (TRACE_INTER) out.println("-=> Updating constraint "+d);
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
            Constraint d = new Constraint(q2, dq);
            if (open_queries.contains(d)) {
                if (TRACE_INTER) out.println("-=> Constraint "+d+" already open");
                return q2;
            }
            open_queries.add(d); worklist.add(d);
            return q2;
        }
        
        public static final QuerySolver GLOBAL = new QuerySolver();
    }

    // Records a directed constraint.  When the result of the 'from' query changes, the
    // 'to' query must be recalculated.
    public static class Constraint {
        Query from;
        DependentQuery to;
        
        Dependence(Query from, DependentQuery to) {
            this.from = from; this.to = to;
        }
        
        public boolean equals(Constraint that) {
            return this.from == that.from && this.to == that.to;
        }
        public boolean equals(Object o) {
            if (o instanceof Constraint) return equals((Constraint)o);
            return false;
        }
        public int hashCode() { return from.hashCode() ^ to.hashCode(); }
        public String toString() { return "q"+from.id+"->q"+to.id; }
    }
    
}

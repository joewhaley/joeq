/*
 * ContextSensitiveCallGraph.java
 *
 * Created on March 7, 2002, 3:03 PM
 */

package Compil3r.Quad;
import Bootstrap.PrimordialClassLoader;
import Util.Templates.List;
import Util.Templates.ListIterator;
import Clazz.*;
import java.util.HashSet;
import java.util.HashMap;
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
import Operator.Special;
import Operator.Unary;
import jq;

/**
 *
 * @author  Administrator
 * @version 
 */
public class ContextSensitiveCallGraph {

    public static java.io.PrintStream out = System.out;
    public static final boolean TRACE = true;
    
    /** Creates new ContextSensitiveCallGraph */
    public ContextSensitiveCallGraph() {
    }

    public static class MethodCall {
        final jq_Method m; final Quad q;
        public MethodCall(jq_Method m, Quad q) {
            this.m = m; this.q = q;
        }
        public int hashCode() { return q.hashCode(); }
        public boolean equals(MethodCall that) { return this.q == that.q; }
        public boolean equals(Object o) { if (o instanceof MethodCall) return equals((MethodCall)o); return false; }
        public String toString() { return m.getName()+"() at quad "+q.getID(); }
    }
    
    public static class PassedParameter {
        final MethodCall m; final int paramNum;
        public PassedParameter(MethodCall m, int paramNum) {
            this.m = m; this.paramNum = paramNum;
        }
        public int hashCode() { return m.hashCode() ^ paramNum; }
        public boolean equals(PassedParameter that) { return this.m.equals(that.m) && this.paramNum == that.paramNum; }
        public boolean equals(Object o) { if (o instanceof PassedParameter) return equals((PassedParameter)o); return false; }
        public String toString() { return m+" param "+paramNum; }
    }
    
    public abstract static class Node implements Cloneable {
        HashSet passedParameters;
        HashMap addedEdges;
        HashMap accessPathEdges;
        
        Node() {}
        Node(Node n) {
            this.passedParameters = n.passedParameters;
            this.addedEdges = n.addedEdges;
            this.accessPathEdges = n.accessPathEdges;
        }
        
        public Object clone() { return this.copy(); }
        
        abstract Node copy();

        boolean recordPassedParameter(MethodCall m, int paramNum) {
            if (passedParameters == null) passedParameters = new HashSet();
            PassedParameter cm = new PassedParameter(m, paramNum);
            return passedParameters.add(cm);
        }
        boolean addEdge(jq_Field m, Node n) {
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
            if (addedEdges == null) addedEdges = new HashMap();
            Object o = addedEdges.get(m);
            if (o == null) {
                addedEdges.put(m, s);
                return true;
            }
            if (o instanceof HashSet) return ((HashSet)o).addAll(s);
            addedEdges.put(m, s); return s.add(o); 
        }
        boolean addAccessPathEdge(jq_Field m, FieldNode n) {
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

        public abstract String toString_short();
        public String toString() {
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
        ConcreteTypeNode(ConcreteTypeNode that) { super(that); this.type = that.type; this.q = that.q; }
        
        public boolean equals(ConcreteTypeNode that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof ConcreteTypeNode) return equals((ConcreteTypeNode)o);
            else return false;
        }
        public int hashCode() { return q.hashCode(); }
        
        Node copy() { return new ConcreteTypeNode(this); }
        
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "Concrete: "+type+" q: "+q.getID(); }
    }
    
    public static final class UnknownTypeNode extends Node {
        jq_Reference type; Quad q;
        
        UnknownTypeNode(jq_Reference type, Quad q) { this.type = type; this.q = q; }
        UnknownTypeNode(UnknownTypeNode that) { super(that); this.type = that.type; this.q = that.q; }
        
        public boolean equals(UnknownTypeNode that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof UnknownTypeNode) return equals((UnknownTypeNode)o);
            else return false;
        }
        public int hashCode() { return q.hashCode(); }
        
        Node copy() { return new UnknownTypeNode(this); }
        
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "Unknown: "+type+" q: "+q.getID(); }
    }
    
    public abstract static class OutsideNode extends Node {
        OutsideNode() {}
        OutsideNode(Node n) { super(n); }
    }
    
    public static final class GlobalNode extends OutsideNode {
        private GlobalNode() {}
        Node copy() { return this; }
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "global"; }
        public static final GlobalNode INSTANCE = new GlobalNode();
    }
    
    public static final class ReturnValueNode extends OutsideNode {
        MethodCall m;
        ReturnValueNode(MethodCall m) { this.m = m; }
        ReturnValueNode(ReturnValueNode that) { super(that); this.m = that.m; }
        public boolean equals(ReturnValueNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ReturnValueNode) return equals((ReturnValueNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        Node copy() { return new ReturnValueNode(this); }
        
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "Return value of "+m; }
    }
    
    public static final class CaughtExceptionNode extends OutsideNode {
        ExceptionHandler eh;
        CaughtExceptionNode(ExceptionHandler eh) { this.eh = eh; }
        CaughtExceptionNode(CaughtExceptionNode that) { super(that); this.eh = that.eh; }
        public boolean equals(CaughtExceptionNode that) { return this.eh.equals(that.eh); }
        public boolean equals(Object o) {
            if (o instanceof CaughtExceptionNode) return equals((CaughtExceptionNode)o);
            else return false;
        }
        public int hashCode() { return eh.hashCode(); }
        
        Node copy() { return new CaughtExceptionNode(this); }
        
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "Caught exception: "+eh; }
    }
    
    public static final class ThrownExceptionNode extends OutsideNode {
        MethodCall m;
        ThrownExceptionNode(MethodCall m) { this.m = m; }
        ThrownExceptionNode(ThrownExceptionNode that) { super(that); this.m = that.m; }
        public boolean equals(ThrownExceptionNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ThrownExceptionNode) return equals((ThrownExceptionNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        Node copy() { return new ThrownExceptionNode(this); }
        
        public String toString() { return toString_short()+super.toString(); }
        public String toString_short() { return "Thrown exception of "+m; }
    }
    
    public static final class ParamNode extends OutsideNode {
        jq_Method m; int n; jq_Type declaredType;
        
        ParamNode(jq_Method m, int n, jq_Type declaredType) { this.m = m; this.n = n; this.declaredType = declaredType; }
        ParamNode(ParamNode that) { super(that); this.m = that.m; this.n = that.n; this.declaredType = that.declaredType; }

        public boolean equals(ParamNode that) { return this.n == that.n && this.m == that.m; }
        public boolean equals(Object o) {
            if (o instanceof ParamNode) return equals((ParamNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode() ^ n; }
        
        Node copy() { return new ParamNode(this); }
        
        public String toString() { return this.toString_short()+super.toString(); }
        public String toString_short() { return "Param#"+n+" method "+m.getName(); }
    }
    
    public static final class FieldNode extends OutsideNode {
        jq_Field f; Quad q;
        
        FieldNode(jq_Field f, Quad q) { this.f = f; this.q = q; }
        FieldNode(FieldNode that) { super(that); this.f = that.f; this.q = that.q; }

        public boolean equals(FieldNode that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof FieldNode) return equals((FieldNode)o);
            else return false;
        }
        public int hashCode() { return q.hashCode(); }
        
        Node copy() { return new FieldNode(this); }
        
        public String toString() { return this.toString_short()+super.toString(); }
        public String toString_short() { return "FieldLoad "+f+" quad "+q.getID(); }
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
                if (a instanceof Node) that.registers[i] = ((Node)a).copy();
                else if (a instanceof HashSet) that.registers[i] = ((HashSet)a).clone();
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
                        if (TRACE) out.println("change in register "+i+" from adding set");
                        change = true;
                    }
                } else {
                    if (q.add(b)) {
                        if (TRACE) out.println("change in register "+i+" from adding "+b);
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
    
    public static final class MethodSummaryBuilder implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            BuildMethodSummary b = new BuildMethodSummary(cfg);
        }
    }
    
    public static final class BuildMethodSummary extends QuadVisitor.EmptyVisitor {
        
        final jq_Method method;
        final int nLocals, nRegisters;
        final ParamNode[] param_nodes;
        final State[] start_states;
        boolean change;
        BasicBlock bb;
        State s;
        final HashSet methodCalls;
        
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
            this.s = this.start_states[0] = new State(this.nRegisters);
            jq_Type[] params = this.method.getParamTypes();
            this.param_nodes = new ParamNode[params.length];
            for (int i=0, j=0; i<params.length; ++i, ++j) {
                if (params[i].isReferenceType()) {
                    setLocal(i, param_nodes[i] = new ParamNode(method, j, params[i]));
                } else if (params[i].getReferenceSize() == 8) ++j;
            }
            
            if (TRACE) out.println("Building summary for "+this.method);
            
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
                    if (TRACE) {
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
                if (TRACE) out.println(succ+" not yet visited.");
                this.start_states[succ.getID()] = this.s.copy();
                this.change = true;
            } else {
                if (TRACE) out.println("merging out set of "+bb+" "+jq.hex8(this.s.hashCode())+" into in set of "+succ+" "+jq.hex8(this.start_states[succ.getID()].hashCode()));
                if (this.start_states[succ.getID()].merge(this.s)) {
                    if (TRACE) out.println(succ+" in set changed");
                    this.change = true;
                }
            }
        }
        
        void heapLoad(HashSet result, Node base, jq_Field f, FieldNode fn) {
            base.addAccessPathEdge(f, fn);
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
            if (v instanceof HashSet) {
                for (Iterator i = ((HashSet)v).iterator(); i.hasNext(); ) {
                    ((Node)i.next()).recordPassedParameter(m, p);
                }
            } else {
                ((Node)v).recordPassedParameter(m, p);
            }
        }
        
        /** An array load instruction. */
        public void visitALoad(Quad obj) {
            if (obj.getOperator() instanceof Operator.ALoad.ALOAD_A) {
                if (TRACE) out.println("Visiting: "+obj);
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
                if (TRACE) out.println("Visiting: "+obj);
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
            if (TRACE) out.println("Visiting: "+obj);
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
                if (TRACE) out.println("Visiting: "+obj);
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
                if (TRACE) out.println("Visiting: "+obj);
                Register r = Getstatic.getDest(obj).getRegister();
                jq_Field f = Getstatic.getField(obj).getField();
                heapLoad(obj, r, GlobalNode.INSTANCE, f);
            }
        }
        /** A type instance of instruction. */
        public void visitInstanceOf(Quad obj) {
            // skip for now.
        }
        /** An invoke instruction. */
        public void visitInvoke(Quad obj) {
            if (TRACE) out.println("Visiting: "+obj);
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
                if (TRACE) out.println("Visiting: "+obj);
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
            if (TRACE) out.println("Visiting: "+obj);
            Register dest_r = New.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)New.getType(obj).getType();
            ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
            setRegister(dest_r, n);
        }
        /** An array allocation instruction. */
        public void visitNewArray(Quad obj) {
            if (TRACE) out.println("Visiting: "+obj);
            Register dest_r = NewArray.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)NewArray.getType(obj).getType();
            ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
            setRegister(dest_r, n);
        }
        /** A put instance field instruction. */
        public void visitPutfield(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A) ||
               (obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A_DYNLINK)) {
                if (TRACE) out.println("Visiting: "+obj);
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
                if (TRACE) out.println("Visiting: "+obj);
                Operand val = Putstatic.getSrc(obj);
                jq_Field f = Putstatic.getField(obj).getField();
                if (val instanceof RegisterOperand) {
                    Register src_r = ((RegisterOperand)val).getRegister();
                    heapStore(GlobalNode.INSTANCE, src_r, f);
                } else {
                    jq.assert(val instanceof AConstOperand);
                    jq_Reference type = ((AConstOperand)val).getType();
                    ConcreteTypeNode n = new ConcreteTypeNode(type, obj);
                    heapStore(GlobalNode.INSTANCE, n, f);
                }
            }
        }
        
        public void visitSpecial(Quad obj) {
            if (obj.getOperator() == Special.GET_THREAD_BLOCK.INSTANCE) {
                if (TRACE) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                ConcreteTypeNode n = new ConcreteTypeNode(Scheduler.jq_Thread._class, obj);
                setRegister(dest_r, n);
            } else if (obj.getOperator() == Special.GET_TYPE_OF.INSTANCE) {
                if (TRACE) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                UnknownTypeNode n = new UnknownTypeNode(Clazz.jq_Reference._class, obj);
                setRegister(dest_r, n);
            }
        }
        public void visitUnary(Quad obj) {
            if (obj.getOperator() == Unary.INT_2OBJECT.INSTANCE) {
                if (TRACE) out.println("Visiting: "+obj);
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
    
}

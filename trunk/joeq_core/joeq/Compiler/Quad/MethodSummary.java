/*
 * MethodSummary.java
 *
 * Created on April 24, 2002, 2:49 PM
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
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Iterator;
import Util.LinkedHashSet;
import Util.LinkedHashMap;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.CallTargets;
import Compil3r.Quad.BytecodeToQuad.jq_ReturnAddressType;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.Quad.Operand.AConstOperand;
import Compil3r.Quad.Operand.ParamListOperand;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operator.ALoad;
import Compil3r.Quad.Operator.AStore;
import Compil3r.Quad.Operator.CheckCast;
import Compil3r.Quad.Operator.Getfield;
import Compil3r.Quad.Operator.Getstatic;
import Compil3r.Quad.Operator.Invoke;
import Compil3r.Quad.Operator.Invoke.InvokeVirtual;
import Compil3r.Quad.Operator.Invoke.InvokeStatic;
import Compil3r.Quad.Operator.Invoke.InvokeInterface;
import Compil3r.Quad.Operator.Move;
import Compil3r.Quad.Operator.New;
import Compil3r.Quad.Operator.NewArray;
import Compil3r.Quad.Operator.Putfield;
import Compil3r.Quad.Operator.Putstatic;
import Compil3r.Quad.Operator.Return;
import Compil3r.Quad.Operator.Special;
import Compil3r.Quad.Operator.Unary;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class MethodSummary {

    public static java.io.PrintStream out = System.out;
    public static final boolean TRACE_INTRA = false;
    public static final boolean TRACE_INTER = false;
    public static final boolean IGNORE_INSTANCE_FIELDS = false;
    public static final boolean IGNORE_STATIC_FIELDS = false;

    public static final class MethodSummaryBuilder implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            MethodSummary s = getSummary(cfg);
        }
    }
    
    public static HashMap summary_cache = new HashMap();
    public static MethodSummary getSummary(ControlFlowGraph cfg) {
        MethodSummary s = (MethodSummary)summary_cache.get(cfg);
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
    
    /** Visitor class to build an intramethod summary. */
    public static final class BuildMethodSummary extends QuadVisitor.EmptyVisitor {
        
        /** The method that we are building a summary for. */
        protected final jq_Method method;
        /** The number of locals and number of registers. */
        protected final int nLocals, nRegisters;
        /** The parameter nodes. */
        protected final ParamNode[] param_nodes;
        /** The global node. */
        protected final GlobalNode my_global;
        /** The start states of the iteration. */
        protected final State[] start_states;
        /** The set of returned and thrown nodes. */
        protected final LinkedHashSet returned, thrown;
        /** The set of method calls made. */
        protected final LinkedHashSet methodCalls;
        /** The set of nodes that were ever passed as a parameter. */
        protected final LinkedHashSet passedAsParameter;
        /** The current basic block. */
        protected BasicBlock bb;
        /** The current state. */
        protected State s;
        /** Change bit for worklist iteration. */
        protected boolean change;
        
        /** Factory for nodes. */
        protected final HashMap quadsToNodes;
        
        /** Returns the summary. Call this after iteration has completed. */
        public MethodSummary getSummary() {
            MethodSummary s = new MethodSummary(method, param_nodes, my_global, methodCalls, returned, thrown, passedAsParameter);
            // merge global nodes.
            if ((my_global.accessPathEdges != null) || (my_global.addedEdges != null)) {
                Set set = Collections.singleton(GlobalNode.GLOBAL);
                my_global.replaceBy(set);
                /*
                for (Iterator i=my_global.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    if (o instanceof FieldNode)
                        GlobalNode.GLOBAL.addAccessPathEdge(f, (FieldNode)o);
                    else
                        GlobalNode.GLOBAL.addAccessPathEdges(f, (LinkedHashSet)o);
                }
                 */
            }
            return s;
        }
        
        /** Set the given local in the current state to point to the given node. */
        protected void setLocal(int i, Node n) { s.registers[i] = n; }
        /** Set the given register in the current state to point to the given node. */
        protected void setRegister(Register r, Node n) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            s.registers[i] = n;
        }
        /** Set the given register in the current state to point to the given node or set of nodes. */
        protected void setRegister(Register r, Object n) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            if (n instanceof LinkedHashSet) n = ((LinkedHashSet)n).clone();
            else jq.assert(n instanceof Node);
            s.registers[i] = n;
        }
        /** Get the node or set of nodes in the given register in the current state. */
        protected Object getRegister(Register r) {
            int i = r.getNumber();
            if (r.isTemp()) i += nLocals;
            return s.registers[i];
        }

        /** Build a summary for the given method. */
        public BuildMethodSummary(ControlFlowGraph cfg) {
            RegisterFactory rf = cfg.getRegisterFactory();
            this.nLocals = rf.getLocalSize(PrimordialClassLoader.getJavaLangObject());
            this.nRegisters = this.nLocals + rf.getStackSize(PrimordialClassLoader.getJavaLangObject());
            this.method = cfg.getMethod();
            this.start_states = new State[cfg.getNumberOfBasicBlocks()];
            this.methodCalls = new LinkedHashSet();
            this.passedAsParameter = new LinkedHashSet();
            this.quadsToNodes = new HashMap();
            this.s = this.start_states[0] = new State(this.nRegisters);
            jq_Type[] params = this.method.getParamTypes();
            this.param_nodes = new ParamNode[params.length];
            for (int i=0, j=0; i<params.length; ++i, ++j) {
                if (params[i].isReferenceType()) {
                    setLocal(j, param_nodes[i] = new ParamNode(method, i, (jq_Reference)params[i]));
                } else if (params[i].getReferenceSize() == 8) ++j;
            }
            this.my_global = new GlobalNode();
            this.returned = new LinkedHashSet(); this.thrown = new LinkedHashSet();
            
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
                    /*
                    if (this.bb.isExceptionHandlerEntry()) {
                        java.util.Iterator i = cfg.getExceptionHandlersMatchingEntry(this.bb);
                        jq.assert(i.hasNext());
                        ExceptionHandler eh = (ExceptionHandler)i.next();
                        CaughtExceptionNode n = new CaughtExceptionNode(eh);
                        if (i.hasNext()) {
                            LinkedHashSet set = new LinkedHashSet(); set.add(n);
                            while (i.hasNext()) {
                                eh = (ExceptionHandler)i.next();
                                n = new CaughtExceptionNode(eh);
                                set.add(n);
                            }
                            s.merge(nLocals, set);
                        } else {
                            s.merge(nLocals, n);
                        }
                    }
                     */
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

        /** Merge the current state into the start state for the given basic block.
         *  If that start state is uninitialized, it is initialized with a copy of
         *  the current state.  This updates the change flag if anything is changed. */
        protected void mergeWith(BasicBlock succ) {
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
        
        /** Merge the current state into the start state for the given basic block.
         *  If that start state is uninitialized, it is initialized with a copy of
         *  the current state.  This updates the change flag if anything is changed. */
        protected void mergeWith(ExceptionHandler eh) {
            BasicBlock succ = eh.getEntry();
            if (this.start_states[succ.getID()] == null) {
                if (TRACE_INTRA) out.println(succ+" not yet visited.");
                this.start_states[succ.getID()] = this.s.copy();
                for (int i=nLocals; i<this.s.registers.length; ++i) {
                    this.start_states[succ.getID()].registers[i] = null;
                }
                this.change = true;
            } else {
                if (TRACE_INTRA) out.println("merging out set of "+bb+" "+jq.hex8(this.s.hashCode())+" into in set of ex handler "+succ+" "+jq.hex8(this.start_states[succ.getID()].hashCode()));
                for (int i=0; i<nLocals; ++i) {
                    if (this.start_states[succ.getID()].merge(i, this.s.registers[i]))
                        this.change = true;
                }
                if (TRACE_INTRA && this.change) out.println(succ+" in set changed");
            }
        }
        
        public static final boolean INSIDE_EDGES = false;
        
        /** Abstractly perform a heap load operation on the given base and field
         *  with the given field node, putting the result in the given set. */
        protected void heapLoad(LinkedHashSet result, Node base, jq_Field f, FieldNode fn) {
            //base.addAccessPathEdge(f, fn);
            result.add(fn);
            if (INSIDE_EDGES)
                base.getEdges(f, result);
        }
        /** Abstractly perform a heap load operation corresponding to quad 'obj'
         *  with the given destination register, bases and field.  The destination
         *  register in the current state is changed to the result. */
        protected void heapLoad(Quad obj, Register dest_r, LinkedHashSet base_s, jq_Field f) {
            LinkedHashSet result = new LinkedHashSet();
            for (Iterator i=base_s.iterator(); i.hasNext(); ) {
                Node base = (Node)i.next();
                FieldNode fn = FieldNode.get(base, f, obj);
                heapLoad(result, base, f, fn);
            }
            setRegister(dest_r, result);
        }
        /** Abstractly perform a heap load operation corresponding to quad 'obj'
         *  with the given destination register, base and field.  The destination
         *  register in the current state is changed to the result. */
        protected void heapLoad(Quad obj, Register dest_r, Node base_n, jq_Field f) {
            FieldNode fn = FieldNode.get(base_n, f, obj);
            LinkedHashSet result = new LinkedHashSet();
            heapLoad(result, base_n, f, fn);
            setRegister(dest_r, result);
        }
        /** Abstractly perform a heap load operation corresponding to quad 'obj'
         *  with the given destination register, base register and field.  The
         *  destination register in the current state is changed to the result. */
        protected void heapLoad(Quad obj, Register dest_r, Register base_r, jq_Field f) {
            Object o = getRegister(base_r);
            if (o instanceof LinkedHashSet) {
                heapLoad(obj, dest_r, (LinkedHashSet)o, f);
            } else {
                heapLoad(obj, dest_r, (Node)o, f);
            }
        }
        
        /** Abstractly perform a heap store operation of the given source node on
         *  the given base node and field. */
        protected void heapStore(Node base, Node src, jq_Field f) {
            base.addEdge(f, src);
        }
        /** Abstractly perform a heap store operation of the given source nodes on
         *  the given base node and field. */
        protected void heapStore(Node base, LinkedHashSet src, jq_Field f) {
            base.addEdges(f, (LinkedHashSet)src.clone());
        }
        /** Abstractly perform a heap store operation of the given source node on
         *  the nodes in the given register in the current state and the given field. */
        protected void heapStore(Register base_r, Node src_n, jq_Field f) {
            Object base = getRegister(base_r);
            if (base instanceof LinkedHashSet) {
                for (Iterator i = ((LinkedHashSet)base).iterator(); i.hasNext(); ) {
                    heapStore((Node)i.next(), src_n, f);
                }
            } else {
                heapStore((Node)base, src_n, f);
            }
        }
        /** Abstractly perform a heap store operation of the nodes in the given register
         *  on the given base node and field. */
        protected void heapStore(Node base, Register src_r, jq_Field f) {
            Object src = getRegister(src_r);
            if (src instanceof Node) {
                heapStore(base, (Node)src, f);
            } else {
                heapStore(base, (LinkedHashSet)src, f);
            }
        }
        /** Abstractly perform a heap store operation of the nodes in the given register
         *  on the nodes in the given register in the current state and the given field. */
        protected void heapStore(Register base_r, Register src_r, jq_Field f) {
            Object base = getRegister(base_r);
            Object src = getRegister(src_r);
            if (src instanceof Node) {
                heapStore(base_r, (Node)src, f);
                return;
            }
            LinkedHashSet src_h = (LinkedHashSet)src;
            if (base instanceof LinkedHashSet) {
                for (Iterator i = ((LinkedHashSet)base).iterator(); i.hasNext(); ) {
                    heapStore((Node)i.next(), src_h, f);
                }
            } else {
                heapStore((Node)base, src_h, f);
            }
        }

        /** Record that the nodes in the given register were passed to the given
         *  method call as the given parameter. */
        void passParameter(Register r, MethodCall m, int p) {
            Object v = getRegister(r);
            if (TRACE_INTRA) out.println("Passing "+r+" to "+m+" param "+p+": "+v);
            if (v instanceof LinkedHashSet) {
                for (Iterator i = ((LinkedHashSet)v).iterator(); i.hasNext(); ) {
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
        
        /** Visit an array load instruction. */
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
        /** Visit an array store instruction. */
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
                        ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                        if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                        heapStore(base_r, n, null);
                    }
                } else {
                    // base is not a register?!
                }
            }
        }
        /** Visit a type cast check instruction. */
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
                ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                setRegister(dest_r, n);
            }
        }
        /** Visit a get instance field instruction. */
        public void visitGetfield(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Getfield.GETFIELD_A) ||
               (obj.getOperator() instanceof Operator.Getfield.GETFIELD_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register r = Getfield.getDest(obj).getRegister();
                Operand o = Getfield.getBase(obj);
                jq_Field f = Getfield.getField(obj).getField();
                if (IGNORE_INSTANCE_FIELDS) f = null;
                if (o instanceof RegisterOperand) {
                    Register b = ((RegisterOperand)o).getRegister();
                    heapLoad(obj, r, b, f);
                } else {
                    // base is not a register?!
                }
            }
        }
        /** Visit a get static field instruction. */
        public void visitGetstatic(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Getstatic.GETSTATIC_A) ||
               (obj.getOperator() instanceof Operator.Getstatic.GETSTATIC_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register r = Getstatic.getDest(obj).getRegister();
                jq_Field f = Getstatic.getField(obj).getField();
                if (IGNORE_STATIC_FIELDS) f = null;
                heapLoad(obj, r, my_global, f);
            }
        }
        /** Visit a type instance of instruction. */
        public void visitInstanceOf(Quad obj) {
            // skip for now.
        }
        /** Visit an invoke instruction. */
        public void visitInvoke(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            jq_Method m = Invoke.getMethod(obj).getMethod();
            MethodCall mc = new MethodCall(m, obj);
            this.methodCalls.add(mc);
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
                    ReturnValueNode n = (ReturnValueNode)quadsToNodes.get(obj);
                    if (n == null) quadsToNodes.put(obj, n = new ReturnValueNode(mc));
                    setRegister(dest_r, n);
                }
            }
        }
        /** Visit a register move instruction. */
        public void visitMove(Quad obj) {
            if (obj.getOperator() instanceof Operator.Move.MOVE_A) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = Move.getDest(obj).getRegister();
                Operand src = Move.getSrc(obj);
                if (src instanceof RegisterOperand) {
                    RegisterOperand rop = ((RegisterOperand)src);
                    if (rop.getType() instanceof jq_ReturnAddressType) return;
                    Register src_r = rop.getRegister();
                    setRegister(dest_r, getRegister(src_r));
                } else {
                    jq.assert(src instanceof AConstOperand);
                    jq_Reference type = ((AConstOperand)src).getType();
                    ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                    if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                    setRegister(dest_r, n);
                }
            }
        }
        /** Visit an object allocation instruction. */
        public void visitNew(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            Register dest_r = New.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)New.getType(obj).getType();
            ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
            if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
            setRegister(dest_r, n);
        }
        /** Visit an array allocation instruction. */
        public void visitNewArray(Quad obj) {
            if (TRACE_INTRA) out.println("Visiting: "+obj);
            Register dest_r = NewArray.getDest(obj).getRegister();
            jq_Reference type = (jq_Reference)NewArray.getType(obj).getType();
            ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
            if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
            setRegister(dest_r, n);
        }
        /** Visit a put instance field instruction. */
        public void visitPutfield(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A) ||
               (obj.getOperator() instanceof Operator.Putfield.PUTFIELD_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Operand base = Putfield.getBase(obj);
                Operand val = Putfield.getSrc(obj);
                jq_Field f = Putfield.getField(obj).getField();
                if (IGNORE_INSTANCE_FIELDS) f = null;
                if (base instanceof RegisterOperand) {
                    Register base_r = ((RegisterOperand)base).getRegister();
                    if (val instanceof RegisterOperand) {
                        Register src_r = ((RegisterOperand)val).getRegister();
                        heapStore(base_r, src_r, f);
                    } else {
                        jq.assert(val instanceof AConstOperand);
                        jq_Reference type = ((AConstOperand)val).getType();
                        ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                        if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                        heapStore(base_r, n, f);
                    }
                } else {
                    // base is not a register?!
                }
            }
        }
        /** Visit a put static field instruction. */
        public void visitPutstatic(Quad obj) {
            if ((obj.getOperator() instanceof Operator.Putstatic.PUTSTATIC_A) ||
               (obj.getOperator() instanceof Operator.Putstatic.PUTSTATIC_A_DYNLINK)) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Operand val = Putstatic.getSrc(obj);
                jq_Field f = Putstatic.getField(obj).getField();
                if (IGNORE_STATIC_FIELDS) f = null;
                if (val instanceof RegisterOperand) {
                    Register src_r = ((RegisterOperand)val).getRegister();
                    heapStore(my_global, src_r, f);
                } else {
                    jq.assert(val instanceof AConstOperand);
                    jq_Reference type = ((AConstOperand)val).getType();
                    ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                    if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                    heapStore(my_global, n, f);
                }
            }
        }
        
        static void addToSet(LinkedHashSet s, Object o) {
            if (o instanceof LinkedHashSet) s.addAll((LinkedHashSet)o);
            else if (o != null) s.add(o);
        }
        
        /** Visit a return/throw instruction. */
        public void visitReturn(Quad obj) {
            Operand src = Return.getSrc(obj);
            LinkedHashSet r;
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
                ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                r.add(n);
            }
        }
            
        public void visitSpecial(Quad obj) {
            if (obj.getOperator() == Special.GET_THREAD_BLOCK.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                jq_Reference type = Scheduler.jq_Thread._class;
                ConcreteTypeNode n = (ConcreteTypeNode)quadsToNodes.get(obj);
                if (n == null) quadsToNodes.put(obj, n = new ConcreteTypeNode(type, obj));
                setRegister(dest_r, n);
            } else if (obj.getOperator() == Special.GET_TYPE_OF.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = ((RegisterOperand)Special.getOp1(obj)).getRegister();
                jq_Reference type = Clazz.jq_Reference._class;
                UnknownTypeNode n = UnknownTypeNode.get(type);
                setRegister(dest_r, n);
            }
        }
        public void visitUnary(Quad obj) {
            if (obj.getOperator() == Unary.INT_2OBJECT.INSTANCE) {
                if (TRACE_INTRA) out.println("Visiting: "+obj);
                Register dest_r = Unary.getDest(obj).getRegister();
                jq_Reference type = PrimordialClassLoader.getJavaLangObject();
                UnknownTypeNode n = (UnknownTypeNode)quadsToNodes.get(type);
                setRegister(dest_r, n);
            }
        }
        public void visitExceptionThrower(Quad obj) {
            // special case for method invocation.
            if (obj.getOperator() instanceof Invoke) {
                jq_Method m = Invoke.getMethod(obj).getMethod();
                MethodCall mc = new MethodCall(m, obj);
                ThrownExceptionNode n = new ThrownExceptionNode(mc);
                ListIterator.ExceptionHandler eh = bb.getExceptionHandlers().exceptionHandlerIterator();
                while (eh.hasNext()) {
                    ExceptionHandler h = eh.nextExceptionHandler();
                    this.mergeWith(h);
                    this.start_states[h.getEntry().getID()].merge(nLocals, n);
                    if (h.mustCatch(Bootstrap.PrimordialClassLoader.getJavaLangThrowable()))
                        return;
                }
                this.thrown.add(n);
                return;
            }
            ListIterator.jq_Class xs = obj.getThrownExceptions().classIterator();
            while (xs.hasNext()) {
                jq_Class x = xs.nextClass();
                UnknownTypeNode n = UnknownTypeNode.get(x);
                ListIterator.ExceptionHandler eh = bb.getExceptionHandlers().exceptionHandlerIterator();
                boolean caught = false;
                while (eh.hasNext()) {
                    ExceptionHandler h = eh.nextExceptionHandler();
                    if (h.mayCatch(x)) {
                        this.mergeWith(h);
                        this.start_states[h.getEntry().getID()].merge(nLocals, n);
                    }
                    if (h.mustCatch(x)) {
                        caught = true;
                        break;
                    }
                }
                if (!caught) this.thrown.add(n);
            }
        }
        
    }
    
    /** Represents a particular method call. */
    public static class MethodCall {
        final jq_Method m; final Quad q;
        public MethodCall(jq_Method m, Quad q) {
            this.m = m; this.q = q;
        }
        public int hashCode() { return (q==null)?-1:q.hashCode(); }
        public boolean equals(MethodCall that) { return this.q == that.q; }
        public boolean equals(Object o) { if (o instanceof MethodCall) return equals((MethodCall)o); return false; }
        public String toString() { return "quad "+((q==null)?-1:q.getID())+" "+m.getName()+"()"; }
        
        private byte getType() {
            if (q.getOperator() instanceof InvokeVirtual) {
                return BytecodeVisitor.INVOKE_VIRTUAL;
            } else if (q.getOperator() instanceof InvokeStatic) {
                if (m instanceof jq_InstanceMethod)
                    return BytecodeVisitor.INVOKE_SPECIAL;
                else
                    return BytecodeVisitor.INVOKE_STATIC;
            } else {
                jq.assert(q.getOperator() instanceof InvokeInterface);
                return BytecodeVisitor.INVOKE_INTERFACE;
            }
        }
        
        public CallTargets getCallTargets() {
            byte type = getType();
            return CallTargets.getTargets(m.getDeclaringClass(), m, type, true);
        }
        
        public CallTargets getCallTargets(Node n) {
            return getCallTargets(n.getDeclaredType(), n instanceof ConcreteTypeNode);
        }
        
        public CallTargets getCallTargets(jq_Reference klass, boolean exact) {
            byte type = getType();
            return CallTargets.getTargets(m.getDeclaringClass(), m, type, klass, exact, true);
        }
        
        public CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact) {
            byte type = getType();
            return CallTargets.getTargets(m.getDeclaringClass(), m, type, receiverTypes, exact, true);
        }
        
        public CallTargets getCallTargets(java.util.Set nodes) {
            byte type = getType();
            boolean exact = true;
            LinkedHashSet types = new LinkedHashSet();
            for (Iterator i=nodes.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                if (!(n instanceof ConcreteTypeNode)) exact = false;
                if (n.getDeclaredType() != null)
                    types.add(n.getDeclaredType());
            }
            return getCallTargets(types, exact);
        }
    }
    
    /** Represents a particular parameter passed to a particular method call. */
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
    
    public static class CallSite {
        final MethodSummary caller; final MethodCall m;
        public CallSite(MethodSummary caller, MethodCall m) {
            this.caller = caller; this.m = m;
        }
        public int hashCode() { return caller.hashCode() ^ m.hashCode(); }
        public boolean equals(CallSite that) { return this.m.equals(that.m) && this.caller == that.caller; }
        public boolean equals(Object o) { if (o instanceof CallSite) return equals((CallSite)o); return false; }
        public String toString() { return caller.getMethod()+" "+m.toString(); }
    }
    
    public abstract static class Node implements Cloneable {
        /** Map from fields to sets of predecessors on that field. 
         *  This only includes inside edges; outside edge predecessors are in FieldNode. */
        LinkedHashMap predecessors;
        /** Set of passed parameters for this node. */
        LinkedHashSet passedParameters;
        /** Map from fields to sets of inside edges from this node on that field. */
        LinkedHashMap addedEdges;
        /** Map from fields to sets of outside edges from this node on that field. */
        LinkedHashMap accessPathEdges;
        
        protected Node() {}
        protected Node(Node n) {
            this.predecessors = n.predecessors;
            this.passedParameters = n.passedParameters;
            this.addedEdges = n.addedEdges;
            this.accessPathEdges = n.accessPathEdges;
        }
        
        /** Replace this node by the given set of nodes.  All inside and outside
         *  edges to and from this node are replaced by sets of edges to and from
         *  the nodes in the set.  The passed parameter set of this node is also
         *  added to every node in the given set. */
        public void replaceBy(Set set) {
            jq.assert(!set.contains(this));
            if (this.predecessors != null) {
                for (Iterator i=this.predecessors.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    jq_Field f = (jq_Field)e.getKey();
                    Object o = e.getValue();
                    i.remove();
                    if (o instanceof Node) {
                        Node that = (Node)o;
                        if (that == this) {
                            // add self-cycles on f to all nodes in set.
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                Node k = (Node)j.next();
                                k.addEdge(f, k);
                            }
                            continue;
                        }
                        that._removeEdge(f, this);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            that.addEdge(f, (Node)j.next());
                        }
                    } else {
                        for (Iterator k=((LinkedHashSet)o).iterator(); k.hasNext(); ) {
                            Node that = (Node)k.next();
                            k.remove();
                            if (that == this) {
                                // add self-cycles on f to all mapped nodes.
                                for (Iterator j=set.iterator(); j.hasNext(); ) {
                                    Node k2 = (Node)j.next();
                                    k2.addEdge(f, k2);
                                }
                                continue;
                            }
                            that._removeEdge(f, this);
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
                    i.remove();
                    if (o instanceof Node) {
                        Node that = (Node)o;
                        jq.assert(that != this); // cyclic edges handled above.
                        that.removePredecessor(f, this);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            Node node2 = (Node)j.next();
                            node2.addEdge(f, that);
                        }
                    } else {
                        for (Iterator k=((LinkedHashSet)o).iterator(); k.hasNext(); ) {
                            Node that = (Node)k.next();
                            k.remove();
                            jq.assert(that != this); // cyclic edges handled above.
                            that.removePredecessor(f, this);
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                Node node2 = (Node)j.next();
                                node2.addEdge(f, that);
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
                    i.remove();
                    if (o instanceof FieldNode) {
                        FieldNode that = (FieldNode)o;
                        jq.assert(that != this); // cyclic edges handled above.
                        that.field_predecessors.remove(this);
                        for (Iterator j=set.iterator(); j.hasNext(); ) {
                            Node node2 = (Node)j.next();
                            node2.addAccessPathEdge(f, that);
                        }
                    } else {
                        for (Iterator k=((LinkedHashSet)o).iterator(); k.hasNext(); ) {
                            FieldNode that = (FieldNode)k.next();
                            k.remove();
                            jq.assert(that != this); // cyclic edges handled above.
                            that.field_predecessors.remove(this);
                            for (Iterator j=set.iterator(); j.hasNext(); ) {
                                Node node2 = (Node)j.next();
                                node2.addAccessPathEdge(f, that);
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
        
        /** Helper function to update map m given an update map um. */
        static void updateMap(HashMap um, Iterator i, LinkedHashMap m) {
            while (i.hasNext()) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                if (o instanceof Node) {
                    m.put(f, um.get(o));
                } else {
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
                        m.put(f, um.get(j.next()));
                    }
                }
            }
        }
        
        /** Update all predecessor and successor nodes with the given update map.
         *  Also clones the passed parameter set.
         */
        public void update(HashMap um) {
            LinkedHashMap m = this.predecessors;
            if (m != null) {
                this.predecessors = new LinkedHashMap();
                updateMap(um, m.entrySet().iterator(), this.predecessors);
            }
            m = this.addedEdges;
            if (m != null) {
                this.addedEdges = new LinkedHashMap();
                updateMap(um, m.entrySet().iterator(), this.addedEdges);
            }
            m = this.accessPathEdges;
            if (m != null) {
                this.accessPathEdges = new LinkedHashMap();
                updateMap(um, m.entrySet().iterator(), this.accessPathEdges);
            }
            if (this.passedParameters != null) {
                this.passedParameters = (LinkedHashSet)this.passedParameters.clone();
            }
        }
        
        /** Return the declared type of this node. */
        public abstract jq_Reference getDeclaredType();
        
        /** Return true if this node equals another node.
         *  Two nodes are equal if they have all the same edges and equivalent passed
         *  parameter sets.
         */
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
        /** Return a shallow copy of this node. */
        public Object clone() { return this.copy(); }
        
        /** Return a shallow copy of this node. */
        public abstract Node copy();

        /** Remove the given predecessor node on the given field from the predecessor set.
         *  Returns true if that predecessor existed, false otherwise. */
        public boolean removePredecessor(jq_Field m, Node n) {
            if (predecessors == null) return false;
            Object o = predecessors.get(m);
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).remove(n);
            else if (o == n) { predecessors.remove(m); return true; }
            else return false;
        }
        /** Add the given predecessor node on the given field to the predecessor set.
         *  Returns true if that predecessor didn't already exist, false otherwise. */
        public boolean addPredecessor(jq_Field m, Node n) {
            if (predecessors == null) predecessors = new LinkedHashMap();
            Object o = predecessors.get(m);
            if (o == null) {
                predecessors.put(m, n);
                return true;
            }
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).add(n);
            if (o == n) return false;
            LinkedHashSet s = new LinkedHashSet(); s.add(o); s.add(n);
            predecessors.put(m, s);
            return true;
        }
        
        /** Record the given passed parameter in the set for this node.
         *  Returns true if that passed parameter didn't already exist, false otherwise. */
        public boolean recordPassedParameter(PassedParameter cm) {
            if (passedParameters == null) passedParameters = new LinkedHashSet();
            return passedParameters.add(cm);
        }
        /** Record the passed parameter of the given method call and argument number in
         *  the set for this node.
         *  Returns true if that passed parameter didn't already exist, false otherwise. */
        public boolean recordPassedParameter(MethodCall m, int paramNum) {
            if (passedParameters == null) passedParameters = new LinkedHashSet();
            PassedParameter cm = new PassedParameter(m, paramNum);
            return passedParameters.add(cm);
        }
        private boolean _removeEdge(jq_Field m, Node n) {
            Object o = addedEdges.get(m);
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).remove(n);
            else if (o == n) { addedEdges.remove(m); return true; }
            else return false;
        }
        /** Remove the given successor node on the given field from the inside edge set.
         *  Also removes the predecessor link from the successor node to this node.
         *  Returns true if that edge existed, false otherwise. */
        public boolean removeEdge(jq_Field m, Node n) {
            if (addedEdges == null) return false;
            n.removePredecessor(m, this);
            return _removeEdge(m, n);
        }
        /** Add the given successor node on the given field to the inside edge set.
         *  Also adds a predecessor link from the successor node to this node.
         *  Returns true if that edge didn't already exist, false otherwise. */
        public boolean addEdge(jq_Field m, Node n) {
            n.addPredecessor(m, this);
            if (addedEdges == null) addedEdges = new LinkedHashMap();
            Object o = addedEdges.get(m);
            if (o == null) {
                addedEdges.put(m, n);
                return true;
            }
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).add(n);
            if (o == n) return false;
            LinkedHashSet s = new LinkedHashSet(); s.add(o); s.add(n);
            addedEdges.put(m, s);
            return true;
        }
        /** Add the given set of successor nodes on the given field to the inside edge set.
         *  The given set is consumed.
         *  Also adds predecessor links from the successor nodes to this node.
         *  Returns true if the inside edge set changed, false otherwise. */
        public boolean addEdges(jq_Field m, LinkedHashSet s) {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Node n = (Node)i.next();
                n.addPredecessor(m, this);
            }
            if (addedEdges == null) addedEdges = new LinkedHashMap();
            Object o = addedEdges.get(m);
            if (o == null) {
                addedEdges.put(m, s);
                return true;
            }
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).addAll(s);
            addedEdges.put(m, s); return s.add(o); 
        }
        /** Add the given successor node on the given field to the inside edge set
         *  of all of the given set of nodes.
         *  Also adds predecessor links from the successor node to the given nodes.
         *  Returns true if anything was changed, false otherwise. */
        public static boolean addEdges(LinkedHashSet s, jq_Field f, Node n) {
            boolean b = false;
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Node a = (Node)i.next();
                if (a.addEdge(f, n)) b = true;
            }
            return b;
        }
        
        private boolean _removeAccessPathEdge(jq_Field m, FieldNode n) {
            Object o = accessPathEdges.get(m);
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).remove(n);
            else if (o == n) { accessPathEdges.remove(m); return true; }
            else return false;
        }
        /** Remove the given successor node on the given field from the outside edge set.
         *  Also removes the predecessor link from the successor node to this node.
         *  Returns true if that edge existed, false otherwise. */
        public boolean removeAccessPathEdge(jq_Field m, FieldNode n) {
            if (accessPathEdges == null) return false;
            if (n.field_predecessors != null) n.field_predecessors.remove(this);
            return _removeAccessPathEdge(m, n);
        }
        /** Add the given successor node on the given field to the outside edge set.
         *  Also adds a predecessor link from the successor node to this node.
         *  Returns true if that edge didn't already exist, false otherwise. */
        public boolean addAccessPathEdge(jq_Field m, FieldNode n) {
            if (n.field_predecessors == null) n.field_predecessors = new LinkedHashSet();
            n.field_predecessors.add(this);
            if (accessPathEdges == null) accessPathEdges = new LinkedHashMap();
            Object o = accessPathEdges.get(m);
            if (o == null) {
                accessPathEdges.put(m, n);
                return true;
            }
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).add(n);
            if (o == n) return false;
            LinkedHashSet s = new LinkedHashSet(); s.add(o); s.add(n);
            accessPathEdges.put(m, s);
            return true;
        }
        /** Add the given set of successor nodes on the given field to the outside edge set.
         *  The given set is consumed.
         *  Also adds predecessor links from the successor nodes to this node.
         *  Returns true if the inside edge set changed, false otherwise. */
        public boolean addAccessPathEdges(jq_Field m, LinkedHashSet s) {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                FieldNode n = (FieldNode)i.next();
                if (n.field_predecessors == null) n.field_predecessors = new LinkedHashSet();
                n.field_predecessors.add(this);
            }
            if (accessPathEdges == null) accessPathEdges = new LinkedHashMap();
            Object o = accessPathEdges.get(m);
            if (o == null) {
                accessPathEdges.put(m, s);
                return true;
            }
            if (o instanceof LinkedHashSet) return ((LinkedHashSet)o).addAll(s);
            accessPathEdges.put(m, s); return s.add(o); 
        }
        
        /** Add the nodes that are targets of inside edges on the given field
         *  to the given result set. */
        public void getEdges(jq_Field m, LinkedHashSet result) {
            if (addedEdges == null) return;
            Object o = addedEdges.get(m);
            if (o == null) return;
            if (o instanceof LinkedHashSet) {
                result.addAll((LinkedHashSet)o);
            } else {
                result.add(o);
            }
        }
        
        /** Return a set of Map.Entry objects corresponding to the inside edges
         *  of this node. */
        public Set getEdges() {
            if (addedEdges == null) return Collections.EMPTY_SET;
            return addedEdges.entrySet();
        }

        /** Return the set of fields that this node has inside edges with. */
        public Set getEdgeFields() {
            if (addedEdges == null) return Collections.EMPTY_SET;
            return addedEdges.keySet();
        }
        
        /** Add the nodes that are targets of outside edges on the given field
         *  to the given result set. */
        public void getAccessPathEdges(jq_Field m, LinkedHashSet result) {
            if (accessPathEdges == null) return;
            Object o = accessPathEdges.get(m);
            if (o == null) return;
            if (o instanceof LinkedHashSet) {
                result.addAll((LinkedHashSet)o);
            } else {
                result.add(o);
            }
        }
        
        /** Return a set of Map.Entry objects corresponding to the outside edges
         *  of this node. */
        public Set getAccessPathEdges() {
            if (accessPathEdges == null) return Collections.EMPTY_SET;
            return accessPathEdges.entrySet();
        }
        
        /** Return the set of fields that this node has outside edges with. */
        public Set getAccessPathEdgeFields() {
            if (accessPathEdges == null) return Collections.EMPTY_SET;
            return accessPathEdges.keySet();
        }
        
        /** Return a string representation of the node in short form. */
        public abstract String toString_short();
        public String toString() { return toString_short(); }
        /** Return a string representation of the node in long form.
         *  Includes inside and outside edges and passed parameters. */
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
                        for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
                           sb.append(((Node)j.next()).toString_short());
                           if (j.hasNext()) sb.append(", ");
                        }
                    }
                    sb.append("}");
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
    
    /** A ConcreteTypeNode refers to an object with a concrete type.
     *  This is the result of a 'new' operation or a constant object.
     *  It is tied to the quad that created it, so nodes of the same type but
     *  from different quads are not equal.
     */
    public static final class ConcreteTypeNode extends Node {
        final jq_Reference type; final Quad q;
        
        static final HashMap FACTORY = new HashMap();
        public static ConcreteTypeNode get(jq_Reference type) {
            ConcreteTypeNode n = (ConcreteTypeNode)FACTORY.get(type);
            if (n == null) {
                FACTORY.put(type, n = new ConcreteTypeNode(type));
            }
            return n;
        }
        
        public ConcreteTypeNode(jq_Reference type) { this.type = type; this.q = null; }
        public ConcreteTypeNode(jq_Reference type, Quad q) { this.type = type; this.q = q; }
        private ConcreteTypeNode(ConcreteTypeNode that) {
            super(that); this.type = that.type; this.q = that.q;
        }
        
        public jq_Reference getDeclaredType() { return type; }
        
        /*
        public boolean equals(ConcreteTypeNode that) { return this.q == that.q; }
        public boolean equals(Object o) {
            if (o instanceof ConcreteTypeNode) return equals((ConcreteTypeNode)o);
            else return false;
        }
        public int hashCode() { return q.hashCode(); }
         */
        
        public Node copy() { return new ConcreteTypeNode(this); }
        
        public String toString_long() { return jq.hex(this)+": "+toString_short()+super.toString_long(); }
        public String toString_short() { return "Concrete: "+type+" q: "+(q==null?-1:q.getID()); }
    }
    
    /** A UnknownTypeNode refers to an object with an unknown type.  All that is
     *  known is that the object is the same or a subtype of some given type.
     *  Nodes with the same "type" are considered to be equal.
     *  This class includes a factory to get UnknownTypeNode's.
     */
    public static final class UnknownTypeNode extends Node {
        public static final boolean ADD_DUMMY_EDGES = false;
        
        static final HashMap FACTORY = new HashMap();
        public static UnknownTypeNode get(jq_Reference type) {
            UnknownTypeNode n = (UnknownTypeNode)FACTORY.get(type);
            if (n == null) {
                FACTORY.put(type, n = new UnknownTypeNode(type));
                if (ADD_DUMMY_EDGES) n.addDummyEdges();
            }
            return n;
        }
        
        final jq_Reference type;
        
        private UnknownTypeNode(jq_Reference type) {
            this.type = type;
        }
        private UnknownTypeNode(UnknownTypeNode that) { super(that); this.type = that.type; }
        
        /** Add the nodes that are targets of inside edges on the given field
         *  to the given result set. */
        public void getEdges(jq_Field m, LinkedHashSet result) {
            if (m == null) {
                if (this.type.isArrayType() || this.type == PrimordialClassLoader.getJavaLangObject())
                    result.add(get(PrimordialClassLoader.getJavaLangObject()));
                return;
            }
            this.type.load(); this.type.verify(); this.type.prepare();
            m.getDeclaringClass().load(); m.getDeclaringClass().verify(); m.getDeclaringClass().prepare();
            if (Run_Time.TypeCheck.isAssignable(this.type, m.getDeclaringClass()) ||
                Run_Time.TypeCheck.isAssignable(m.getDeclaringClass(), this.type)) {
                jq_Reference r = (jq_Reference)m.getType();
                result.add(get(r));
            }
            super.getEdges(m, result);
        }
        
        private void addDummyEdges() {
            if (type instanceof jq_Class) {
                jq_Class klass = (jq_Class)type;
                klass.load(); klass.verify(); klass.prepare();
                jq_InstanceField[] fields = klass.getInstanceFields();
                for (int i=0; i<fields.length; ++i) {
                    jq_InstanceField f = fields[i];
                    if (f.getType() instanceof jq_Reference) {
                        UnknownTypeNode n = get((jq_Reference)f.getType());
                        this.addEdge(f, n);
                    }
                }
            } else {
                jq_Array array = (jq_Array)type;
                if (array.getElementType() instanceof jq_Reference) {
                    UnknownTypeNode n = get((jq_Reference)array.getElementType());
                    this.addEdge(null, n);
                }
            }
        }
        
        public jq_Reference getDeclaredType() { return type; }
        
        /*
        public boolean equals(UnknownTypeNode that) { return this.type == that.type; }
        public boolean equals(Object o) {
            if (o instanceof UnknownTypeNode) return equals((UnknownTypeNode)o);
            else return false;
        }
        public int hashCode() { return type.hashCode(); }
         */
        
        public Node copy() { return new UnknownTypeNode(this); }
        
        public String toString_long() { return jq.hex(this)+": "+toString_short()+super.toString_long(); }
        public String toString_short() { return "Unknown: "+type; }
    }
    
    /** An outside node is some node that can be mapped to other nodes.
     *  This is just a marker for some of the other node classes below.
     */
    public abstract static class OutsideNode extends Node {
        OutsideNode() {}
        OutsideNode(Node n) { super(n); }
        
        public abstract jq_Reference getDeclaredType();
        
        OutsideNode skip;
        boolean visited;
        
    }
    
    /** A GlobalNode stores references to the static variables.
     *  It has no predecessors, and there is a global copy stored in GLOBAL.
     */
    public static final class GlobalNode extends OutsideNode {
        public GlobalNode() {}
        public jq_Reference getDeclaredType() { jq.UNREACHABLE(); return null; }
        public Node copy() { return this; }
        public String toString_long() { return jq.hex(this)+": "+toString_short()+super.toString_long(); }
        public String toString_short() { return "global"; }
        public static final GlobalNode GLOBAL = new GlobalNode();
    }
    
    /** A ReturnedNode represents a return value or thrown exception from a method call. */
    public abstract static class ReturnedNode extends OutsideNode {
        final MethodCall m;
        public ReturnedNode(MethodCall m) { this.m = m; }
        public ReturnedNode(ReturnedNode that) {
            super(that); this.m = that.m;
        }
        public final MethodCall getMethodCall() { return m; }
    }
    
    /** A ReturnValueNode represents the return value of a method call.
     */
    public static final class ReturnValueNode extends ReturnedNode {
        public ReturnValueNode(MethodCall m) { super(m); }
        private ReturnValueNode(ReturnValueNode that) { super(that); }
        
        public boolean equals(ReturnValueNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ReturnValueNode) return equals((ReturnValueNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        public jq_Reference getDeclaredType() { return (jq_Reference)m.m.getReturnType(); }
        
        public Node copy() { return new ReturnValueNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return jq.hex(this)+": "+"Return value of "+m; }
    }
    
    public static final class CaughtExceptionNode extends OutsideNode {
        final ExceptionHandler eh;
        LinkedHashSet caughtExceptions;
        public CaughtExceptionNode(ExceptionHandler eh) { this.eh = eh; }
        private CaughtExceptionNode(CaughtExceptionNode that) {
            super(that); this.eh = that.eh;
        }
        /*
        public boolean equals(CaughtExceptionNode that) { return this.eh.equals(that.eh); }
        public boolean equals(Object o) {
            if (o instanceof CaughtExceptionNode) return equals((CaughtExceptionNode)o);
            else return false;
        }
        public int hashCode() { return eh.hashCode(); }
         */
        
        public void addCaughtException(ThrownExceptionNode n) {
            if (caughtExceptions == null) caughtExceptions = new LinkedHashSet();
            caughtExceptions.add(n);
        }
        
        public jq_Reference getDeclaredType() { return (jq_Reference)eh.getExceptionType(); }
        
        public Node copy() { return new CaughtExceptionNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return jq.hex(this)+": "+"Caught exception: "+eh; }
    }
    
    /** A ThrownExceptionNode represents the thrown exception of a method call.
     */
    public static final class ThrownExceptionNode extends ReturnedNode {
        public ThrownExceptionNode(MethodCall m) { super(m); }
        private ThrownExceptionNode(ThrownExceptionNode that) { super(that); }
        
        public boolean equals(ThrownExceptionNode that) { return this.m.equals(that.m); }
        public boolean equals(Object o) {
            if (o instanceof ThrownExceptionNode) return equals((ThrownExceptionNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode(); }
        
        public jq_Reference getDeclaredType() { return Bootstrap.PrimordialClassLoader.getJavaLangObject(); }
        
        public Node copy() { return new ThrownExceptionNode(this); }
        
        public String toString_long() { return toString_short()+super.toString_long(); }
        public String toString_short() { return jq.hex(this)+": "+"Thrown exception of "+m; }
    }
    
    /** A ParamNode represents an incoming parameter.
     */
    public static final class ParamNode extends OutsideNode {
        final jq_Method m; final int n; final jq_Reference declaredType;
        
        public ParamNode(jq_Method m, int n, jq_Reference declaredType) { this.m = m; this.n = n; this.declaredType = declaredType; }
        private ParamNode(ParamNode that) {
            super(that); this.m = that.m; this.n = that.n; this.declaredType = that.declaredType;
        }

        /*
        public boolean equals(ParamNode that) { return this.n == that.n && this.m == that.m; }
        public boolean equals(Object o) {
            if (o instanceof ParamNode) return equals((ParamNode)o);
            else return false;
        }
        public int hashCode() { return m.hashCode() ^ n; }
         */
        
        public jq_Reference getDeclaredType() { return declaredType; }
        
        public Node copy() { return new ParamNode(this); }
        
        public String toString_long() { return jq.hex(this)+": "+this.toString_short()+super.toString_long(); }
        public String toString_short() { return "Param#"+n+" method "+m.getName(); }
    }
    
    /** A FieldNode represents the result of a 'load' instruction.
     *  There are outside edge links from the nodes that can be the base object
     *  of the load to this node.
     *  Two nodes are equal if the fields match and they are from the same quad.
     */
    public static final class FieldNode extends OutsideNode {
        final jq_Field f; final HashSet quads;
        LinkedHashSet field_predecessors;
        
        private static FieldNode findPredecessor(FieldNode base, Quad obj) {
            if (TRACE_INTRA) out.println("Checking "+base+" for predecessor "+obj.getID());
            if (base.quads.contains(obj)) {
                if (TRACE_INTRA) out.println("Success!");
                return base;
            }
            if (base.visited) {
                if (TRACE_INTRA) out.println(base+" already visited");
                return null;
            }
            base.visited = true;
            if (base.field_predecessors != null) {
                for (Iterator i=base.field_predecessors.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    if (o instanceof FieldNode) {
                        FieldNode fn = (FieldNode)o;
                        FieldNode fn2 = findPredecessor(fn, obj);
                        if (fn2 != null) {
                            base.visited = false;
                            return fn2;
                        }
                    }
                }
            }
            base.visited = false;
            return null;
        }
        
        public static FieldNode get(Node base, jq_Field f, Quad obj) {
            if (TRACE_INTRA) out.println("Getting field node for "+base+(f==null?"[]":("."+f.getName()))+" quad "+obj.getID());
            LinkedHashSet s = null;
            if (base.accessPathEdges != null) {
                Object o = base.accessPathEdges.get(f);
                if (o instanceof FieldNode) {
                    if (TRACE_INTRA) out.println("Field node for "+base+" already exists, reusing: "+o);
                    return (FieldNode)o;
                } else if (o != null) {
                    s = (LinkedHashSet)o;
                    if (!s.isEmpty()) {
                        if (TRACE_INTRA) out.println("Field node for "+base+" already exists, reusing: "+o);
                        return (FieldNode)s.iterator().next();
                    }
                }
            } else {
                base.accessPathEdges = new LinkedHashMap();
            }
            FieldNode fn;
            if (base instanceof FieldNode) fn = findPredecessor((FieldNode)base, obj);
            else fn = null;
            if (fn == null) fn = new FieldNode(f, obj);
            if (fn.field_predecessors == null) fn.field_predecessors = new LinkedHashSet();
            fn.field_predecessors.add(base);
            if (s != null) {
                jq.assert(base.accessPathEdges.get(f) == s);
                s.add(fn);
            } else {
                base.accessPathEdges.put(f, fn);
            }
            return fn;
        }
        
        private FieldNode(jq_Field f, Quad q) { this.f = f; this.quads = new HashSet(); this.quads.add(q); }
        private FieldNode(jq_Field f) { this.f = f; this.quads = new HashSet(); }
        private FieldNode(FieldNode that) {
            super(that); this.f = that.f; this.quads = that.quads; this.field_predecessors = that.field_predecessors;
        }

        /** Returns a new FieldNode that is the unification of the given set of FieldNodes.
         *  In essence, all of the given nodes are replaced by a new, returned node.
         *  The given field nodes must be on the given field.
         */
        public static FieldNode unify(jq_Field f, LinkedHashSet s) {
            if (TRACE_INTRA) out.println("Unifying the set of field nodes: "+s);
            FieldNode dis = new FieldNode(f);
            // go through once to add all quads, so that the hash code will be stable.
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                FieldNode dat = (FieldNode)i.next();
                jq.assert(f == dat.f);
                dis.quads.addAll(dat.quads);
            }
            // once again to do the replacement.
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                FieldNode dat = (FieldNode)i.next();
                Set s2 = Collections.singleton(dis);
                dat.replaceBy(s2);
            }
            if (TRACE_INTRA) out.println("Resulting field node: "+dis);
            return dis;
        }
        
        public void replaceBy(Set set) {
            jq.assert(!set.contains(this));
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
                    that._removeAccessPathEdge(f, this); i.remove();
                    for (Iterator j=set.iterator(); j.hasNext(); ) {
                        that.addAccessPathEdge(f, (FieldNode)j.next());
                    }
                }
            }
            super.replaceBy(set);
        }
        
        public void update(HashMap um) {
            super.update(um);
            LinkedHashSet m = this.field_predecessors;
            if (m != null) {
                this.field_predecessors = new LinkedHashSet();
                for (Iterator j=m.iterator(); j.hasNext(); ) {
                    this.field_predecessors.add(um.get(j.next()));
                }
            }
        }
        
        /*
        public boolean equals(FieldNode that) { return this.f == that.f && this.quads.equals(that.quads); }
        public boolean equals(Object o) {
            if (o instanceof FieldNode) return equals((FieldNode)o);
            else return false;
        }
        public int hashCode() { return ((f != null)?f.hashCode():0x6a953) ^ quads.hashCode(); }
         */
        
        public String fieldName() {
            if (f != null) return f.getName().toString();
            return getDeclaredType()+"[]";
        }
        
        public jq_Reference getDeclaredType() {
            if (f != null) {
                return (jq_Reference)f.getType();
            }
            if (quads.isEmpty()) return PrimordialClassLoader.getJavaLangObject();
            RegisterOperand r = ALoad.getDest((Quad)quads.iterator().next());
            return (jq_Reference)r.getType();
        }
        
        public Node copy() { return new FieldNode(this); }
        
        public String toString_long() { return jq.hex(this)+": "+this.toString_short()+super.toString_long(); }
        public String toString_short() {
            StringBuffer sb = new StringBuffer();
            sb.append("FieldLoad ");
            sb.append(fieldName());
            Iterator i=quads.iterator();
            if (i.hasNext()) {
                int id = ((Quad)i.next()).getID();
                if (!i.hasNext()) {
                    sb.append(" quad ");
                    sb.append(id);
                } else {
                    sb.append(" quads {");
                    sb.append(id);
                    while (i.hasNext()) {
                        sb.append(',');
                        sb.append(((Quad)i.next()).getID());
                    }
                    sb.append('}');
                }
            }
            return sb.toString();
        }
    }
    
    /** Records the state of the intramethod analysis at some point in the method. */
    public static final class State implements Cloneable {
        final Object[] registers;
        /** Return a new state with the given number of registers. */
        public State(int nRegisters) {
            this.registers = new Object[nRegisters];
        }
        public Object clone() { return this.copy(); }
        /** Return a shallow copy of this state.
         *  Sets of nodes are copied, but the individual nodes are not. */
        public State copy() {
            State that = new State(this.registers.length);
            for (int i=0; i<this.registers.length; ++i) {
                Object a = this.registers[i];
                if (a == null) continue;
                if (a instanceof Node)
                    //that.registers[i] = ((Node)a).copy();
                    that.registers[i] = a;
                else {
                    that.registers[i] = ((LinkedHashSet)a).clone();
                }
            }
            return that;
        }
        /** Merge two states.  Mutates this state, the other is unchanged. */
        public boolean merge(State that) {
            boolean change = false;
            for (int i=0; i<this.registers.length; ++i) {
                if (merge(i, that.registers[i])) change = true;
            }
            return change;
        }
        /** Merge the given node or set of nodes into the given register. */
        public boolean merge(int i, Object b) {
            if (b == null) return false;
            Object a = this.registers[i];
            if (b.equals(a)) return false;
            LinkedHashSet q;
            if (!(a instanceof LinkedHashSet)) {
                this.registers[i] = q = new LinkedHashSet();
                if (a != null) q.add(a);
            } else {
                q = (LinkedHashSet)a;
            }
            if (b instanceof LinkedHashSet) {
                if (q.addAll((LinkedHashSet)b)) {
                    if (TRACE_INTRA) out.println("change in register "+i+" from adding set");
                    return true;
                }
            } else {
                if (q.add(b)) {
                    if (TRACE_INTRA) out.println("change in register "+i+" from adding "+b);
                    return true;
                }
            }
            return false;
        }
        /** Dump a textual representation of the state to the given print stream. */
        public void dump(java.io.PrintStream out) {
            for (int i=0; i<registers.length; ++i) {
                if (registers[i] == null) continue;
                out.print(i+": "+registers[i]+" ");
            }
            out.println();
        }
    }
    
    /** Encodes an access path.
     *  An access path is an NFA, where transitions are field names.
     *  Each node in the NFA is represented by an AccessPath object.
     *  We try to share AccessPath objects as much as possible.
     */
    public static class AccessPath {
        /** All incoming transitions have this field. */
        jq_Field _field;
        /** The incoming transitions are associated with this AccessPath object. */
        Node _n;
        /** Whether this is a valid end state. */
        boolean _last;
        
        /** The set of (wrapped) successor AccessPath objects. */
        LinkedHashSet succ;

        /** Adds the set of (wrapped) AccessPath objects that are reachable from this
         *  AccessPath object to the given set. */
        private void reachable(LinkedHashSet s) {
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    s.add(ap);
                    ((AccessPath)ap.getObject()).reachable(s);
                }
            }
        }
        /** Return an iteration of the AccessPath objects that are reachable from
         *  this AccessPath. */
        public Iterator reachable() {
            LinkedHashSet s = new LinkedHashSet();
            s.add(IdentityHashCodeWrapper.create(this));
            this.reachable(s);
            return new FilterIterator(s.iterator(), filter);
        }
        
        /** Add the given AccessPath object as a successor to this AccessPath object. */
        private void addSuccessor(AccessPath ap) {
            succ.add(IdentityHashCodeWrapper.create(ap));
        }
        
        /** Return an access path that is equivalent to the given access path prepended
         *  with a transition on the given field and node.  The given access path can
         *  be null (empty). */
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
        
        /** Return an access path that is equivalent to the given access path appended
         *  with a transition on the given field and node.  The given access path can
         *  be null (empty). */
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
        
        /** Helper function for findLast(), below. */
        private void findLast(HashSet s, LinkedHashSet last) {
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    s.add(ap);
                    AccessPath that = (AccessPath)ap.getObject();
                    if (that._last) last.add(ap);
                    that.findLast(s, last);
                }
            }
        }
        
        /** Return an iteration of the AccessPath nodes that correspond to end states. */
        public Iterator findLast() {
            HashSet visited = new HashSet();
            LinkedHashSet last = new LinkedHashSet();
            IdentityHashCodeWrapper ap = IdentityHashCodeWrapper.create(this);
            visited.add(ap);
            if (this._last) last.add(ap);
            this.findLast(visited, last);
            return new FilterIterator(last.iterator(), filter);
        }
        
        /** Helper function for findNode(Node n), below. */
        private AccessPath findNode(Node n, HashSet s) {
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (!s.contains(ap)) {
                    AccessPath p = (AccessPath)ap.getObject();
                    if (n == p._n) return p;
                    s.add(ap);
                    AccessPath q = p.findNode(n, s);
                    if (q != null) return q;
                }
            }
            return null;
        }
        
        /** Find the AccessPath object that corresponds to the given node. */
        public AccessPath findNode(Node n) {
            if (n == this._n) return this;
            HashSet visited = new HashSet();
            IdentityHashCodeWrapper ap = IdentityHashCodeWrapper.create(this);
            visited.add(ap);
            return findNode(n, visited);
        }
        
        /** Set this transition as a valid end transition. */
        private void setLast() { this._last = true; }
        /** Unset this transition as a valid end transition. */
        private void unsetLast() { this._last = false; }
        
        /** Helper function for copy(), below. */
        private void copy(HashMap m, AccessPath that) {
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                AccessPath p = (AccessPath)m.get(ap);
                if (p == null) {
                    AccessPath that2 = (AccessPath)ap.getObject();
                    p = new AccessPath(that2._field, that2._n, that2._last);
                    m.put(ap, p);
                    that2.copy(m, p);
                }
                that.addSuccessor(p);
            }
        }

        /** Return a copy of this (complete) access path. */
        public AccessPath copy() {
            HashMap m = new HashMap();
            IdentityHashCodeWrapper ap = IdentityHashCodeWrapper.create(this);
            AccessPath p = new AccessPath(this._field, this._n, this._last);
            m.put(ap, p);
            this.copy(m, p);
            return p;
        }
        
        /** Helper function for toString(), below. */
        private void toString(StringBuffer sb, HashSet set) {
            if (this._field == null) sb.append("[]");
            else sb.append(this._field.getName());
            if (this._last) sb.append("<e>");
            sb.append("->(");
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper ap = (IdentityHashCodeWrapper)i.next();
                if (set.contains(ap)) {
                    sb.append("<backedge>");
                } else {
                    set.add(ap);
                    ((AccessPath)ap.getObject()).toString(sb, set);
                }
            }
            sb.append(')');
        }
        /** Returns a string representation of this (complete) access path. */
        public String toString() {
            StringBuffer sb = new StringBuffer();
            HashSet visited = new HashSet();
            IdentityHashCodeWrapper ap = IdentityHashCodeWrapper.create(this);
            visited.add(ap);
            toString(sb, visited);
            return sb.toString();
        }
        
        /** Private constructor.  Use the create() methods above. */
        private AccessPath(jq_Field f, Node n, boolean last) {
            this._field = f; this._n = n; this._last = last;
            this.succ = new LinkedHashSet();
        }
        /** Private constructor.  Use the create() methods above. */
        private AccessPath(jq_Field f, Node n) {
            this(f, n, false);
        }
        
        /** Helper function for equals(AccessPath), below. */
        private boolean oneEquals(AccessPath that) {
            //if (this._n != that._n) return false;
            if (this._field != that._field) return false;
            if (this._last != that._last) return false;
            if (this.succ.size() != that.succ.size()) return false;
            return true;
        }
        /** Helper function for equals(AccessPath), below. */
        private boolean equals(AccessPath that, HashSet s) {
            // Relies on the fact that the iterators are stable for equivalent sets.
            // Otherwise, it is an n^2 algorithm.
            for (Iterator i = this.succ.iterator(), j = that.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper a = (IdentityHashCodeWrapper)i.next();
                IdentityHashCodeWrapper b = (IdentityHashCodeWrapper)j.next();
                AccessPath p = (AccessPath)a.getObject();
                AccessPath q = (AccessPath)b.getObject();
                if (!p.oneEquals(q)) return false;
                if (s.contains(a)) continue;
                s.add(a);
                if (!p.equals(q, s)) return false;
            }
            return true;
        }
        /** Returns true if this access path is equal to the given access path. */
        public boolean equals(AccessPath that) {
            HashSet s = new HashSet();
            if (!oneEquals(that)) return false;
            s.add(IdentityHashCodeWrapper.create(this));
            return this.equals(that, s);
        }
        public boolean equals(Object o) {
            if (o instanceof AccessPath) return equals((AccessPath)o);
            return false;
        }
        /** Returns the hashcode for this access path. */
        public int hashCode() {
            int x = this.local_hashCode();
            for (Iterator i = this.succ.iterator(); i.hasNext(); ) {
                IdentityHashCodeWrapper a = (IdentityHashCodeWrapper)i.next();
                x ^= (((AccessPath)a.getObject()).local_hashCode() << 1);
            }
            return x;
        }
        /** Returns the hashcode for this individual AccessPath object. */
        private int local_hashCode() {
            return _field != null ? _field.hashCode() : 0x31337;
        }
        /** Returns the first field of this access path. */
        public jq_Field first() { return _field; }
        /** Returns an iteration of the next AccessPath objects. */
        public Iterator next() {
            return new FilterIterator(succ.iterator(), filter);
        }
        /** A filter to unwrap objects from their IdentityHashCodeWrapper. */
        public static final FilterIterator.Filter filter = new FilterIterator.Filter() {
            public Object map(Object o) { return ((IdentityHashCodeWrapper)o).getObject(); }
        };
    }
    
    /** vvvvv   Actual MethodSummary stuff is below.   vvvvv */
    
    /** The method that this is a summary for. */
    final jq_Method method;
    /** The parameter nodes. */
    final ParamNode[] params;
    /** All nodes in the summary graph. */
    final LinkedHashMap nodes;
    /** All method calls that this method makes. */
    final LinkedHashSet calls;
    /** The returned nodes. */
    final LinkedHashSet returned;
    /** The thrown nodes. */
    final LinkedHashSet thrown;

    MethodSummary(jq_Method method, ParamNode[] param_nodes, GlobalNode my_global, LinkedHashSet methodCalls, LinkedHashSet returned, LinkedHashSet thrown, LinkedHashSet passedAsParameters) {
        this.method = method;
        this.params = param_nodes;
        this.calls = methodCalls;
        this.returned = returned;
        this.thrown = thrown;
        this.nodes = new LinkedHashMap();
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
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
                        addIfUseful(visited, (Node)j.next());
                    }
                }
            }
        }
        if (my_global.addedEdges != null) {
            for (Iterator i=my_global.addedEdges.entrySet().iterator(); i.hasNext(); ) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                Object o = e.getValue();
                if (o instanceof Node) {
                    addAsUseful(visited, (Node)o);
                } else {
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
                        addAsUseful(visited, (Node)j.next());
                    }
                }
            }
        }
        
        if (UNIFY_ACCESS_PATHS) {
            HashSet roots = new HashSet();
            for (int i=0; i<params.length; ++i) {
                if (params[i] == null) continue;
                roots.add(params[i]);
            }
            roots.addAll(returned); roots.addAll(thrown); roots.addAll(passedAsParameters);
            unifyAccessPaths(roots);
        }
    }

    public static final boolean UNIFY_ACCESS_PATHS = false;
    
    private MethodSummary(jq_Method method, ParamNode[] params, LinkedHashSet methodCalls, LinkedHashSet returned, LinkedHashSet thrown, LinkedHashMap nodes) {
        this.method = method;
        this.params = params;
        this.calls = methodCalls;
        this.returned = returned;
        this.thrown = thrown;
        this.nodes = nodes;
    }

    public ParamNode getParamNode(int i) { return params[i]; }
    public LinkedHashSet getCalls() { return calls; }

    /** Add all nodes that are passed as the given passed parameter to the given result set. */
    public void getNodesThatCall(PassedParameter pp, LinkedHashSet result) {
        for (Iterator i = this.nodeIterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if ((n.passedParameters != null) && n.passedParameters.contains(pp))
                result.add(n);
        }
    }

    /** Utility function to add to a multi map. */
    static boolean addToMultiMap(HashMap mm, Object from, Object to) {
        HashSet s = (HashSet)mm.get(from);
        if (s == null) {
            mm.put(from, s = new HashSet());
        }
        return s.add(to);
    }

    /** Utility function to add to a multi map. */
    static boolean addToMultiMap(HashMap mm, Object from, HashSet to) {
        HashSet s = (HashSet)mm.get(from);
        if (s == null) {
            mm.put(from, s = new HashSet());
        }
        return s.addAll(to);
    }

    /** Utility function to get the mapping for a callee node. */
    static HashSet get_mapping(HashMap callee_to_caller, Node callee_n) {
        HashSet s = (HashSet)callee_to_caller.get(callee_n);
        if (s != null) return s;
        s = new HashSet(); s.add(callee_n);
        return s;
    }

    /** Return a deep copy of this analysis summary.
     *  Nodes, edges, everything is copied.
     */
    public MethodSummary copy() {
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
        LinkedHashSet calls = (LinkedHashSet)this.calls.clone();
        LinkedHashSet returned = new LinkedHashSet();
        for (Iterator i=this.returned.iterator(); i.hasNext(); ) {
            returned.add(m.get(i.next()));
        }
        LinkedHashSet thrown = new LinkedHashSet();
        for (Iterator i=this.thrown.iterator(); i.hasNext(); ) {
            thrown.add(m.get(i.next()));
        }
        ParamNode[] params = new ParamNode[this.params.length];
        for (int i=0; i<params.length; ++i) {
            if (this.params[i] == null) continue;
            params[i] = (ParamNode)m.get(this.params[i]);
        }
        LinkedHashMap nodes = new LinkedHashMap();
        for (Iterator i=m.entrySet().iterator(); i.hasNext(); ) {
            java.util.Map.Entry e = (java.util.Map.Entry)i.next();
            nodes.put(e.getValue(), e.getValue());
        }
        return new MethodSummary(method, params, calls, returned, thrown, nodes);
    }

    /** Unify similar access paths from the given roots.
     *  The given set is consumed.
     */
    public void unifyAccessPaths(HashSet roots) {
        LinkedList worklist = new LinkedList();
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            worklist.add(i.next());
        }
        while (!worklist.isEmpty()) {
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
            if (n.addedEdges != null) {
                for (Iterator i=n.addedEdges.entrySet().iterator(); i.hasNext(); ) {
                    java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                    Object o = e.getValue();
                    if (o instanceof Node) {
                        Node n2 = (Node)o;
                        if (roots.contains(n2)) continue;
                        worklist.add(n2); roots.add(n2);
                    } else {
                        LinkedHashSet s = (LinkedHashSet)o;
                        s = (LinkedHashSet)s.clone();
                        for (Iterator j=s.iterator(); j.hasNext(); ) {
                            Object p = j.next();
                            if (roots.contains(p)) j.remove();
                        }
                        if (!s.isEmpty()) {
                            worklist.addAll(s); roots.addAll(s);
                        }
                    }
                }
            }
        }
    }

    /** Unify similar access path edges from the given node.
     */
    public void unifyAccessPathEdges(Node n) {
        if (TRACE_INTRA) out.println("Unifying access path edges from: "+n);
        if (n.accessPathEdges != null) {
            for (Iterator i=n.accessPathEdges.entrySet().iterator(); i.hasNext(); ) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                FieldNode n2;
                if (o instanceof LinkedHashSet) {
                    LinkedHashSet s = (LinkedHashSet)((LinkedHashSet)o).clone();
                    if (s.size() == 0) {
                        i.remove();
                        continue;
                    }
                    if (s.size() == 1) {
                        n2 = (FieldNode)s.iterator().next();
                        e.setValue(n2);
                        continue;
                    }
                    if (TRACE_INTRA) out.println("Node "+n+" has duplicate access path edges on field "+f+": "+s);
                    n2 = FieldNode.unify(f, s);
                    for (Iterator j=s.iterator(); j.hasNext(); ) {
                        FieldNode n3 = (FieldNode)j.next();
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

    /** Instantiate a copy of the callee summary into the caller. */
    public static void instantiate(MethodSummary caller, MethodCall mc, MethodSummary callee) {
        callee = callee.copy();
        //System.out.println("Instantiating "+callee+" into "+caller);
        HashMap callee_to_caller = new HashMap();
        // initialize map with parameters.
        for (int i=0; i<callee.params.length; ++i) {
            ParamNode pn = callee.params[i];
            if (pn == null) continue;
            PassedParameter pp = new PassedParameter(mc, i);
            LinkedHashSet s = new LinkedHashSet();
            caller.getNodesThatCall(pp, s);
            callee_to_caller.put(pn, s);
        }
        for (int ii=0; ii<callee.params.length; ++ii) {
            ParamNode pn = callee.params[ii];
            if (pn == null) continue;
            LinkedHashSet s = (LinkedHashSet)callee_to_caller.get(pn);
            pn.replaceBy(s);
            if (callee.returned.contains(pn)) {
                callee.returned.remove(pn); callee.returned.addAll(s);
            }
            if (callee.thrown.contains(pn)) {
                callee.thrown.remove(pn); callee.thrown.addAll(s);
            }
        }
        ReturnValueNode rvn = new ReturnValueNode(mc);
        rvn = (ReturnValueNode)caller.nodes.get(rvn);
        if (rvn != null) {
            rvn.replaceBy(callee.returned);
            if (caller.returned.contains(rvn)) {
                caller.returned.remove(rvn); caller.returned.addAll(callee.returned);
            }
            if (caller.thrown.contains(rvn)) {
                caller.thrown.remove(rvn); caller.thrown.addAll(callee.returned);
            }
        }
        ThrownExceptionNode ten = new ThrownExceptionNode(mc);
        ten = (ThrownExceptionNode)caller.nodes.get(ten);
        if (ten != null) {
            ten.replaceBy(callee.thrown);
            if (caller.returned.contains(ten)) {
                caller.returned.remove(ten); caller.returned.addAll(callee.thrown);
            }
            if (caller.thrown.contains(ten)) {
                caller.thrown.remove(ten); caller.thrown.addAll(callee.thrown);
            }
        }
        HashSet s = new HashSet();
        s.addAll(callee.returned);
        s.addAll(callee.thrown);
        for (int ii=0; ii<callee.params.length; ++ii) {
            ParamNode pn = callee.params[ii];
            if (pn == null) continue;
            HashSet t = (HashSet)callee_to_caller.get(pn);
            s.addAll(t);
        }
        caller.unifyAccessPaths(s);
    }

    public jq_Method getMethod() { return method; }
    
    public static final String lineSep = System.getProperty("line.separator");
    
    /** Return a string representation of this summary. */
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Summary for ");
        sb.append(method.toString());
        sb.append(':');
        sb.append(lineSep);
        for (Iterator i=nodes.keySet().iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            sb.append(n.toString_long());
            sb.append(lineSep);
        }
        if (returned != null && !returned.isEmpty()) {
            sb.append("Returned: ");
            sb.append(returned);
            sb.append(lineSep);
        }
        if (thrown != null && !thrown.isEmpty()) {
            sb.append("Thrown: ");
            sb.append(thrown);
            sb.append(lineSep);
        }
        return sb.toString();
    }

    /** Utility function to add the given node to the node set if it is useful,
     *  and transitively for other nodes. */
    private boolean addIfUseful(HashSet visited, Node n) {
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
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
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
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
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
    /** Utility function to add the given node to the node set as useful,
     *  and transitively for other nodes. */
    private void addAsUseful(HashSet visited, Node n) {
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
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
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
                    for (Iterator j=((LinkedHashSet)o).iterator(); j.hasNext(); ) {
                        Node j_n = (Node)j.next();
                        if (!addIfUseful(visited, j_n)) {
                            j.remove();
                        }
                    }
                }
            }
        }
    }

    /** Returns an iteration of all nodes in this summary. */
    public Iterator nodeIterator() { return nodes.keySet().iterator(); }

}

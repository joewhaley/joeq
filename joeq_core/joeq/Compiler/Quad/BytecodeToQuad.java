/*
 * BytecodeToQuad.java
 *
 * Created on April 22, 2001, 11:53 AM
 *
 */

package Compil3r.Quad;
import Clazz.*;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.ControlFlowGraph.InitialPass;
import Compil3r.Quad.Operator.Move;
import Compil3r.Quad.Operator.Binary;
import Compil3r.Quad.Operator.Unary;
import Compil3r.Quad.Operator.ALoad;
import Compil3r.Quad.Operator.AStore;
import Compil3r.Quad.Operator.IntIfCmp;
import Compil3r.Quad.Operator.Goto;
import Compil3r.Quad.Operator.TableSwitch;
import Compil3r.Quad.Operator.LookupSwitch;
import Compil3r.Quad.Operator.Return;
import Compil3r.Quad.Operator.Getstatic;
import Compil3r.Quad.Operator.Putstatic;
import Compil3r.Quad.Operator.Getfield;
import Compil3r.Quad.Operator.Putfield;
import Compil3r.Quad.Operator.NullCheck;
import Compil3r.Quad.Operator.ZeroCheck;
import Compil3r.Quad.Operator.BoundsCheck;
import Compil3r.Quad.Operator.StoreCheck;
import Compil3r.Quad.Operator.Invoke;
import Compil3r.Quad.Operator.New;
import Compil3r.Quad.Operator.NewArray;
import Compil3r.Quad.Operator.CheckCast;
import Compil3r.Quad.Operator.InstanceOf;
import Compil3r.Quad.Operator.ALength;
import Compil3r.Quad.Operator.Monitor;
import Compil3r.Quad.Operator.MemLoad;
import Compil3r.Quad.Operator.MemStore;
import Compil3r.Quad.Operator.Special;
import Compil3r.Quad.Operator.Jsr;
import Compil3r.Quad.Operator.Ret;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operand.ConditionOperand;
import Compil3r.Quad.Operand.TargetOperand;
import Compil3r.Quad.Operand.FieldOperand;
import Compil3r.Quad.Operand.TypeOperand;
import Compil3r.Quad.Operand.MethodOperand;
import Compil3r.Quad.Operand.IConstOperand;
import Compil3r.Quad.Operand.FConstOperand;
import Compil3r.Quad.Operand.LConstOperand;
import Compil3r.Quad.Operand.DConstOperand;
import Compil3r.Quad.Operand.AConstOperand;
import Compil3r.Quad.Operand.UnnecessaryGuardOperand;
import Compil3r.Quad.RegisterFactory.Register;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;
import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;
import jq;

/**
 * Converts stack-based Java bytecode to Quad intermediate format.
 * This utilizes the ControlFlowGraph in the BytecodeAnalysis package to build
 * up a control flow graph, then iterates over the graph to generate the Quad
 * code.
 *
 * @see  BytecodeVisitor
 * @see  BytecodeAnalysis.ControlFlowGraph
 * @author  John Whaley
 * @version  @version $Id$
 */

public class BytecodeToQuad extends BytecodeVisitor {
    
    private ControlFlowGraph quad_cfg;
    private BasicBlock quad_bb;
    private Compil3r.BytecodeAnalysis.ControlFlowGraph bc_cfg;
    private Compil3r.BytecodeAnalysis.BasicBlock bc_bb;
    private BasicBlock[] quad_bbs;
    private RegisterFactory rf;
    
    private boolean uncond_branch;

    public static boolean ALWAYS_TRACE = false;

    /** Initializes the conversion from bytecode to quad format for the given method.
     * @param  method the method to convert. */
    public BytecodeToQuad(jq_Method method) {
        super(method);
        TRACE = ALWAYS_TRACE;
    }
    
    /** Returns a string with the name of the pass and the method being converted.
     * @return  a string with the name of the pass and the method being converted. */
    public String toString() {
        return "BC2Q/"+jq.left(method.getName().toString(), 10);
    }
    /** Perform conversion process from bytecode to quad.
     * @return  the control flow graph of the resulting quad representation. */
    public ControlFlowGraph convert() {
        bc_cfg = Compil3r.BytecodeAnalysis.ControlFlowGraph.computeCFG(method);
        
        // copy bytecode cfg to quad cfg
        jq_TryCatchBC[] exs = method.getExceptionTable();
        this.quad_cfg = new ControlFlowGraph(bc_cfg.getExit().getNumberOfPredecessors(),
                                                 exs.length);
        quad_bbs = new BasicBlock[bc_cfg.getNumberOfBasicBlocks()];
        quad_bbs[0] = this.quad_cfg.entry();
        quad_bbs[1] = this.quad_cfg.exit();
        for (int i=2; i<quad_bbs.length; ++i) {
            Compil3r.BytecodeAnalysis.BasicBlock bc_bb = bc_cfg.getBasicBlock(i);
            int n_pred = bc_bb.getNumberOfPredecessors();
            int n_succ = bc_bb.getNumberOfSuccessors();
            int n_inst = bc_bb.getEnd() - bc_bb.getStart() + 1; // estimate
            quad_bbs[i] = BasicBlock.createBasicBlock(i, n_pred, n_succ, n_inst);
        }
	this.quad_cfg.updateBBcounter(quad_bbs.length);

        // add exception handlers.
        for (int i=exs.length-1; i>=0; --i) {
            jq_TryCatchBC ex = exs[i];
            Compil3r.BytecodeAnalysis.BasicBlock bc_bb = bc_cfg.getBasicBlockByBytecodeIndex(ex.getStartPC());
            jq.assert(bc_bb.getStart() < ex.getEndPC());
            BasicBlock ex_handler = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(ex.getHandlerPC()).id];
            ex_handler.setExceptionHandlerEntry();
            int numOfProtectedBlocks = (ex.getEndPC()==method.getBytecode().length?quad_bbs.length:bc_cfg.getBasicBlockByBytecodeIndex(ex.getEndPC()).id) - bc_bb.id;
            ExceptionHandler eh = new ExceptionHandler(ex.getExceptionType(), numOfProtectedBlocks, ex_handler);
            ExceptionHandlerList ehs = new ExceptionHandlerList(eh, null);
            BasicBlock bb = quad_bbs[bc_bb.id];
            bb.addExceptionHandler_first(ehs);
            while (bc_bb.getStart() < ex.getEndPC()) {
                eh.addHandledBasicBlock(bb);
                ehs = bb.addExceptionHandler(ehs);
                bc_bb = bc_cfg.getBasicBlock(bc_bb.id+1);
                bb = quad_bbs[bc_bb.id];
            }
        }
        this.start_states = new AbstractState[quad_bbs.length];
        for (int i=0; i<quad_bbs.length; ++i) {
            Compil3r.BytecodeAnalysis.BasicBlock bc_bb = bc_cfg.getBasicBlock(i);
            BasicBlock bb = quad_bbs[i];
            for (int j=0; j<bc_bb.getNumberOfPredecessors(); ++j) {
                bb.addPredecessor(quad_bbs[bc_bb.getPredecessor(j).id]);
            }
            for (int j=0; j<bc_bb.getNumberOfSuccessors(); ++j) {
                bb.addSuccessor(quad_bbs[bc_bb.getSuccessor(j).id]);
            }
            // --> start state allocated on demand in merge
            //this.start_states[i] = new AbstractState(max_stack, max_locals);
        }

        // initialize register factory
        this.rf = new RegisterFactory(method);
        
        // initialize start state
        this.start_states[2] = AbstractState.allocateInitialState(rf, method);
        this.current_state = AbstractState.allocateEmptyState(method);
        
        // traverse reverse post-order over basic blocks to generate instructions
        Compil3r.BytecodeAnalysis.ControlFlowGraph.RPOBasicBlockIterator rpo = bc_cfg.reversePostOrderIterator();
        Compil3r.BytecodeAnalysis.BasicBlock first_bb = rpo.nextBB();
        jq.assert(first_bb == bc_cfg.getEntry());
        while (rpo.hasNext()) {
            Compil3r.BytecodeAnalysis.BasicBlock bc_bb = rpo.nextBB();
            this.traverseBB(bc_bb);
        }
        
        return this.quad_cfg;
    }

    private boolean endBasicBlock;
    
    /**
     * @param  bc_bb  */
    public void traverseBB(Compil3r.BytecodeAnalysis.BasicBlock bc_bb) {
        if (start_states[bc_bb.id] == null) {
            // unreachable block!
            if (TRACE) out.println("Basic block "+bc_bb+" is unreachable!");
            return;
        }
        if (bc_bb.getStart() == -1) {
            return; // entry or exit
        }
        if (TRACE) out.println("Visiting "+bc_bb);
        this.quad_bb = quad_bbs[bc_bb.id];
        this.bc_bb = bc_bb;
        this.uncond_branch = false;
        this.current_state.overwriteWith(start_states[bc_bb.id]);
	if (TRACE) this.current_state.dumpState();
        this.endBasicBlock = false;
        for (i_end=bc_bb.getStart()-1; ; ) {
            i_start = i_end+1;
            if (isEndOfBB()) break;
            this.visitBytecode();
        }
        for (int i=0; i<bc_bb.getNumberOfSuccessors(); ++i) {
            this.mergeStateWith(bc_bb.getSuccessor(i));
        }
    }
    
    private boolean isEndOfBB() {
        return i_start > bc_bb.getEnd() || endBasicBlock;
    }
    
    private void mergeStateWith(Compil3r.BytecodeAnalysis.BasicBlock bc_bb) {
        if (start_states[bc_bb.id] == null) {
	    if (TRACE) System.out.println("Copying current state to "+bc_bb);
            start_states[bc_bb.id] = current_state.copy();
        } else {
	    if (TRACE) System.out.println("Merging current state with "+bc_bb);
            start_states[bc_bb.id].merge(current_state, rf);
        }
    }

    private void replaceLocalsOnStack(int index, jq_Type type) {
        for (int i=0; i<current_state.getStackSize(); ++i) {
            Operand op = current_state.peekStack(i);
            if (rf.isLocal(op, index, type)) {
                RegisterOperand rop = (RegisterOperand)op;
                RegisterOperand t = getStackRegister(type, i);
                t.setFlags(rop.getFlags()); t.scratchObject = rop.scratchObject;
                Quad q = Move.create(Move.getMoveOp(type), t, rop);
                appendQuad(q);
                current_state.pokeStack(i, t.copy());
            }
        }
    }
    
    void appendQuad(Quad q) {
        if (TRACE) System.out.println(q.toString());
        quad_bb.appendQuad(q);
    }
    
    RegisterOperand getStackRegister(jq_Type type, int i) {
        return new RegisterOperand(rf.getStack(current_state.getStackSize()-i-1, type), type);
    }
    
    RegisterOperand getStackRegister(jq_Type type) {
        return getStackRegister(type, -1);
    }

    RegisterOperand makeLocal(int i, jq_Type type) {
        return new RegisterOperand(rf.getLocal(i, type), type);
    }
    
    RegisterOperand makeLocal(int i, RegisterOperand rop) {
        jq_Type type = rop.getType();
        return new RegisterOperand(rf.getLocal(i, type), type, rop.getFlags());
    }
    
    static boolean hasGuard(RegisterOperand rop) { return rop.scratchObject != null; }
    static void setGuard(RegisterOperand rop, Operand guard) { rop.scratchObject = guard; }
    
    static Operand getGuard(Operand op) {
        if (op instanceof RegisterOperand) {
            RegisterOperand rop = (RegisterOperand)op;
            return (Operand)rop.scratchObject;
        }
        jq.assert(op instanceof AConstOperand);
        return new UnnecessaryGuardOperand();
    }
    
    Operand currentGuard;
    
    void setCurrentGuard(Operand guard) { currentGuard = guard; }
    void clearCurrentGuard() { currentGuard = null; }
    Operand getCurrentGuard() { return currentGuard.copy(); }
    
    private AbstractState[] start_states;
    private AbstractState current_state;
    
    public void visitNOP() {
        super.visitNOP();
        // do nothing
    }
    public void visitACONST(Object s) {
        super.visitACONST(s);
        current_state.push_A(new AConstOperand(s));
    }
    public void visitICONST(int c) {
        super.visitICONST(c);
        current_state.push_I(new IConstOperand(c));
    }
    public void visitLCONST(long c) {
        super.visitLCONST(c);
        current_state.push_L(new LConstOperand(c));
    }
    public void visitFCONST(float c) {
        super.visitFCONST(c);
        current_state.push_F(new FConstOperand(c));
    }
    public void visitDCONST(double c) {
        super.visitDCONST(c);
        current_state.push_D(new DConstOperand(c));
    }
    public void visitILOAD(int i) {
        super.visitILOAD(i);
        current_state.push_I(current_state.getLocal_I(i));
    }
    public void visitLLOAD(int i) {
        super.visitLLOAD(i);
        current_state.push_L(current_state.getLocal_L(i));
    }
    public void visitFLOAD(int i) {
        super.visitFLOAD(i);
        current_state.push_F(current_state.getLocal_F(i));
    }
    public void visitDLOAD(int i) {
        super.visitDLOAD(i);
        current_state.push_D(current_state.getLocal_D(i));
    }
    public void visitALOAD(int i) {
        super.visitALOAD(i);
        current_state.push_A(current_state.getLocal_A(i));
    }
    private void STOREhelper(int i, jq_Type type) {
        replaceLocalsOnStack(i, type);
        Operand op1 = current_state.pop(type);
        Operand local_value;
        RegisterOperand op0;
        if (op1 instanceof RegisterOperand) {
            // move from one local variable to another.
            local_value = op0 = makeLocal(i, (RegisterOperand)op1); // copy attributes.
        } else {
            // move a constant to a local variable.
            local_value = op1;
            op0 = makeLocal(i, type);
        }
        if (type.getReferenceSize() == 8) current_state.setLocalDual(i, local_value);
        else current_state.setLocal(i, local_value);
        Quad q = Move.create(Move.getMoveOp(type), op0, op1);
        appendQuad(q);
    }
    public void visitISTORE(int i) {
        super.visitISTORE(i);
        STOREhelper(i, jq_Primitive.INT);
    }
    public void visitLSTORE(int i) {
        super.visitLSTORE(i);
        STOREhelper(i, jq_Primitive.LONG);
    }
    public void visitFSTORE(int i) {
        super.visitFSTORE(i);
        STOREhelper(i, jq_Primitive.FLOAT);
    }
    public void visitDSTORE(int i) {
        super.visitDSTORE(i);
        STOREhelper(i, jq_Primitive.DOUBLE);
    }
    public void visitASTORE(int i) {
        super.visitASTORE(i);
        STOREhelper(i, PrimordialClassLoader.loader.getJavaLangObject());
    }
    private void ALOADhelper(ALoad operator, jq_Type t) {
        Operand index = current_state.pop_I();
        Operand ref = current_state.pop_A();
        clearCurrentGuard();
        if (performNullCheck(ref)) {
	    if (TRACE) System.out.println("Null check triggered on "+ref);
	    return;
	}
	if (performBoundsCheck(ref, index)) {
	    if (TRACE) System.out.println("Bounds check triggered on "+ref+" "+index);
	    return;
	}
        if (t.isReferenceType()) {
            // refine type.
            t = getArrayTypeOf(ref).getElementType();
        }
        RegisterOperand r = getStackRegister(t);
        Quad q = ALoad.create(operator, r, ref, index, getCurrentGuard());
        appendQuad(q);
        current_state.push(r.copy(), t);
    }
    public void visitIALOAD() {
        super.visitIALOAD();
        ALOADhelper(ALoad.ALOAD_I.INSTANCE, jq_Primitive.INT);
    }
    public void visitLALOAD() {
        super.visitLALOAD();
        ALOADhelper(ALoad.ALOAD_L.INSTANCE, jq_Primitive.LONG);
    }
    public void visitFALOAD() {
        super.visitFALOAD();
        ALOADhelper(ALoad.ALOAD_F.INSTANCE, jq_Primitive.FLOAT);
    }
    public void visitDALOAD() {
        super.visitDALOAD();
        ALOADhelper(ALoad.ALOAD_D.INSTANCE, jq_Primitive.DOUBLE);
    }
    public void visitAALOAD() {
        super.visitAALOAD();
        ALOADhelper(ALoad.ALOAD_A.INSTANCE, PrimordialClassLoader.loader.getJavaLangObject());
    }
    public void visitBALOAD() {
        super.visitBALOAD();
        ALOADhelper(ALoad.ALOAD_B.INSTANCE, jq_Primitive.BYTE);
    }
    public void visitCALOAD() {
        super.visitCALOAD();
        ALOADhelper(ALoad.ALOAD_C.INSTANCE, jq_Primitive.CHAR);
    }
    public void visitSALOAD() {
        super.visitSALOAD();
        ALOADhelper(ALoad.ALOAD_S.INSTANCE, jq_Primitive.SHORT);
    }
    private void ASTOREhelper(AStore operator, jq_Type t) {
        Operand val = current_state.pop(t);
        Operand index = current_state.pop_I();
        Operand ref = current_state.pop_A();
        clearCurrentGuard();
        if (performNullCheck(ref)) {
	    if (TRACE) System.out.println("Null check triggered on "+ref);
	    return;
	}
	if (performBoundsCheck(ref, index)) {
	    if (TRACE) System.out.println("Bounds check triggered on "+ref+" "+index);
	    return;
	}
        if (t.isReferenceType()) {
            // refine type and perform checkstore
            if (performCheckStore((RegisterOperand)ref, val)) return;
        }
        Quad q = AStore.create(operator, val, ref, index, getCurrentGuard());
        appendQuad(q);
    }
    public void visitIASTORE() {
        super.visitIASTORE();
        ASTOREhelper(AStore.ASTORE_I.INSTANCE, jq_Primitive.INT);
    }
    public void visitLASTORE() {
        super.visitLASTORE();
        ASTOREhelper(AStore.ASTORE_L.INSTANCE, jq_Primitive.LONG);
    }
    public void visitFASTORE() {
        super.visitFASTORE();
        ASTOREhelper(AStore.ASTORE_F.INSTANCE, jq_Primitive.FLOAT);
    }
    public void visitDASTORE() {
        super.visitDASTORE();
        ASTOREhelper(AStore.ASTORE_D.INSTANCE, jq_Primitive.DOUBLE);
    }
    public void visitAASTORE() {
        super.visitAASTORE();
        ASTOREhelper(AStore.ASTORE_A.INSTANCE, PrimordialClassLoader.loader.getJavaLangObject());
    }
    public void visitBASTORE() {
        super.visitBASTORE();
        ASTOREhelper(AStore.ASTORE_B.INSTANCE, jq_Primitive.BYTE);
    }
    public void visitCASTORE() {
        super.visitCASTORE();
        ASTOREhelper(AStore.ASTORE_C.INSTANCE, jq_Primitive.CHAR);
    }
    public void visitSASTORE() {
        super.visitSASTORE();
        ASTOREhelper(AStore.ASTORE_S.INSTANCE, jq_Primitive.SHORT);
    }
    public void visitPOP() {
        super.visitPOP();
        current_state.pop();
    }
    public void visitPOP2() {
        super.visitPOP2();
        current_state.pop(); current_state.pop();
    }
    public void visitDUP() {
        super.visitDUP();
        Operand op = current_state.pop();
        current_state.push(op);
        current_state.push(op.copy());
    }
    public void visitDUP_x1() {
        super.visitDUP_x1();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        current_state.push(op1);
        current_state.push(op2);
        current_state.push(op1.copy());
    }
    public void visitDUP_x2() {
        super.visitDUP_x2();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        Operand op3 = current_state.pop();
        current_state.push(op1);
        current_state.push(op3);
        current_state.push(op2);
        current_state.push(op1.copy());
    }
    public void visitDUP2() {
        super.visitDUP2();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        current_state.push(op2);
        current_state.push(op1);
        current_state.push(op2.copy());
        current_state.push(op1.copy());
    }
    public void visitDUP2_x1() {
        super.visitDUP2_x1();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        Operand op3 = current_state.pop();
        current_state.push(op2);
        current_state.push(op1);
        current_state.push(op3);
        current_state.push(op2.copy());
        current_state.push(op1.copy());
    }
    public void visitDUP2_x2() {
        super.visitDUP2_x2();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        Operand op3 = current_state.pop();
        Operand op4 = current_state.pop();
        current_state.push(op2);
        current_state.push(op1);
        current_state.push(op4);
        current_state.push(op3);
        current_state.push(op2.copy());
        current_state.push(op1.copy());
    }
    public void visitSWAP() {
        super.visitSWAP();
        Operand op1 = current_state.pop();
        Operand op2 = current_state.pop();
        current_state.push(op1);
        current_state.push(op2);
    }
    private void BINOPhelper(Binary operator, jq_Type tr, jq_Type t1, jq_Type t2, boolean zero_check) {
        Operand op2 = current_state.pop(t2);
        Operand op1 = current_state.pop(t1);
        if (zero_check && performZeroCheck(op2)) {
            if (TRACE) System.out.println("Zero check triggered on "+op2);
            return;
        }
        RegisterOperand r = getStackRegister(tr);
        Quad q = Binary.create(operator, r, op1, op2);
        appendQuad(q);
        current_state.push(r.copy(), tr);
    }
    public void visitIBINOP(byte op) {
        super.visitIBINOP(op);
        Binary operator=null; boolean zero_check = false;
        switch (op) {
            case BINOP_ADD: operator = Binary.ADD_I.INSTANCE; break;
            case BINOP_SUB: operator = Binary.SUB_I.INSTANCE; break;
            case BINOP_MUL: operator = Binary.MUL_I.INSTANCE; break;
            case BINOP_DIV: operator = Binary.DIV_I.INSTANCE; zero_check = true; break;
            case BINOP_REM: operator = Binary.REM_I.INSTANCE; zero_check = true; break;
            case BINOP_AND: operator = Binary.AND_I.INSTANCE; break;
            case BINOP_OR: operator = Binary.OR_I.INSTANCE; break;
            case BINOP_XOR: operator = Binary.XOR_I.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.INT, jq_Primitive.INT, jq_Primitive.INT, zero_check);
    }
    public void visitLBINOP(byte op) {
        super.visitLBINOP(op);
        Binary operator=null; boolean zero_check = false;
        switch (op) {
            case BINOP_ADD: operator = Binary.ADD_L.INSTANCE; break;
            case BINOP_SUB: operator = Binary.SUB_L.INSTANCE; break;
            case BINOP_MUL: operator = Binary.MUL_L.INSTANCE; break;
            case BINOP_DIV: operator = Binary.DIV_L.INSTANCE; zero_check = true; break;
            case BINOP_REM: operator = Binary.REM_L.INSTANCE; zero_check = true; break;
            case BINOP_AND: operator = Binary.AND_L.INSTANCE; break;
            case BINOP_OR: operator = Binary.OR_L.INSTANCE; break;
            case BINOP_XOR: operator = Binary.XOR_L.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.LONG, jq_Primitive.LONG, jq_Primitive.LONG, zero_check);
    }
    public void visitFBINOP(byte op) {
        super.visitFBINOP(op);
        Binary operator=null;
        switch (op) {
            case BINOP_ADD: operator = Binary.ADD_F.INSTANCE; break;
            case BINOP_SUB: operator = Binary.SUB_F.INSTANCE; break;
            case BINOP_MUL: operator = Binary.MUL_F.INSTANCE; break;
            case BINOP_DIV: operator = Binary.DIV_F.INSTANCE; break;
            case BINOP_REM: operator = Binary.REM_F.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.FLOAT, jq_Primitive.FLOAT, jq_Primitive.FLOAT, false);
    }
    public void visitDBINOP(byte op) {
        super.visitDBINOP(op);
        Binary operator=null;
        switch (op) {
            case BINOP_ADD: operator = Binary.ADD_D.INSTANCE; break;
            case BINOP_SUB: operator = Binary.SUB_D.INSTANCE; break;
            case BINOP_MUL: operator = Binary.MUL_D.INSTANCE; break;
            case BINOP_DIV: operator = Binary.DIV_D.INSTANCE; break;
            case BINOP_REM: operator = Binary.REM_D.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.DOUBLE, jq_Primitive.DOUBLE, jq_Primitive.DOUBLE, false);
    }
    public void UNOPhelper(Unary operator, jq_Type tr, jq_Type t1) {
        Operand op1 = current_state.pop(t1);
        RegisterOperand r = getStackRegister(tr);
        Quad q = Unary.create(operator, r, op1);
        appendQuad(q);
        current_state.push(r.copy(), tr);
    }
    public void visitIUNOP(byte op) {
        super.visitIUNOP(op);
        Unary operator=null;
        switch (op) {
            case UNOP_NEG: operator = Unary.NEG_I.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        UNOPhelper(operator, jq_Primitive.INT, jq_Primitive.INT);
    }
    public void visitLUNOP(byte op) {
        super.visitLUNOP(op);
        Unary operator=null;
        switch (op) {
            case UNOP_NEG: operator = Unary.NEG_L.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        UNOPhelper(operator, jq_Primitive.LONG, jq_Primitive.LONG);
    }
    public void visitFUNOP(byte op) {
        super.visitFUNOP(op);
        Unary operator=null;
        switch (op) {
            case UNOP_NEG: operator = Unary.NEG_F.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        UNOPhelper(operator, jq_Primitive.FLOAT, jq_Primitive.FLOAT);
    }
    public void visitDUNOP(byte op) {
        super.visitDUNOP(op);
        Unary operator=null;
        switch (op) {
            case UNOP_NEG: operator = Unary.NEG_D.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        UNOPhelper(operator, jq_Primitive.DOUBLE, jq_Primitive.DOUBLE);
    }
    public void visitISHIFT(byte op) {
        super.visitISHIFT(op);
        Binary operator=null;
        switch (op) {
            case SHIFT_LEFT: operator = Binary.SHL_I.INSTANCE; break;
            case SHIFT_RIGHT: operator = Binary.SHR_I.INSTANCE; break;
            case SHIFT_URIGHT: operator = Binary.USHR_I.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.INT, jq_Primitive.INT, jq_Primitive.INT, false);
    }
    public void visitLSHIFT(byte op) {
        super.visitLSHIFT(op);
        Binary operator=null;
        switch (op) {
            case SHIFT_LEFT: operator = Binary.SHL_L.INSTANCE; break;
            case SHIFT_RIGHT: operator = Binary.SHR_L.INSTANCE; break;
            case SHIFT_URIGHT: operator = Binary.USHR_L.INSTANCE; break;
            default: jq.UNREACHABLE(); break;
        }
        BINOPhelper(operator, jq_Primitive.LONG, jq_Primitive.LONG, jq_Primitive.INT, false);
    }
    public void visitIINC(int i, int v) {
        super.visitIINC(i, v);
        Operand op1 = current_state.getLocal_I(i);
        replaceLocalsOnStack(i, jq_Primitive.INT);
        RegisterOperand op0 = makeLocal(i, jq_Primitive.INT);
        Quad q = Binary.create(Binary.ADD_I.INSTANCE, op0, op1, new IConstOperand(v));
        appendQuad(q);
        current_state.setLocal(i, op0);
    }
    public void visitI2L() {
        super.visitI2L();
        UNOPhelper(Unary.INT_2LONG.INSTANCE, jq_Primitive.LONG, jq_Primitive.INT);
    }
    public void visitI2F() {
        super.visitI2F();
        UNOPhelper(Unary.INT_2FLOAT.INSTANCE, jq_Primitive.FLOAT, jq_Primitive.INT);
    }
    public void visitI2D() {
        super.visitI2D();
        UNOPhelper(Unary.INT_2DOUBLE.INSTANCE, jq_Primitive.DOUBLE, jq_Primitive.INT);
    }
    public void visitL2I() {
        super.visitL2I();
        UNOPhelper(Unary.LONG_2INT.INSTANCE, jq_Primitive.INT, jq_Primitive.LONG);
    }
    public void visitL2F() {
        super.visitL2F();
        UNOPhelper(Unary.LONG_2FLOAT.INSTANCE, jq_Primitive.FLOAT, jq_Primitive.LONG);
    }
    public void visitL2D() {
        super.visitL2D();
        UNOPhelper(Unary.LONG_2DOUBLE.INSTANCE, jq_Primitive.DOUBLE, jq_Primitive.LONG);
    }
    public void visitF2I() {
        super.visitF2I();
        UNOPhelper(Unary.FLOAT_2INT.INSTANCE, jq_Primitive.INT, jq_Primitive.FLOAT);
    }
    public void visitF2L() {
        super.visitF2L();
        UNOPhelper(Unary.FLOAT_2LONG.INSTANCE, jq_Primitive.LONG, jq_Primitive.FLOAT);
    }
    public void visitF2D() {
        super.visitF2D();
        UNOPhelper(Unary.FLOAT_2DOUBLE.INSTANCE, jq_Primitive.DOUBLE, jq_Primitive.FLOAT);
    }
    public void visitD2I() {
        super.visitD2I();
        UNOPhelper(Unary.DOUBLE_2INT.INSTANCE, jq_Primitive.INT, jq_Primitive.DOUBLE);
    }
    public void visitD2L() {
        super.visitD2L();
        UNOPhelper(Unary.DOUBLE_2LONG.INSTANCE, jq_Primitive.LONG, jq_Primitive.DOUBLE);
    }
    public void visitD2F() {
        super.visitD2F();
        UNOPhelper(Unary.DOUBLE_2FLOAT.INSTANCE, jq_Primitive.FLOAT, jq_Primitive.DOUBLE);
    }
    public void visitI2B() {
        super.visitI2B();
        UNOPhelper(Unary.INT_2BYTE.INSTANCE, jq_Primitive.BYTE, jq_Primitive.INT);
    }
    public void visitI2C() {
        super.visitI2C();
        UNOPhelper(Unary.INT_2CHAR.INSTANCE, jq_Primitive.CHAR, jq_Primitive.INT);
    }
    public void visitI2S() {
        super.visitI2S();
        UNOPhelper(Unary.INT_2SHORT.INSTANCE, jq_Primitive.SHORT, jq_Primitive.INT);
    }
    public void visitLCMP2() {
        super.visitLCMP2();
        BINOPhelper(Binary.CMP_L.INSTANCE, jq_Primitive.INT, jq_Primitive.LONG, jq_Primitive.LONG, false);
    }
    public void visitFCMP2(byte op) {
        super.visitFCMP2(op);
        BINOPhelper(Binary.CMP_F.INSTANCE, jq_Primitive.INT, jq_Primitive.FLOAT, jq_Primitive.FLOAT, false);
    }
    public void visitDCMP2(byte op) {
        super.visitDCMP2(op);
        BINOPhelper(Binary.CMP_D.INSTANCE, jq_Primitive.INT, jq_Primitive.DOUBLE, jq_Primitive.DOUBLE, false);
    }
    public void visitIF(byte op, int target) {
        super.visitIF(op, target);
        Operand op0 = current_state.pop_I();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        ConditionOperand cond = new ConditionOperand(op);
        Quad q = IntIfCmp.create(IntIfCmp.IFCMP_I.INSTANCE, op0, new IConstOperand(0), cond, new TargetOperand(target_bb));
        appendQuad(q);
    }
    public void visitIFREF(byte op, int target) {
        super.visitIFREF(op, target);
        Operand op0 = current_state.pop_A();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        ConditionOperand cond = new ConditionOperand(op);
        Quad q = IntIfCmp.create(IntIfCmp.IFCMP_A.INSTANCE, op0, new AConstOperand(null), cond, new TargetOperand(target_bb));
        appendQuad(q);
    }
    public void visitIFCMP(byte op, int target) {
        super.visitIFCMP(op, target);
        Operand op1 = current_state.pop_I();
        Operand op0 = current_state.pop_I();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        ConditionOperand cond = new ConditionOperand(op);
        Quad q = IntIfCmp.create(IntIfCmp.IFCMP_I.INSTANCE, op0, op1, cond, new TargetOperand(target_bb));
        appendQuad(q);
    }
    public void visitIFREFCMP(byte op, int target) {
        super.visitIFREFCMP(op, target);
        Operand op1 = current_state.pop_A();
        Operand op0 = current_state.pop_A();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        ConditionOperand cond = new ConditionOperand(op);
        Quad q = IntIfCmp.create(IntIfCmp.IFCMP_I.INSTANCE, op0, op1, cond, new TargetOperand(target_bb));
        appendQuad(q);
    }
    public void visitGOTO(int target) {
        super.visitGOTO(target);
        this.uncond_branch = true;
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        Quad q = Goto.create(Goto.GOTO.INSTANCE, new TargetOperand(target_bb));
        appendQuad(q);
    }
    public void visitJSR(int target) {
        super.visitJSR(target);
        this.uncond_branch = true;
	BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(target).id];
        RegisterOperand op0 = getStackRegister(jq_ReturnAddressType.INSTANCE);
        Quad q = Jsr.create(Jsr.JSR.INSTANCE, op0, new TargetOperand(target_bb));
        appendQuad(q);
        current_state.push(op0.copy());
    }
    public void visitRET(int i) {
        super.visitRET(i);
        this.uncond_branch = true;	
        RegisterOperand op0 = makeLocal(i, jq_ReturnAddressType.INSTANCE);
        Quad q = Ret.create(Ret.RET.INSTANCE, op0);
        appendQuad(q);
        current_state.setLocal(i, null);
    }
    public void visitTABLESWITCH(int default_target, int low, int high, int[] targets) {
        super.visitTABLESWITCH(default_target, low, high, targets);
        this.uncond_branch = true;
        Operand op0 = current_state.pop_I();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(default_target).id];
        jq.assert(high-low+1 == targets.length);
        Quad q = TableSwitch.create(TableSwitch.TABLESWITCH.INSTANCE, op0, new IConstOperand(low),
                                    new TargetOperand(target_bb), targets.length);
        for (int i = 0; i < targets.length; ++i) {
            target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(targets[i]).id];
            TableSwitch.setTarget(q, i, target_bb);
        }
        appendQuad(q);
    }
    public void visitLOOKUPSWITCH(int default_target, int[] values, int[] targets) {
        super.visitLOOKUPSWITCH(default_target, values, targets);
        this.uncond_branch = true;
        Operand op0 = current_state.pop_I();
        BasicBlock target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(default_target).id];
        Quad q = LookupSwitch.create(LookupSwitch.LOOKUPSWITCH.INSTANCE, op0, new TargetOperand(target_bb), values.length);
        for (int i = 0; i < values.length; ++i) {
            LookupSwitch.setMatch(q, i, values[i]);
            target_bb = quad_bbs[bc_cfg.getBasicBlockByBytecodeIndex(targets[i]).id];
            LookupSwitch.setTarget(q, i, target_bb);
        }
        appendQuad(q);
    }
    public void visitIRETURN() {
        super.visitIRETURN();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_I();
        Quad q = Return.create(Return.RETURN_I.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitLRETURN() {
        super.visitLRETURN();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_L();
        Quad q = Return.create(Return.RETURN_L.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitFRETURN() {
        super.visitFRETURN();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_F();
        Quad q = Return.create(Return.RETURN_F.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitDRETURN() {
        super.visitDRETURN();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_D();
        Quad q = Return.create(Return.RETURN_D.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitARETURN() {
        super.visitARETURN();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_A();
        Quad q = Return.create(Return.RETURN_A.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitVRETURN() {
        super.visitVRETURN();
        this.uncond_branch = true;
        Quad q = Return.create(Return.RETURN_V.INSTANCE);
        appendQuad(q);
	current_state.clearStack();
    }
    private void GETSTATIChelper(jq_StaticField f, Getstatic oper1, Getstatic oper2) {
        boolean dynlink = f.needsDynamicLink(method);
        Getstatic operator = dynlink?oper1:oper2;
        jq_Type t = f.getType();
        RegisterOperand op0 = getStackRegister(t);
        Quad q = Getstatic.create(operator, op0, new FieldOperand(f));
        appendQuad(q);
        current_state.push(op0.copy(), t);
    }
    public void visitIGETSTATIC(jq_StaticField f) {
        super.visitIGETSTATIC(f);
        GETSTATIChelper(f, Getstatic.GETSTATIC_I_DYNLINK.INSTANCE, Getstatic.GETSTATIC_I.INSTANCE);
    }
    public void visitLGETSTATIC(jq_StaticField f) {
        super.visitLGETSTATIC(f);
        GETSTATIChelper(f, Getstatic.GETSTATIC_L_DYNLINK.INSTANCE, Getstatic.GETSTATIC_L.INSTANCE);
    }
    public void visitFGETSTATIC(jq_StaticField f) {
        super.visitFGETSTATIC(f);
        GETSTATIChelper(f, Getstatic.GETSTATIC_F_DYNLINK.INSTANCE, Getstatic.GETSTATIC_F.INSTANCE);
    }
    public void visitDGETSTATIC(jq_StaticField f) {
        super.visitDGETSTATIC(f);
        GETSTATIChelper(f, Getstatic.GETSTATIC_D_DYNLINK.INSTANCE, Getstatic.GETSTATIC_D.INSTANCE);
    }
    public void visitAGETSTATIC(jq_StaticField f) {
        super.visitAGETSTATIC(f);
        GETSTATIChelper(f, Getstatic.GETSTATIC_A_DYNLINK.INSTANCE, Getstatic.GETSTATIC_A.INSTANCE);
    }
    private void PUTSTATIChelper(jq_StaticField f, Putstatic oper1, Putstatic oper2) {
        boolean dynlink = f.needsDynamicLink(method);
        Putstatic operator = dynlink?oper1:oper2;
        jq_Type t = f.getType();
        Operand op0 = current_state.pop(t);
        Quad q = Putstatic.create(operator, op0, new FieldOperand(f));
        appendQuad(q);
    }
    public void visitIPUTSTATIC(jq_StaticField f) {
        super.visitIPUTSTATIC(f);
        PUTSTATIChelper(f, Putstatic.PUTSTATIC_I_DYNLINK.INSTANCE, Putstatic.PUTSTATIC_I.INSTANCE);
    }
    public void visitLPUTSTATIC(jq_StaticField f) {
        super.visitLPUTSTATIC(f);
        PUTSTATIChelper(f, Putstatic.PUTSTATIC_L_DYNLINK.INSTANCE, Putstatic.PUTSTATIC_L.INSTANCE);
    }
    public void visitFPUTSTATIC(jq_StaticField f) {
        super.visitFPUTSTATIC(f);
        PUTSTATIChelper(f, Putstatic.PUTSTATIC_F_DYNLINK.INSTANCE, Putstatic.PUTSTATIC_F.INSTANCE);
    }
    public void visitDPUTSTATIC(jq_StaticField f) {
        super.visitDPUTSTATIC(f);
        PUTSTATIChelper(f, Putstatic.PUTSTATIC_D_DYNLINK.INSTANCE, Putstatic.PUTSTATIC_D.INSTANCE);
    }
    public void visitAPUTSTATIC(jq_StaticField f) {
        super.visitAPUTSTATIC(f);
        PUTSTATIChelper(f, Putstatic.PUTSTATIC_A_DYNLINK.INSTANCE, Putstatic.PUTSTATIC_A.INSTANCE);
    }
    private void GETFIELDhelper(jq_InstanceField f, Getfield oper1, Getfield oper2) {
        boolean dynlink = f.needsDynamicLink(method);
        Operand op1 = current_state.pop_A();
        clearCurrentGuard();
        if (performNullCheck(op1)) {
	    if (TRACE) System.out.println("Null check triggered on "+op1);
	    return;
	}
        jq_Type t = f.getType();
        RegisterOperand op0 = getStackRegister(t);
        Getfield operator = dynlink?oper1:oper2;
        Quad q = Getfield.create(operator, op0, op1, new FieldOperand(f), getCurrentGuard());
        appendQuad(q);
        current_state.push(op0.copy(), t);
    }
    public void visitIGETFIELD(jq_InstanceField f) {
        super.visitIGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_I_DYNLINK.INSTANCE, Getfield.GETFIELD_I.INSTANCE);
    }
    public void visitLGETFIELD(jq_InstanceField f) {
        super.visitLGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_L_DYNLINK.INSTANCE, Getfield.GETFIELD_L.INSTANCE);
    }
    public void visitFGETFIELD(jq_InstanceField f) {
        super.visitFGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_F_DYNLINK.INSTANCE, Getfield.GETFIELD_F.INSTANCE);
    }
    public void visitDGETFIELD(jq_InstanceField f) {
        super.visitDGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_D_DYNLINK.INSTANCE, Getfield.GETFIELD_D.INSTANCE);
    }
    public void visitAGETFIELD(jq_InstanceField f) {
        super.visitAGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_A_DYNLINK.INSTANCE, Getfield.GETFIELD_A.INSTANCE);
    }
    public void visitBGETFIELD(jq_InstanceField f) {
        super.visitBGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_B_DYNLINK.INSTANCE, Getfield.GETFIELD_B.INSTANCE);
    }
    public void visitCGETFIELD(jq_InstanceField f) {
        super.visitCGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_C_DYNLINK.INSTANCE, Getfield.GETFIELD_C.INSTANCE);
    }
    public void visitSGETFIELD(jq_InstanceField f) {
        super.visitSGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_S_DYNLINK.INSTANCE, Getfield.GETFIELD_S.INSTANCE);
    }
    public void visitZGETFIELD(jq_InstanceField f) {
        super.visitZGETFIELD(f);
        GETFIELDhelper(f, Getfield.GETFIELD_Z_DYNLINK.INSTANCE, Getfield.GETFIELD_Z.INSTANCE);
    }
    private void PUTFIELDhelper(jq_InstanceField f, Putfield oper1, Putfield oper2) {
        boolean dynlink = f.needsDynamicLink(method);
        Operand op0 = current_state.pop(f.getType());
        Operand op1 = current_state.pop_A();
        clearCurrentGuard();
        if (performNullCheck(op1)) {	
	    if (TRACE) System.out.println("Null check triggered on "+op1);
	    return;
	}
        Putfield operator = dynlink?oper1:oper2;
        Quad q = Putfield.create(operator, op0, op1, new FieldOperand(f), getCurrentGuard());
        appendQuad(q);
    }
    public void visitIPUTFIELD(jq_InstanceField f) {
        super.visitIPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_I_DYNLINK.INSTANCE, Putfield.PUTFIELD_I.INSTANCE);
    }
    public void visitLPUTFIELD(jq_InstanceField f) {
        super.visitLPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_L_DYNLINK.INSTANCE, Putfield.PUTFIELD_L.INSTANCE);
    }
    public void visitFPUTFIELD(jq_InstanceField f) {
        super.visitFPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_F_DYNLINK.INSTANCE, Putfield.PUTFIELD_F.INSTANCE);
    }
    public void visitDPUTFIELD(jq_InstanceField f) {
        super.visitDPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_D_DYNLINK.INSTANCE, Putfield.PUTFIELD_D.INSTANCE);
    }
    public void visitAPUTFIELD(jq_InstanceField f) {
        super.visitAPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_A_DYNLINK.INSTANCE, Putfield.PUTFIELD_A.INSTANCE);
    }
    public void visitBPUTFIELD(jq_InstanceField f) {
        super.visitBPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_B_DYNLINK.INSTANCE, Putfield.PUTFIELD_B.INSTANCE);
    }
    public void visitCPUTFIELD(jq_InstanceField f) {
        super.visitCPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_C_DYNLINK.INSTANCE, Putfield.PUTFIELD_C.INSTANCE);
    }
    public void visitSPUTFIELD(jq_InstanceField f) {
        super.visitSPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_S_DYNLINK.INSTANCE, Putfield.PUTFIELD_S.INSTANCE);
    }
    public void visitZPUTFIELD(jq_InstanceField f) {
        super.visitZPUTFIELD(f);
        PUTFIELDhelper(f, Putfield.PUTFIELD_Z_DYNLINK.INSTANCE, Putfield.PUTFIELD_Z.INSTANCE);
    }
    private void UNSAFEhelper(jq_Method m) {
        Quad q;
        if (m == Unsafe._addressOf) {
            Operand op = current_state.pop_A();
            RegisterOperand res = getStackRegister(jq_Primitive.INT);
            q = Unary.create(Unary.OBJECT_2INT.INSTANCE, res, op);
            current_state.push_I(res);
        } else if (m == Unsafe._asObject) {
            Operand op = current_state.pop_I();
            RegisterOperand res = getStackRegister(PrimordialClassLoader.loader.getJavaLangObject());
            q = Unary.create(Unary.INT_2OBJECT.INSTANCE, res, op);
            current_state.push_A(res);
        } else if (m == Unsafe._floatToIntBits) {
            Operand op = current_state.pop_F();
            RegisterOperand res = getStackRegister(jq_Primitive.INT);
            q = Unary.create(Unary.FLOAT_2INTBITS.INSTANCE, res, op);
            current_state.push_I(res);
        } else if (m == Unsafe._intBitsToFloat) {
            Operand op = current_state.pop_I();
            RegisterOperand res = getStackRegister(jq_Primitive.FLOAT);
            q = Unary.create(Unary.INTBITS_2FLOAT.INSTANCE, res, op);
            current_state.push_F(res);
        } else if (m == Unsafe._doubleToLongBits) {
            Operand op = current_state.pop_D();
            RegisterOperand res = getStackRegister(jq_Primitive.LONG);
            q = Unary.create(Unary.DOUBLE_2LONGBITS.INSTANCE, res, op);
            current_state.push_L(res);
        } else if (m == Unsafe._longBitsToDouble) {
            Operand op = current_state.pop_L();
            RegisterOperand res = getStackRegister(jq_Primitive.DOUBLE);
            q = Unary.create(Unary.LONGBITS_2DOUBLE.INSTANCE, res, op);
            current_state.push_D(res);
        } else if (m == Unsafe._poke1) {
            Operand val = current_state.pop_I();
            Operand loc = current_state.pop_I();
            q = MemStore.create(MemStore.POKE_1.INSTANCE, loc, val);
        } else if (m == Unsafe._poke2) {
            Operand val = current_state.pop_I();
            Operand loc = current_state.pop_I();
            q = MemStore.create(MemStore.POKE_2.INSTANCE, loc, val);
        } else if (m == Unsafe._poke4) {
            Operand val = current_state.pop_I();
            Operand loc = current_state.pop_I();
            q = MemStore.create(MemStore.POKE_4.INSTANCE, loc, val);
        } else if (m == Unsafe._peek) {
            Operand loc = current_state.pop_I();
            RegisterOperand res = getStackRegister(jq_Primitive.INT);
            q = MemLoad.create(MemLoad.PEEK_4.INSTANCE, res, loc);
            current_state.push_F(res);
        } else if (m == Unsafe._getThreadBlock) {
            RegisterOperand res = getStackRegister(jq_Thread._class);
            q = Special.create(Special.GET_THREAD_BLOCK.INSTANCE, res);
            current_state.push_A(res);
        } else if (m == Unsafe._setThreadBlock) {
            Operand loc = current_state.pop_A();
            q = Special.create(Special.SET_THREAD_BLOCK.INSTANCE, loc);
        } else if (m == Unsafe._alloca) {
            Operand amt = current_state.pop_I();
            RegisterOperand res = getStackRegister(jq_Primitive.INT);
            q = Special.create(Special.ALLOCA.INSTANCE, res, amt);
            current_state.push_I(res);
        } else if (m == Unsafe._longJump) {
            Operand eax = current_state.pop_I();
            Operand sp = current_state.pop_I();
            Operand fp = current_state.pop_I();
            Operand ip = current_state.pop_I();
            q = Special.create(Special.LONG_JUMP.INSTANCE, ip, fp, sp, eax);
            endBasicBlock = true;
        } else {
            // TODO
            INVOKEhelper(Invoke.INVOKESTATIC_V.INSTANCE, m, jq_Primitive.VOID);
            return;
        }
        appendQuad(q);
    }
    private void INVOKEhelper(Invoke oper, jq_Method f, jq_Type returnType) {
        jq_Type[] paramTypes = f.getParamTypes();
        RegisterOperand result;
        if (returnType == jq_Primitive.VOID) result = null;
        else result = getStackRegister(returnType, f.getParamWords()-1);
        Quad q = Invoke.create(oper, result, new MethodOperand(f), paramTypes.length);
        for (int i = paramTypes.length; --i >= 0; ) {
            jq_Type ptype = paramTypes[i];
            Operand op = current_state.pop(ptype);
            RegisterOperand rop;
            if (op instanceof RegisterOperand) rop = (RegisterOperand)op;
            else {
                rop = getStackRegister(ptype);
                Quad q2 = Move.create(Move.getMoveOp(ptype), rop, op);
                appendQuad(q2);
            }
            Invoke.setParam(q, i, rop);
        }
        appendQuad(q);
        if (result != null) current_state.push(result, returnType);
    }
    public void visitIINVOKE(byte op, jq_Method f) {
        super.visitIINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_I_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_I.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_I_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_I.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_I_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_I.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_I.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, jq_Primitive.INT);
    }
    public void visitLINVOKE(byte op, jq_Method f) {
        super.visitLINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_L_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_L.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_L_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_L.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_L_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_L.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_L.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, jq_Primitive.LONG);
    }
    public void visitFINVOKE(byte op, jq_Method f) {
        super.visitFINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_F_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_F.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_F_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_F.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_F_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_F.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_F.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, jq_Primitive.FLOAT);
    }
    public void visitDINVOKE(byte op, jq_Method f) {
        super.visitDINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_D_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_D.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_D_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_D.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_D_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_D.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_D.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, jq_Primitive.DOUBLE);
    }
    public void visitAINVOKE(byte op, jq_Method f) {
        super.visitAINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_A_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_A.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_A_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_A.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_A_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_A.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_A.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, f.getReturnType());
    }
    public void visitVINVOKE(byte op, jq_Method f) {
        super.visitVINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            UNSAFEhelper(f);
            return;
        }
        Invoke oper;
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    oper = Invoke.INVOKEVIRTUAL_V_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKEVIRTUAL_V.INSTANCE;
                break;
            case INVOKE_STATIC:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESTATIC_V_DYNLINK.INSTANCE;
                else
                    oper = Invoke.INVOKESTATIC_V.INSTANCE;
                break;
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    oper = Invoke.INVOKESPECIAL_V_DYNLINK.INSTANCE;
                else {
                    f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                    oper = Invoke.INVOKESTATIC_V.INSTANCE;
                }
                break;
            case INVOKE_INTERFACE:
                oper = Invoke.INVOKEINTERFACE_V.INSTANCE;
                break;
            default:
                throw new InternalError();
        }
        INVOKEhelper(oper, f, f.getReturnType());
    }
    public void visitNEW(jq_Type f) {
        super.visitNEW(f);
        RegisterOperand res = getStackRegister(f);
        Quad q = New.create(New.NEW.INSTANCE, res, new TypeOperand(f));
        appendQuad(q);
        current_state.push_A(res);
    }
    public void visitNEWARRAY(jq_Array f) {
        super.visitNEWARRAY(f);
        Operand size = current_state.pop_I();
        RegisterOperand res = getStackRegister(f);
        Quad q = NewArray.create(NewArray.NEWARRAY.INSTANCE, res, size, new TypeOperand(f));
        appendQuad(q);
        current_state.push_A(res);
    }
    public void visitCHECKCAST(jq_Type f) {
        super.visitCHECKCAST(f);
        Operand op = current_state.pop_A();
        RegisterOperand res = getStackRegister(f);
        Quad q = CheckCast.create(CheckCast.CHECKCAST.INSTANCE, res, op, new TypeOperand(f));
        appendQuad(q);
        current_state.push_A(res);
    }
    public void visitINSTANCEOF(jq_Type f) {
        super.visitINSTANCEOF(f);
        Operand op = current_state.pop_A();
        RegisterOperand res = getStackRegister(jq_Primitive.BOOLEAN);
        Quad q = InstanceOf.create(InstanceOf.INSTANCEOF.INSTANCE, res, op, new TypeOperand(f));
        appendQuad(q);
        current_state.push_I(res);
    }
    public void visitARRAYLENGTH() {
        super.visitARRAYLENGTH();
        Operand op = current_state.pop_A();
        RegisterOperand res = getStackRegister(jq_Primitive.INT);
        Quad q = ALength.create(ALength.ARRAYLENGTH.INSTANCE, res, op);
        appendQuad(q);
        current_state.push_I(res);
    }
    public void visitATHROW() {
        super.visitATHROW();
        this.uncond_branch = true;
        Operand op0 = current_state.pop_A();
        Quad q = Return.create(Return.THROW_A.INSTANCE, op0);
        appendQuad(q);
	current_state.clearStack();
    }
    public void visitMONITOR(byte op) {
        super.visitMONITOR(op);
        Operand op0 = current_state.pop_A();
        Monitor oper = op==MONITOR_ENTER ? (Monitor)Monitor.MONITORENTER.INSTANCE : (Monitor)Monitor.MONITOREXIT.INSTANCE;
        Quad q = Monitor.create(oper, op0);
        appendQuad(q);
    }
    public void visitMULTINEWARRAY(jq_Type f, char dim) {
        super.visitMULTINEWARRAY(f, dim);
        // TODO
        jq.UNREACHABLE();
    }
    
    boolean performNullCheck(Operand op) {
        if (op instanceof AConstOperand) {
            Object val = ((AConstOperand)op).getValue();
            if (val != null) {
                setCurrentGuard(new UnnecessaryGuardOperand());
                return false;
            } else {
                Quad q = NullCheck.create(NullCheck.NULL_CHECK.INSTANCE, null, op);
		if (false) {
		    endBasicBlock = true;
		    mergeStateWithNullPtrExHandler(true);
		    return true;
		} else {
		    mergeStateWithNullPtrExHandler(false);
		    return false;
		}
            }
        }
        RegisterOperand rop = (RegisterOperand)op;
	if (hasGuard(rop)) {
            Operand guard = getGuard(rop);
            setCurrentGuard(guard);
            return false;
	}
        RegisterOperand guard = makeGuardReg();
        Quad q = NullCheck.create(NullCheck.NULL_CHECK.INSTANCE, guard, rop.copy());
        appendQuad(q);
        mergeStateWithNullPtrExHandler(false);
	setCurrentGuard(guard);
	setGuard(rop, guard);
        
        jq_Type type = rop.getType();
        int number = getLocalNumber(rop.getRegister(), type);
        if (rf.isLocal(rop, number, type)) {
            Operand op2 = current_state.getLocal_A(number);
            if (op2 instanceof RegisterOperand) {
                setGuard((RegisterOperand)op2, guard);
            }
            current_state.setLocal(number, op2);
            replaceLocalsOnStack(number, type);
        }
	return false;
    }
    
    boolean performBoundsCheck(Operand ref, Operand index) {
        Quad q = BoundsCheck.create(BoundsCheck.BOUNDS_CHECK.INSTANCE, ref.copy(), index.copy(), getCurrentGuard());
        mergeStateWithArrayBoundsExHandler(false);
        return false;
    }
    
    boolean performCheckStore(RegisterOperand ref, Operand elem) {
        jq_Type type = getTypeOf(elem);
        if (type == jq_Reference.jq_NullType.NULL_TYPE) return false;
        jq_Type arrayElemType = getArrayTypeOf(ref).getElementType();
        if (ref.isExactType()) {
            if (isAssignable(type, arrayElemType) == YES)
                return false;
        }
        jq_Type arrayElemType2 = arrayElemType;
        if (arrayElemType.isArrayType()) {
            arrayElemType2 = ((jq_Array)arrayElemType).getInnermostElementType();
        }
        if (arrayElemType2.isLoaded() && arrayElemType2.isFinal()) {
            if (arrayElemType == type)
                return false;
        }
        Quad q = StoreCheck.create(StoreCheck.ASTORE_CHECK.INSTANCE, ref.copy(), elem.copy(), getCurrentGuard());
        appendQuad(q);
        mergeStateWithObjArrayStoreExHandler(false);
        return false;
    }

    boolean performZeroCheck(Operand op) {
        if (op instanceof IConstOperand) {
            int val = ((IConstOperand)op).getValue();
            if (val != 0) {
                setCurrentGuard(new UnnecessaryGuardOperand());
                return false;
            } else {
                Quad q = ZeroCheck.create(ZeroCheck.ZERO_CHECK.INSTANCE, null, op);
		if (false) {
		    endBasicBlock = true;
		    mergeStateWithArithExHandler(true);
		    return true;
		} else {
		    mergeStateWithArithExHandler(false);
		    return false;
		}
            }
        }
        RegisterOperand rop = (RegisterOperand)op;
	if (hasGuard(rop)) {
            Operand guard = getGuard(rop);
            setCurrentGuard(guard);
            return false;
	}
        RegisterOperand guard = makeGuardReg();
        Quad q = ZeroCheck.create(ZeroCheck.ZERO_CHECK.INSTANCE, guard, rop.copy());
        appendQuad(q);
        mergeStateWithArithExHandler(false);
	setCurrentGuard(guard);
	setGuard(rop, guard);
        
        jq_Type type = rop.getType();
        int number = getLocalNumber(rop.getRegister(), type);
        if (rf.isLocal(rop, number, type)) {
            Operand op2 = current_state.getLocal_I(number);
            if (op2 instanceof RegisterOperand) {
                setGuard((RegisterOperand)op2, guard);
            }
            current_state.setLocal(number, op2);
            replaceLocalsOnStack(number, type);
        }
	return false;
    }
    
    static jq_Type getTypeOf(Operand op) {
        if (op instanceof IConstOperand) return jq_Primitive.INT;
        if (op instanceof FConstOperand) return jq_Primitive.FLOAT;
        if (op instanceof LConstOperand) return jq_Primitive.LONG;
        if (op instanceof DConstOperand) return jq_Primitive.DOUBLE;
        if (op instanceof AConstOperand) {
            Object val = ((AConstOperand)op).getValue();
            if (val == null) return jq_Reference.jq_NullType.NULL_TYPE;
            return Reflection.getTypeOf(val);
        }
        return ((RegisterOperand)op).getType();
    }
    static jq_Array getArrayTypeOf(Operand op) {
        return (jq_Array)((RegisterOperand)op).getType();
    }
    
    static final byte YES = 2;
    static final byte MAYBE = 1;
    static final byte NO = 0;
    // returns YES if "T = S;" would be legal. (T is same or supertype of S)
    // S and T should already be prepared.
    static byte isAssignable(jq_Type S, jq_Type T) {
        if (S == jq_Reference.jq_NullType.NULL_TYPE) {
            if (T.isReferenceType()) return YES;
            else return NO;
        }
        if (T == jq_Reference.jq_NullType.NULL_TYPE) return NO;
        if (T == S) return YES;
        if (T.isIntLike() && S.isIntLike()) return YES;
        if (T == PrimordialClassLoader.loader.getJavaLangObject() && S.isReferenceType()) return YES;
        if (!T.isPrepared() || !S.isPrepared()) return MAYBE;
        if (T.isArrayType()) {
            jq_Type elemType = ((jq_Array)T).getInnermostElementType();
            if (!elemType.isPrepared()) return MAYBE;
        }
        if (S.isArrayType()) {
            jq_Type elemType = ((jq_Array)S).getInnermostElementType();
            if (!elemType.isPrepared()) return MAYBE;
        }
        if (TypeCheck.isAssignable(S, T)) return YES;
        else return NO;
    }
    
    void mergeStateWithNullPtrExHandler(boolean cfgEdgeToExit) {
        // TODO.
    }
    void mergeStateWithArithExHandler(boolean cfgEdgeToExit) {
        // TODO.
    }
    void mergeStateWithArrayBoundsExHandler(boolean cfgEdgeToExit) {
        // TODO.
    }
    void mergeStateWithObjArrayStoreExHandler(boolean cfgEdgeToExit) {
        // TODO.
    }
    
    RegisterOperand makeGuardReg() {
        return rf.makeGuardReg();
    }
    
    int getLocalNumber(Register r, jq_Type t) {
        return r.getNumber();
    }
    
    /** Class used to store the abstract state of the bytecode-to-quad converter. */
    public static class AbstractState {

        public static boolean TRACE = false;
        
        private int stackptr;
        private Operand[] stack;
        private Operand[] locals;
        
        static class DummyOperand implements Operand {
            private DummyOperand() {}
            static final DummyOperand DUMMY = new DummyOperand();
            public Quad getQuad() { jq.UNREACHABLE(); return null; }
            public void attachToQuad(Quad q) { jq.UNREACHABLE(); }
            public Operand copy() { return DUMMY; }
            public boolean isSimilar(Operand that) { return that == DUMMY; }
        }
        
        static AbstractState allocateEmptyState(jq_Method m) {
            AbstractState s = new AbstractState(m.getMaxStack(), m.getMaxLocals());
            return s;
        }
        
        static AbstractState allocateInitialState(RegisterFactory rf, jq_Method m) {
            AbstractState s = new AbstractState(m.getMaxStack(), m.getMaxLocals());
            jq_Type[] paramTypes = m.getParamTypes();
            for (int i=0, j=-1; i<paramTypes.length; ++i) {
                jq_Type paramType = paramTypes[i];
                s.locals[++j] = new RegisterOperand(rf.getLocal(i, paramType), paramType);
                if (paramType.getReferenceSize() == 8) {
                    s.locals[++j] = DummyOperand.DUMMY;
                }
            }
            return s;
        }
        
        private AbstractState(int nstack, int nlocals) {
            this.stack = new Operand[nstack]; this.locals = new Operand[nlocals];
        }
        
        AbstractState copy() {
            AbstractState that = new AbstractState(this.stack.length, this.locals.length);
            System.arraycopy(this.stack, 0, that.stack, 0, this.stackptr);
            System.arraycopy(this.locals, 0, that.locals, 0, this.locals.length);
            that.stackptr = this.stackptr;
            return that;
        }
        
        AbstractState copyFull() {
            AbstractState that = new AbstractState(this.stack.length, this.locals.length);
            for (int i=0; i<stackptr; ++i) {
                that.stack[i] = this.stack[i].copy();
            }
            for (int i=0; i<this.locals.length; ++i) {
                that.locals[i] = this.locals[i].copy();
            }
            that.stackptr = this.stackptr;
            return that;
        }
        
        void overwriteWith(AbstractState that) {
            jq.assert(this.stack.length == that.stack.length);
            jq.assert(this.locals.length == that.locals.length);
            System.arraycopy(that.stack, 0, this.stack, 0, that.stackptr);
            System.arraycopy(that.locals, 0, this.locals, 0, that.locals.length);
            this.stackptr = that.stackptr;
        }
        
        void merge(AbstractState that, RegisterFactory rf) {
            if (this.stackptr != that.stackptr) throw new VerifyError(this.stackptr+" != "+that.stackptr);
            jq.assert(this.locals.length == that.locals.length);
            for (int i=0; i<this.stackptr; ++i) {
                this.stack[i] = meet(this.stack[i], that.stack[i], true, i, rf);
            }
            for (int i=0; i<this.locals.length; ++i) {
                this.locals[i] = meet(this.locals[i], that.locals[i], false, i, rf);
            }
        }
        
        static Operand meet(Operand op1, Operand op2, boolean stack, int index, RegisterFactory rf) {
            if (TRACE) System.out.println("Meeting "+op1+" with "+op2+", "+(stack?"S":"L")+index);
            if (op1 == op2) {
                // same operand, or both null.
                return op1;
            }
            if ((op1 == null) || (op2 == null)) {
                // no information about one of the operands.
                return null;
            }
            if (Operand.Util.isConstant(op1)) {
                if (op1.isSimilar(op2)) {
                    // same constant value.
                    return op1;
                }
                jq_Type type = TypeCheck.findCommonSuperclass(getTypeOf(op1), getTypeOf(op2));
                if (type != null) {
                    // different constants of the same type
                    RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, type):rf.getLocal(index, type), type);
                    return res;
                } else {
                    // constants of incompatible types.
                    return null;
                }
            }
            if (op1 instanceof RegisterOperand) {
                RegisterOperand rop1 = (RegisterOperand)op1;
                jq_Type t1 = rop1.getType();
                if (op2 instanceof RegisterOperand) {
                    // both are registers.
                    RegisterOperand rop2 = (RegisterOperand)op2;
                    jq_Type t2 = rop2.getType();
                    if (t1 == t2) {
                        // registers have same type.
                        if (rop1.hasMoreConservativeFlags(rop2)) {
                            // registers have compatible flags.
                            if ((rop1.scratchObject == null) ||
                                ((Operand)rop1.scratchObject).isSimilar((Operand)rop2.scratchObject)) {
                                // null guards match.
                                return rop1;
                            }
                            // null guards don't match.
                            RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                            res.setFlags(rop1.getFlags());
                            return res;
                        }
                        // incompatible flags.
                        RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                        if ((rop1.scratchObject == null) ||
                            ((Operand)rop1.scratchObject).isSimilar((Operand)rop2.scratchObject)) {
                            // null guards match.
                            res.scratchObject = rop1.scratchObject;
                        }
                        res.setFlags(rop1.getFlags());
                        res.meetFlags(rop2.getFlags());
                        return res;
                    }
                    if (isAssignable(t2, t1) == YES) {
                        // t2 is a subtype of t1.
                        if (!rop1.isExactType() && rop1.hasMoreConservativeFlags(rop2)) {
                            // flags and exact type matches.
                            if ((rop1.scratchObject == null) ||
                                ((Operand)rop1.scratchObject).isSimilar((Operand)rop2.scratchObject)) {
                                // null guards match.
                                return rop1;
                            }
                            // null guards don't match.
                            RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                            res.setFlags(rop1.getFlags());
                            return res;
                        }
                        // doesn't match.
                        RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                        if ((rop1.scratchObject == null) ||
                            ((Operand)rop1.scratchObject).isSimilar((Operand)rop2.scratchObject)) {
                            // null guards match.
                            res.scratchObject = rop1.scratchObject;
                        }
                        res.setFlags(rop1.getFlags());
                        res.meetFlags(rop2.getFlags());
                        res.clearExactType();
                        return res;
                    }
                    if ((t2 = TypeCheck.findCommonSuperclass(t1, t2)) != null) {
                        // common superclass
                        RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t2):rf.getLocal(index, t2), t2);
                        if (rop1.scratchObject != null) {
                            if (((Operand)rop1.scratchObject).isSimilar((Operand)rop2.scratchObject)) {
                                // null guards match.
                                res.scratchObject = rop1.scratchObject;
                            }
                        }
                        res.setFlags(rop1.getFlags());
                        res.meetFlags(rop2.getFlags());
                        res.clearExactType();
                        return res;
                    }
                    // no common superclass
                    return null;
                }
                // op2 is not a register.
                jq_Type t2 = getTypeOf(op2);
                if (t1 == t2) {
                    // same type.
                    if ((rop1.scratchObject == null) || (t2 != jq_Reference.jq_NullType.NULL_TYPE)) {
                        // null guard matches.
                        return rop1;
                    }
                    // null guards doesn't match.
                    RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                    res.setFlags(rop1.getFlags());
                    return res;
                }
                if (isAssignable(t2, t1) == YES) {
                    // compatible type.
                    if (!rop1.isExactType()) {
                        if ((rop1.scratchObject == null) || (t2 != jq_Reference.jq_NullType.NULL_TYPE)) {
                            // null guard matches.
                            return rop1;
                        }
                        // null guard doesn't match.
                        RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                        res.setFlags(rop1.getFlags());
                        return res;
                    }
                    RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t1):rf.getLocal(index, t1), t1);
                    if (t2 != jq_Reference.jq_NullType.NULL_TYPE) {
                        // null guard matches.
                        res.scratchObject = rop1.scratchObject;
                    }
                    res.setFlags(rop1.getFlags());
                    res.clearExactType();
                    return res;
                }
                if ((t2 = TypeCheck.findCommonSuperclass(t1, t2)) != null) {
                    // common superclass
                    RegisterOperand res = new RegisterOperand(stack?rf.getStack(index, t2):rf.getLocal(index, t2), t2);
                    if (t2 != jq_Reference.jq_NullType.NULL_TYPE) {
                        // null guard matches.
                        res.scratchObject = rop1.scratchObject;
                    }
                    res.setFlags(rop1.getFlags());
                    res.clearExactType();
                    return res;
                }
                // no common superclass
                return null;
            }
            // op1 is not a register.
            if (op1.isSimilar(op2)) {
                return op1;
            } else {
                return null;
            }
        }
        
        int getStackSize() { return this.stackptr; }
        
        void push_I(Operand op) { jq.assert(getTypeOf(op).isIntLike()); push(op); }
        void push_F(Operand op) { jq.assert(getTypeOf(op) == jq_Primitive.FLOAT); push(op); }
        void push_L(Operand op) { jq.assert(getTypeOf(op) == jq_Primitive.LONG); push(op); pushDummy(); }
        void push_D(Operand op) { jq.assert(getTypeOf(op) == jq_Primitive.DOUBLE); push(op); pushDummy(); }
        void push_A(Operand op) { jq.assert(getTypeOf(op).isReferenceType()); push(op); }
        void push(Operand op, jq_Type t) {
            jq.assert(isAssignable(getTypeOf(op), t) == YES);
            push(op); if (t.getReferenceSize() == 8) pushDummy();
        }
        void pushDummy() { push(DummyOperand.DUMMY); }
        void push(Operand op) {
            if (TRACE) System.out.println("Pushing "+op+" on stack "+(this.stackptr));
            this.stack[this.stackptr++] = op;
        }

        Operand pop_I() { Operand op = pop(); jq.assert(getTypeOf(op).isIntLike()); return op; }
        Operand pop_F() { Operand op = pop(); jq.assert(getTypeOf(op) == jq_Primitive.FLOAT); return op; }
        Operand pop_L() { popDummy(); Operand op = pop(); jq.assert(getTypeOf(op) == jq_Primitive.LONG); return op; }
        Operand pop_D() { popDummy(); Operand op = pop(); jq.assert(getTypeOf(op) == jq_Primitive.DOUBLE); return op; }
        Operand pop_A() { Operand op = pop(); jq.assert(getTypeOf(op).isReferenceType()); return op; }
        void popDummy() { Operand op = pop(); jq.assert(op == DummyOperand.DUMMY); }
        Operand pop(jq_Type t) {
            if (t.getReferenceSize() == 8) popDummy();
            Operand op = pop(); jq.assert(isAssignable(getTypeOf(op), t) != NO);
            return op;
        }
        Operand pop() {
            if (TRACE) System.out.println("Popping "+this.stack[this.stackptr-1]+" from stack "+(this.stackptr-1));
            return this.stack[--this.stackptr];
        }

        Operand peekStack(int i) { return this.stack[this.stackptr-i-1]; }
        void pokeStack(int i, Operand op) { this.stack[this.stackptr-i-1] = op; }
	void clearStack() { this.stackptr = 0; }
        
        Operand getLocal_I(int i) { Operand op = getLocal(i); jq.assert(getTypeOf(op).isIntLike()); return op; }
        Operand getLocal_F(int i) { Operand op = getLocal(i); jq.assert(getTypeOf(op) == jq_Primitive.FLOAT); return op; }
        Operand getLocal_L(int i) {
            Operand op = getLocal(i);
            jq.assert(getTypeOf(op) == jq_Primitive.LONG);
            jq.assert(getLocal(i+1) == DummyOperand.DUMMY);
            return op;
        }
        Operand getLocal_D(int i) {
            Operand op = getLocal(i);
            jq.assert(getTypeOf(op) == jq_Primitive.DOUBLE);
            jq.assert(getLocal(i+1) == DummyOperand.DUMMY);
            return op;
        }
        Operand getLocal_A(int i) { Operand op = getLocal(i); jq.assert(getTypeOf(op).isReferenceType()); return op; }
        Operand getLocal(int i) {
            return this.locals[i].copy();
        }
        void setLocal(int i, Operand op) {
            this.locals[i] = op;
        }
        void setLocalDual(int i, Operand op) {
            this.locals[i] = op; this.locals[i+1] = DummyOperand.DUMMY;
        }
	void dumpState() {
	    System.out.print("Locals:");
	    for (int i=0; i<this.locals.length; ++i) {
		if (this.locals[i] != null)
		    System.out.print(" L"+i+":"+this.locals[i]);
	    }
	    System.out.print("\nStack: ");
	    for (int i=0; i<this.stackptr; ++i) {
		System.out.print(" S"+stackptr+":"+this.locals[i]);
	    }
	    System.out.println();
	}
    }

    public static class jq_ReturnAddressType extends jq_Reference {
        public static final jq_ReturnAddressType INSTANCE = new jq_ReturnAddressType();
        private BasicBlock returnTarget;
        private jq_ReturnAddressType() { super(Utf8.get("L&ReturnAddress;"), Bootstrap.PrimordialClassLoader.loader); }
        private jq_ReturnAddressType(BasicBlock returnTarget) {
            super(Utf8.get("L&ReturnAddress;"), Bootstrap.PrimordialClassLoader.loader);
            this.returnTarget = returnTarget;
        }
        public String getJDKName() { jq.UNREACHABLE(); return null; }
        public String getJDKDesc() { jq.UNREACHABLE(); return null; }
        public Clazz.jq_Class[] getInterfaces() { jq.UNREACHABLE(); return null; }
        public Clazz.jq_Class getInterface(Utf8 desc) { jq.UNREACHABLE(); return null; }
        public boolean implementsInterface(Clazz.jq_Class k) { jq.UNREACHABLE(); return false; }
        public Clazz.jq_InstanceMethod getVirtualMethod(Clazz.jq_NameAndDesc nd) { jq.UNREACHABLE(); return null; }
        public String getName() { return "<retaddr>"; }
        public String shortName() { return "<retaddr>"; }
        public boolean isClassType() { jq.UNREACHABLE(); return false; }
        public boolean isArrayType() { jq.UNREACHABLE(); return false; }
        public boolean isFinal() { jq.UNREACHABLE(); return false; }
        public void load() { jq.UNREACHABLE(); }
        public void verify() { jq.UNREACHABLE(); }
        public void prepare() { jq.UNREACHABLE(); }
        public void sf_initialize() { jq.UNREACHABLE(); }
        public void cls_initialize() { jq.UNREACHABLE(); }
        public String toString() { return "<retaddr> (target="+returnTarget+")"; }
        public boolean equals(Object rat) {
            if (!(rat instanceof jq_ReturnAddressType)) return false;
            return ((jq_ReturnAddressType)rat).returnTarget.equals(this.returnTarget);
        }
        public int hashCode() {
            return returnTarget.hashCode();
        }
    }
}

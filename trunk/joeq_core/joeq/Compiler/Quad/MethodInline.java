package Compil3r.Quad;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Compil3r.Quad.Operand.ParamListOperand;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operand.TargetOperand;
import Compil3r.Quad.Operator.Goto;
import Compil3r.Quad.Operator.Invoke;
import Compil3r.Quad.Operator.Move;
import Compil3r.Quad.Operator.Return;
import Compil3r.Quad.RegisterFactory.Register;
import Main.jq;
import Util.FilterIterator.Filter;
import Util.Templates.ListIterator;

public class MethodInline implements ControlFlowGraphVisitor {

    public static final boolean TRACE = false;
    public static final java.io.PrintStream out = System.out;

    protected Filter f;

    public MethodInline(Filter f) { this.f = f; }
    public MethodInline() { }
    
    public void visitCFG(ControlFlowGraph cfg) {
        QuadIterator qi = new QuadIterator(cfg);
        java.util.LinkedList inline_blocks = new java.util.LinkedList();
        java.util.LinkedList inline_quads = new java.util.LinkedList();
        while (qi.hasNext()) {
            Quad q = qi.nextQuad();
            if (q.getOperator() instanceof Invoke.INVOKESTATIC_V ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_I ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_F ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_L ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_D ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_A ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_V_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_I_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_F_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_L_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_D_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESTATIC_A_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_V_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_I_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_F_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_L_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_D_DYNLINK ||
                q.getOperator() instanceof Invoke.INVOKESPECIAL_A_DYNLINK) {
                if (Invoke.getMethod(q).getMethod().needsDynamicLink(cfg.getMethod())) continue;
                if (Invoke.getMethod(q).getMethod().getBytecode() == null) continue;
                // HACK: for interpreter.
                if (!Interpreter.QuadInterpreter.State.interpret_filter.isElement(Invoke.getMethod(q).getMethod())) continue;
                inline_quads.add(q);
                inline_blocks.add(qi.getCurrentBasicBlock());
            }
        }
        // do the inlining backwards, so that basic blocks don't change.
        java.util.ListIterator li1 = inline_blocks.listIterator();
        java.util.ListIterator li2 = inline_quads.listIterator();
        while (li1.hasNext()) { li1.next(); li2.next(); }
        while (li1.hasPrevious()) {
            BasicBlock bb = (BasicBlock)li1.previous();
            Quad q = (Quad)li2.previous();
            MethodInline.inlineNonVirtualCallSite(cfg, bb, q);
        }
    }
    
    public static void inlineNonVirtualCallSite(ControlFlowGraph caller, BasicBlock bb, Quad q) {
        if (TRACE) out.println("Inlining "+q+" in "+bb);
        jq_Method m = Invoke.getMethod(q).getMethod();
        ControlFlowGraph callee = CodeCache.getCode(m);

        int invokeLocation = bb.getQuadIndex(q);
        jq.Assert(invokeLocation != -1);

        if (TRACE) out.println("Code to inline:");
        if (TRACE) out.println(callee.fullDump());
        if (TRACE) out.println(callee.getRegisterFactory().fullDump());

        // copy the callee's control flow graph, renumbering the basic blocks,
        // registers and quads, and add the exception handlers.
        callee = caller.merge(callee);
        if (TRACE) out.println("After renumbering:");
        if (TRACE) out.println(callee.fullDump());
        if (TRACE) out.println(callee.getRegisterFactory().fullDump());
        callee.appendExceptionHandlers(bb.getExceptionHandlers());

        if (TRACE) out.println("Original basic block containing invoke:");
        if (TRACE) out.println(bb.fullDump());

        // split the basic block containing the invoke, and start splicing
        // in the callee's control flow graph.
        BasicBlock successor_bb = caller.createBasicBlock(callee.exit().getNumberOfPredecessors(), bb.getNumberOfSuccessors(), bb.size() - invokeLocation, bb.getExceptionHandlers());
        int bb_size = bb.size();
        for (int i=invokeLocation+1; i<bb_size; ++i) {
            successor_bb.appendQuad(bb.removeQuad(invokeLocation+1));
        }
        Quad invokeQuad = bb.removeQuad(invokeLocation);
        jq.Assert(invokeQuad == q);
        jq.Assert(bb.size() == invokeLocation);
        if (TRACE) out.println("Result of splitting:");
        if (TRACE) out.println(bb.fullDump());
        if (TRACE) out.println(successor_bb.fullDump());

        // add instructions to set parameters.
        ParamListOperand plo = Invoke.getParamList(q);
        jq_Type[] params = m.getParamTypes();
        for (int i=0, j=0; i<params.length; ++i, ++j) {
            Move op = Move.getMoveOp(params[i]);
            Register dest_r = callee.getRegisterFactory().getLocal(j, params[i]);
            RegisterOperand dest = new RegisterOperand(dest_r, params[i]);
            Register src_r = plo.get(i).getRegister();
            RegisterOperand src = new RegisterOperand(src_r, params[i]);
            Quad q2 = Move.create(caller.getNewQuadID(), op, dest, src);
            bb.appendQuad(q2);
            if (params[i].getReferenceSize() == 8) ++j;
        }
        if (TRACE) out.println("Result after adding parameter moves:");
        if (TRACE) out.println(bb.fullDump());

        // replace return instructions with moves and gotos, and
        // finish splicing in the control flow graph.
        for (ListIterator.BasicBlock it = bb.getSuccessors().basicBlockIterator();
             it.hasNext(); ) {
            BasicBlock next_bb = it.nextBasicBlock();
            next_bb.removePredecessor(bb);
            next_bb.addPredecessor(successor_bb);
            successor_bb.addSuccessor(next_bb);
        }
        bb.removeAllSuccessors();
        bb.addSuccessor(callee.entry().getFallthroughSuccessor());
        callee.entry().getFallthroughSuccessor().removeAllPredecessors();
        callee.entry().getFallthroughSuccessor().addPredecessor(bb);
        RegisterOperand dest = Invoke.getDest(q);
        Register dest_r = null;
        Move op = null;
        if (dest != null) {
            dest_r = dest.getRegister();
            op = Move.getMoveOp(callee.getMethod().getReturnType());
        }
outer:
        for (ListIterator.BasicBlock it = callee.exit().getPredecessors().basicBlockIterator();
             it.hasNext(); ) {
            BasicBlock pred_bb = it.nextBasicBlock();
            while (pred_bb.size() == 0) { 
                if (TRACE) System.out.println("Predecessor of exit has no quads? "+pred_bb);
                if (pred_bb.getNumberOfPredecessors() == 0) continue outer;
                pred_bb = pred_bb.getFallthroughPredecessor();
            }
            Quad lq = pred_bb.getLastQuad();
            if (lq.getOperator() instanceof Return.THROW_A) {
                pred_bb.removeAllSuccessors(); pred_bb.addSuccessor(caller.exit());
                caller.exit().addPredecessor(pred_bb);
                continue;
            }
            jq.Assert(lq.getOperator() instanceof Return);
            pred_bb.removeQuad(lq);
            if (dest_r != null) {
                RegisterOperand ldest = new RegisterOperand(dest_r, callee.getMethod().getReturnType());
                Operand src = Return.getSrc(lq).copy();
                Quad q3 = Move.create(caller.getNewQuadID(), op, ldest, src);
                pred_bb.appendQuad(q3);
            }
            Quad q2 = Goto.create(caller.getNewQuadID(),
                                  Goto.GOTO.INSTANCE,
                                  new TargetOperand(successor_bb));
            pred_bb.appendQuad(q2);
            pred_bb.removeAllSuccessors(); pred_bb.addSuccessor(successor_bb);
            successor_bb.addPredecessor(pred_bb);
        }
        
        if (TRACE) out.println("Final result:");
        if (TRACE) out.println(caller.fullDump());
        if (TRACE) out.println(caller.getRegisterFactory().fullDump());

        if (TRACE) out.println("Original code:");
        if (TRACE) out.println(CodeCache.getCode(m).fullDump());
        if (TRACE) out.println(CodeCache.getCode(m).getRegisterFactory().fullDump());
    }

}

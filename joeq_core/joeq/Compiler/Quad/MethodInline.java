// MethodInline.java, created Wed Mar 13  1:39:18 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Quad;

import java.awt.event.InvocationEvent;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;

import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_Primitive;
import joeq.Class.jq_Type;
import joeq.Compiler.Analysis.IPA.PA;
import joeq.Compiler.Analysis.IPA.ProgramLocation;
import joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import joeq.Compiler.BytecodeAnalysis.BytecodeVisitor;
import joeq.Compiler.Quad.Operand.ConditionOperand;
import joeq.Compiler.Quad.Operand.IConstOperand;
import joeq.Compiler.Quad.Operand.MethodOperand;
import joeq.Compiler.Quad.Operand.ParamListOperand;
import joeq.Compiler.Quad.Operand.RegisterOperand;
import joeq.Compiler.Quad.Operand.TargetOperand;
import joeq.Compiler.Quad.Operand.TypeOperand;
import joeq.Compiler.Quad.Operator.Goto;
import joeq.Compiler.Quad.Operator.InstanceOf;
import joeq.Compiler.Quad.Operator.IntIfCmp;
import joeq.Compiler.Quad.Operator.Invoke;
import joeq.Compiler.Quad.Operator.Move;
import joeq.Compiler.Quad.Operator.Return;
import joeq.Compiler.Quad.RegisterFactory.Register;
import joeq.Util.Templates.ListIterator;
import jwutil.util.Assert;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class MethodInline implements ControlFlowGraphVisitor {

    public static final boolean TRACE = false;
    public static final boolean TRACE_ORACLE = false;
    public static final boolean TRACE_DECISIONS = false;
    public static final java.io.PrintStream out = System.out;

    Oracle oracle;
    CallGraph cg;
    private static MethodOperand fakeMethodOperand;

    public MethodInline(Oracle o) {
        this.oracle = o;
    }
    public MethodInline(CallGraph cg) {
//        this(new InlineSmallSingleTargetCalls(cg));\
        this(new InlineSelectedCalls(cg));
        this.cg = cg;
    }
    public MethodInline() {
//        this(new InlineSmallSingleTargetCalls(CHACallGraph.INSTANCE));
        this(new InlineSelectedCalls(CHACallGraph.INSTANCE));
        this.cg = CHACallGraph.INSTANCE;
    }

    public static interface Oracle {
        InliningDecision shouldInline(ControlFlowGraph caller, BasicBlock bb, Quad callSite);
    }
    
    public static abstract class InliningDecision {
        public abstract void inlineCall(ControlFlowGraph caller, BasicBlock bb, Quad q);
    }
    
    public static final class DontInline extends InliningDecision {
        private DontInline() {}
        public void inlineCall(ControlFlowGraph caller, BasicBlock bb, Quad q) {
        }
        public static final DontInline INSTANCE = new DontInline();
    }
    
    public static class NoCheckInliningDecision extends InliningDecision {
        ControlFlowGraph callee;
        public NoCheckInliningDecision(jq_Method target) {
            this(CodeCache.getCode(target));
        }
        public NoCheckInliningDecision(ControlFlowGraph target) {
            this.callee = target;
        }
        public void inlineCall(ControlFlowGraph caller, BasicBlock bb, Quad q) {
            inlineNonVirtualCallSite(caller, bb, q, callee);
        }
        public String toString() { return callee.getMethod().toString(); }
    }
    
    public static class TypeCheckInliningDecision extends InliningDecision {
        jq_Class expectedType;
        ControlFlowGraph callee;
        public TypeCheckInliningDecision(jq_Method target) {
            this.expectedType = target.getDeclaringClass();
            this.callee = CodeCache.getCode(target);
        }
        public void inlineCall(ControlFlowGraph caller, BasicBlock bb, Quad q) {
            inlineVirtualCallSiteWithTypeCheck(caller, bb, q, callee, expectedType);
        }
        public String toString() { return callee.getMethod().toString(); }
    }
    
    public static class InlineSmallSingleTargetCalls implements Oracle {
        
        protected CallGraph cg;
        int bcThreshold;
        public static final int DEFAULT_THRESHOLD = 25;
        
        public InlineSmallSingleTargetCalls(CallGraph cg) {
            this(cg, DEFAULT_THRESHOLD);
        }
        public InlineSmallSingleTargetCalls(CallGraph cg, int bcThreshold) {
            this.cg = cg;
            this.bcThreshold = bcThreshold;
        }
    
        public InliningDecision shouldInline(ControlFlowGraph caller, BasicBlock bb, Quad callSite) {
            if (TRACE_ORACLE) out.println("Oracle evaluating "+callSite);
            if (Invoke.getMethod(callSite).getMethod().needsDynamicLink(caller.getMethod())) {
                if (TRACE_ORACLE) out.println("Skipping because call site needs dynamic link.");
                return null;
            }
            Invoke i = (Invoke) callSite.getOperator();
            jq_Method target;
            if (i.isVirtual()) {
                ProgramLocation pl = new QuadProgramLocation(caller.getMethod(), callSite);
                Collection c = cg.getTargetMethods(pl);
                if (c.size() != 1) {
                    if (TRACE_ORACLE) out.println("Skipping because call site has multiple targets.");
                    return null;
                }
                target = (jq_Method) c.iterator().next();
            } else {
                target = Invoke.getMethod(callSite).getMethod();
            }
            target.getDeclaringClass().prepare();
            if (target.getBytecode() == null) {
                if (TRACE_ORACLE) out.println("Skipping because target method is native.");
                return null;
            }
            if (target.getBytecode().length > bcThreshold) {
                if (TRACE_ORACLE) out.println("Skipping because target method is too large ("+target.getBytecode().length+").");
                return null;
            }
            if (target == caller.getMethod()) {
                if (TRACE_ORACLE) out.println("Skipping because method is recursive.");
                return null;
            }
            // HACK: for interpreter.
            if (!joeq.Interpreter.QuadInterpreter.interpret_filter.isElement(target)) {
                if (TRACE_ORACLE) out.println("Skipping because the interpreter cannot handle the target method.");
                return null;
            }
            
            if (TRACE_ORACLE) out.println("Oracle says to inline!");
            return new NoCheckInliningDecision(target);
        }

    }
    
    /**
     *     Inline methods whose munged names are read from file methodNameFile (methods.txt). 
     * */
    public static class InlineSelectedCalls implements Oracle {        
        protected CallGraph cg;
        protected HashMap/*<String,null>*/ knownMethods = new HashMap(10);
        public static final String methodNameFile = "methods.txt";
          
        public InlineSelectedCalls(CallGraph cg) {
            this.cg = cg;
            initializeNames();
        }
        
        private void initializeNames() {            
            BufferedReader r = null;
            try {
                r = new BufferedReader(new FileReader(methodNameFile));
                
                String s;
                while ((s = r.readLine()) != null) {
                    if(s.startsWith("#")) continue;
                    if(s.trim().length() == 0) continue;
                    
                    if(s.charAt(s.length()-1) == '\n'){
                        s = s.substring(s.length()-1);
                    }
                    
                    knownMethods.put(s, null);
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (r != null)
                    try {
                        r.close();
                    } catch (IOException e1) {
                        e1.printStackTrace();
                    }
            }
        }

        public InliningDecision shouldInline(ControlFlowGraph caller, BasicBlock bb, Quad callSite) {
            if (TRACE_ORACLE) out.println("Oracle evaluating "+callSite);
            Invoke i = (Invoke) callSite.getOperator();
            jq_Method target;
            if (i.isVirtual()) {
                ProgramLocation pl = new QuadProgramLocation(caller.getMethod(), callSite);
                Collection c = cg.getTargetMethods(pl);
                if (c.size() != 1) {
                    if (TRACE_ORACLE) out.println("Skipping because call site has multiple targets.");
                    return null;
                }
                target = (jq_Method) c.iterator().next();
            } else {
                target = Invoke.getMethod(callSite).getMethod();
            }
            target.getDeclaringClass().prepare();
            if (target.getBytecode() == null) {
                if (TRACE_ORACLE) out.println("Skipping because target method is native.");
                return null;
            }

            if (target == caller.getMethod()) {
                if (TRACE_ORACLE) out.println("Skipping because method is recursive.");
                return null;
            }
            // HACK: for interpreter.
            if (!joeq.Interpreter.QuadInterpreter.interpret_filter.isElement(target)) {
                if (TRACE_ORACLE) out.println("Skipping because the interpreter cannot handle the target method.");
                return null;
            }
            
            String mungedName = PA.mungeMethodName(target);
            if (!knownMethods.containsKey(mungedName)){
                return null;
            }
            
            if (Invoke.getMethod(callSite).getMethod().needsDynamicLink(caller.getMethod())) {
              if (TRACE_ORACLE) out.println("Skipping because call site needs dynamic link.");
              return null;
            }
            
            if (TRACE_ORACLE) out.println("Oracle says to inline!");
            //return new NoCheckInliningDecision(target);
            return new TypeCheckInliningDecision(target);
        }
    }
    
    public void visitCFG(ControlFlowGraph cfg) {
        QuadIterator qi = new QuadIterator(cfg);
        java.util.LinkedList inline_blocks = new java.util.LinkedList();
        java.util.LinkedList inline_quads = new java.util.LinkedList();
        java.util.LinkedList inline_decisions = new java.util.LinkedList();
        while (qi.hasNext()) {
            Quad q = qi.nextQuad();
            if (q.getOperator() instanceof Invoke) {
                BasicBlock bb = qi.getCurrentBasicBlock();
                Invoke.getMethod(q).getMethod().getDeclaringClass().load();
                InliningDecision d = oracle.shouldInline(cfg, bb, q);
                if (d != null) {
                    if (TRACE_DECISIONS) out.println("Decided to inline "+d);
                    inline_blocks.add(bb);
                    inline_quads.add(q);
                    inline_decisions.add(d);
                }
            }
        }
        // do the inlining backwards, so that basic blocks don't change.
        java.util.ListIterator li1 = inline_blocks.listIterator();
        java.util.ListIterator li2 = inline_quads.listIterator();
        java.util.ListIterator li3 = inline_decisions.listIterator();
        while (li1.hasNext()) { li1.next(); li2.next(); li3.next(); }
        while (li1.hasPrevious()) {
            BasicBlock bb = (BasicBlock)li1.previous();
            Quad q = (Quad)li2.previous();
            InliningDecision d = (InliningDecision)li3.previous();
            if (TRACE_DECISIONS) out.println("Performing inlining "+d + " using " + d.getClass());
            d.inlineCall(cfg, bb, q);
            if (cg instanceof CachedCallGraph && d instanceof NoCheckInliningDecision) {
                CachedCallGraph ccg = (CachedCallGraph) cg;
                jq_Method caller = cfg.getMethod();
                ProgramLocation pl = new ProgramLocation.QuadProgramLocation(caller, q);
                jq_Method callee = ((NoCheckInliningDecision) d).callee.getMethod();
                ccg.inlineEdge(caller, pl, callee);
            }
        }
    }
    
    public static void inlineNonVirtualCallSite(ControlFlowGraph caller, BasicBlock bb, Quad q, ControlFlowGraph callee) {
        if (TRACE) out.println("Inlining "+q+" target "+callee.getMethod()+" in "+bb);

        int invokeLocation = bb.getQuadIndex(q);
        Assert._assert(invokeLocation != -1);

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
        Assert._assert(invokeQuad == q);
        Assert._assert(bb.size() == invokeLocation);
        if (TRACE) out.println("Result of splitting:");
        if (TRACE) out.println(bb.fullDump());
        if (TRACE) out.println(successor_bb.fullDump());

        // add instructions to set parameters.
        ParamListOperand plo = Invoke.getParamList(q);
        jq_Type[] params = callee.getMethod().getParamTypes();
        for (int i=0, j=0; i<params.length; ++i, ++j) {
            Move op = Move.getMoveOp(params[i]);
            Register dest_r = callee.getRegisterFactory().getOrCreateLocal(j, params[i]);
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
            Assert._assert(lq.getOperator() instanceof Return);
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
        if (TRACE) out.println(CodeCache.getCode(callee.getMethod()).fullDump());
        if (TRACE) out.println(CodeCache.getCode(callee.getMethod()).getRegisterFactory().fullDump());
    }

    public static void inlineVirtualCallSiteWithTypeCheck(ControlFlowGraph caller, BasicBlock bb, Quad q, ControlFlowGraph callee, jq_Class type) {
        if (TRACE) out.println("Inlining "+q+" in "+bb+" for target "+callee.getMethod());

        int invokeLocation = bb.getQuadIndex(q);
        Assert._assert(invokeLocation != -1);

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
        BasicBlock successor_bb = caller.createBasicBlock(callee.exit().getNumberOfPredecessors() + 1, bb.getNumberOfSuccessors(), bb.size() - invokeLocation, bb.getExceptionHandlers());
        
        Quad fakeCallQuad = q.copy(caller.getNewQuadID());
        Assert._assert(Invoke.getParamList(q).length() == Invoke.getParamList(fakeCallQuad).length());
        MethodOperand fakeOperand = getFakeMethodOperand(Invoke.getMethod(q).getMethod());
        if(fakeOperand != null && true){
            Invoke.setMethod(fakeCallQuad, fakeOperand);
//            Invoke.set
            successor_bb.appendQuad(fakeCallQuad);
        }
        successor_bb.appendQuad(fakeCallQuad);
        
        int bb_size = bb.size();
        for (int i=invokeLocation+1; i<bb_size; ++i) {
            successor_bb.appendQuad(bb.removeQuad(invokeLocation+1));
        }
        Quad invokeQuad = bb.removeQuad(invokeLocation);
        Assert._assert(invokeQuad == q);
        Assert._assert(bb.size() == invokeLocation);
        if (TRACE) out.println("Result of splitting:");
        if (TRACE) out.println(bb.fullDump());
        if (TRACE) out.println(successor_bb.fullDump());

        // create failsafe case block
        BasicBlock bb_fail = caller.createBasicBlock(1, 1, 2, bb.getExceptionHandlers());
        bb_fail.appendQuad(invokeQuad);
        Quad q2 = Goto.create(caller.getNewQuadID(),
                              Goto.GOTO.INSTANCE,
                              new TargetOperand(successor_bb));
        bb_fail.appendQuad(q2);
        bb_fail.addSuccessor(successor_bb);
        successor_bb.addPredecessor(bb_fail);
        if (TRACE) out.println("Fail-safe block:");
        if (TRACE) out.println(bb_fail.fullDump());
        
        // create success case block
        BasicBlock bb_success = caller.createBasicBlock(1, 1, Invoke.getParamList(q).length(), bb.getExceptionHandlers());
        
        // add test.
        jq_Type dis_t = Invoke.getParam(q, 0).getType();
        RegisterOperand dis_op = new RegisterOperand(Invoke.getParam(q, 0).getRegister(), dis_t);
        Register res = caller.getRegisterFactory().getOrCreateStack(0, jq_Primitive.BOOLEAN);
        RegisterOperand res_op = new RegisterOperand(res, jq_Primitive.BOOLEAN);
        TypeOperand type_op = new TypeOperand(type);
        q2 = InstanceOf.create(caller.getNewQuadID(), InstanceOf.INSTANCEOF.INSTANCE, res_op, dis_op, type_op);
        bb.appendQuad(q2);
        res_op = new RegisterOperand(res, jq_Primitive.BOOLEAN);
        IConstOperand zero_op = new IConstOperand(0);
        ConditionOperand ne_op = new ConditionOperand(BytecodeVisitor.CMP_NE);
        TargetOperand success_op = new TargetOperand(bb_success);
        q2 = IntIfCmp.create(caller.getNewQuadID(), IntIfCmp.IFCMP_I.INSTANCE, res_op, zero_op, ne_op, success_op);
        bb.appendQuad(q2);
        
        // add instructions to set parameters.
        ParamListOperand plo = Invoke.getParamList(q);
        jq_Type[] params = callee.getMethod().getParamTypes();
        for (int i=0, j=0; i<params.length; ++i, ++j) {
            Move op = Move.getMoveOp(params[i]);
            Register dest_r = callee.getRegisterFactory().getOrCreateLocal(j, params[i]);
            RegisterOperand dest = new RegisterOperand(dest_r, params[i]);
            Register src_r = plo.get(i).getRegister();
            RegisterOperand src = new RegisterOperand(src_r, params[i]);
            q2 = Move.create(caller.getNewQuadID(), op, dest, src);
            bb_success.appendQuad(q2);
            if (params[i].getReferenceSize() == 8) ++j;
        }
        if (TRACE) out.println("Success block after adding parameter moves:");
        if (TRACE) out.println(bb_success.fullDump());

        // replace return instructions with moves and gotos, and
        // finish splicing in the control flow graph.
        for (ListIterator.BasicBlock it = bb.getSuccessors().basicBlockIterator();
             it.hasNext(); ) {
            BasicBlock next_bb = it.nextBasicBlock();
            next_bb.removePredecessor(bb);
            next_bb.addPredecessor(successor_bb);
            successor_bb.addSuccessor(next_bb);
        }
        bb_success.addSuccessor(callee.entry().getFallthroughSuccessor());
        callee.entry().getFallthroughSuccessor().removeAllPredecessors();
        callee.entry().getFallthroughSuccessor().addPredecessor(bb_success);
        bb.removeAllSuccessors();
        bb.addSuccessor(bb_fail);
        bb_fail.addPredecessor(bb);
        bb.addSuccessor(bb_success);
        bb_success.addPredecessor(bb);
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
            Assert._assert(lq.getOperator() instanceof Return);
            pred_bb.removeQuad(lq);
            if (dest_r != null) {
                RegisterOperand ldest = new RegisterOperand(dest_r, callee.getMethod().getReturnType());
                Operand src = Return.getSrc(lq).copy();
                Quad q3 = Move.create(caller.getNewQuadID(), op, ldest, src);
                pred_bb.appendQuad(q3);
            }
            q2 = Goto.create(caller.getNewQuadID(),
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
        if (TRACE) out.println(CodeCache.getCode(callee.getMethod()).fullDump());
        if (TRACE) out.println(CodeCache.getCode(callee.getMethod()).getRegisterFactory().fullDump());
    }
    
    /**
     * Get replacement method for @param originalMethod.
     * This method caches the result it returns.
     * */
    private static MethodOperand getFakeMethodOperand(jq_Method originalMethod) {
        if(fakeMethodOperand == null){
            jq_Method newMethod = null;
            jq_Class fakeString = (jq_Class)jq_Type.parseType("MyMockLib.MyString");
            jq_Class fakeStringBuffer = (jq_Class)jq_Type.parseType("MyMockLib.MyStringBuffer");
            fakeString.prepare(); fakeStringBuffer.prepare();
            
            if(originalMethod.getDeclaringClass().getName().equals("java.lang.String")){            
                newMethod = fakeString.getDeclaredMethod(originalMethod.getName().toString());
                Assert._assert(newMethod != null, "Missing method " + originalMethod.getName() + " in " + fakeString);
            }else
            if(originalMethod.getDeclaringClass().getName().equals("java.lang.StringBuffer")){            
                newMethod = fakeStringBuffer.getDeclaredMethod(originalMethod.getName().toString());
                Assert._assert(newMethod != null, "Missing method " + originalMethod.getName() + " in " + fakeStringBuffer);
            }else{
                System.err.println("Unexpected method " + originalMethod);
                return null;
            }
            System.out.println("Replaced " + originalMethod + " with " + newMethod);
            fakeMethodOperand = new MethodOperand(newMethod);
        }
        
        return fakeMethodOperand;
    }
}

// BuildBDDIR.java, created Mar 17, 2004 2:43:55 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.BDD;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;

import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.ControlFlowGraphVisitor;
import joeq.Compiler.Quad.Operand;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.QuadIterator;
import joeq.Compiler.Quad.QuadVisitor;
import joeq.Compiler.Quad.Operand.Const4Operand;
import joeq.Compiler.Quad.Operand.FieldOperand;
import joeq.Compiler.Quad.Operand.MethodOperand;
import joeq.Compiler.Quad.Operand.RegisterOperand;
import joeq.Compiler.Quad.Operand.TargetOperand;
import joeq.Compiler.Quad.Operand.TypeOperand;
import joeq.Compiler.Quad.Operator.ALength;
import joeq.Compiler.Quad.Operator.ALoad;
import joeq.Compiler.Quad.Operator.AStore;
import joeq.Compiler.Quad.Operator.Binary;
import joeq.Compiler.Quad.Operator.BoundsCheck;
import joeq.Compiler.Quad.Operator.CheckCast;
import joeq.Compiler.Quad.Operator.Getfield;
import joeq.Compiler.Quad.Operator.Getstatic;
import joeq.Compiler.Quad.Operator.Goto;
import joeq.Compiler.Quad.Operator.InstanceOf;
import joeq.Compiler.Quad.Operator.IntIfCmp;
import joeq.Compiler.Quad.Operator.Invoke;
import joeq.Compiler.Quad.Operator.Jsr;
import joeq.Compiler.Quad.Operator.Monitor;
import joeq.Compiler.Quad.Operator.Move;
import joeq.Compiler.Quad.Operator.New;
import joeq.Compiler.Quad.Operator.NewArray;
import joeq.Compiler.Quad.Operator.NullCheck;
import joeq.Compiler.Quad.Operator.Putfield;
import joeq.Compiler.Quad.Operator.Putstatic;
import joeq.Compiler.Quad.Operator.Ret;
import joeq.Compiler.Quad.Operator.Return;
import joeq.Compiler.Quad.Operator.StoreCheck;
import joeq.Compiler.Quad.Operator.Unary;
import joeq.Compiler.Quad.Operator.ZeroCheck;
import joeq.Compiler.Quad.RegisterFactory.Register;
import joeq.Util.Assert;
import joeq.Util.Collections.IndexMap;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;

/**
 * BuildBDDIR
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BuildBDDIR extends QuadVisitor.EmptyVisitor implements ControlFlowGraphVisitor {
    
    IndexMap opMap;
    IndexMap quadMap;
    IndexMap regMap;
    IndexMap memberMap;
    
    BDDFactory bdd;
    BDDDomain quad, opc, dest, src1, src2, constant, fallthrough, target, member;
    BDD allQuads;
    BDD currentQuad;
    
    public BuildBDDIR() {
        opMap = new IndexMap("op");
        quadMap = new IndexMap("quad");
        regMap = new IndexMap("reg");
        memberMap = new IndexMap("member");
        bdd = BDDFactory.init(1000000, 10000);
        quad = makeDomain("quad", 14);
        opc = makeDomain("opc", 8);
        dest = makeDomain("dest", 13);
        src1 = makeDomain("src1", 13);
        src2 = makeDomain("src2", 13);
        constant = makeDomain("constant", 32);
        fallthrough = makeDomain("fallthrough", 14);
        target = makeDomain("target", 14);
        member = makeDomain("member", 11);
        allQuads = bdd.zero();
    }
    
    void handleTarget(TargetOperand top) {
        Quad q = top.getTarget().getQuad(0);
        int qid = quadMap.get(q);
        currentQuad.andWith(target.ithVar(qid));
    }
    void handleConst(Const4Operand cop) {
        long val = ((long) cop.getBits()) & 0xFFFFFFFFL;
        currentQuad.andWith(constant.ithVar(val));
    }
    void handleDest(RegisterOperand rop) {
        if (rop == null) return;
        Register r = rop.getRegister();
        int rid = regMap.get(r);
        currentQuad.andWith(dest.ithVar(rid));
    }
    void handleSrc1(Operand op) {
        if (op instanceof RegisterOperand) {
            RegisterOperand rop = (RegisterOperand) op;
            Register r = rop.getRegister();
            int rid = regMap.get(r);
            currentQuad.andWith(src1.ithVar(rid));
        } else if (op instanceof Const4Operand) {
            Const4Operand cop = (Const4Operand) op;
            handleConst(cop);
        }
    }
    void handleSrc2(Operand op) {
        if (op instanceof RegisterOperand) {
            RegisterOperand rop = (RegisterOperand) op;
            Register r = rop.getRegister();
            int rid = regMap.get(r);
            currentQuad.andWith(src2.ithVar(rid));
        } else if (op instanceof Const4Operand) {
            Const4Operand cop = (Const4Operand) op;
            handleConst(cop);
        }
    }
    void handleMember(Operand op) {
        Object o;
        if (op instanceof FieldOperand) {
            o = ((FieldOperand) op).getField();
        } else if (op instanceof MethodOperand) {
            o = ((MethodOperand) op).getMethod();
        } else if (op instanceof TypeOperand) {
            o = ((TypeOperand) op).getType();
        } else {
            Assert.UNREACHABLE();
            return;
        }
        int mid = memberMap.get(o);
        currentQuad.andWith(member.ithVar(mid));
    }
    
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitALength(joeq.Compiler.Quad.Quad)
     */
    public void visitALength(Quad obj) {
        super.visitALength(obj);
        handleDest(ALength.getDest(obj));
        handleSrc1(ALength.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitALoad(joeq.Compiler.Quad.Quad)
     */
    public void visitALoad(Quad obj) {
        super.visitALoad(obj);
        handleDest(ALoad.getDest(obj));
        handleSrc1(ALoad.getBase(obj));
        handleSrc2(ALoad.getIndex(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitAStore(joeq.Compiler.Quad.Quad)
     */
    public void visitAStore(Quad obj) {
        super.visitAStore(obj);
        handleDest((RegisterOperand) AStore.getBase(obj));
        handleSrc1(AStore.getIndex(obj));
        handleSrc2(AStore.getValue(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitBinary(joeq.Compiler.Quad.Quad)
     */
    public void visitBinary(Quad obj) {
        super.visitBinary(obj);
        handleDest(Binary.getDest(obj));
        handleSrc1(Binary.getSrc1(obj));
        handleSrc2(Binary.getSrc2(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitBoundsCheck(joeq.Compiler.Quad.Quad)
     */
    public void visitBoundsCheck(Quad obj) {
        super.visitBoundsCheck(obj);
        handleSrc1(BoundsCheck.getRef(obj));
        handleSrc2(BoundsCheck.getIndex(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitCheckCast(joeq.Compiler.Quad.Quad)
     */
    public void visitCheckCast(Quad obj) {
        super.visitCheckCast(obj);
        handleDest(CheckCast.getDest(obj));
        handleSrc1(CheckCast.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitGetfield(joeq.Compiler.Quad.Quad)
     */
    public void visitGetfield(Quad obj) {
        super.visitGetfield(obj);
        handleDest(Getfield.getDest(obj));
        handleSrc1(Getfield.getBase(obj));
        handleMember(Getfield.getField(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitGetstatic(joeq.Compiler.Quad.Quad)
     */
    public void visitGetstatic(Quad obj) {
        super.visitGetstatic(obj);
        handleDest(Getstatic.getDest(obj));
        handleMember(Getstatic.getField(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitGoto(joeq.Compiler.Quad.Quad)
     */
    public void visitGoto(Quad obj) {
        super.visitGoto(obj);
        handleTarget(Goto.getTarget(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitInstanceOf(joeq.Compiler.Quad.Quad)
     */
    public void visitInstanceOf(Quad obj) {
        super.visitInstanceOf(obj);
        handleDest(InstanceOf.getDest(obj));
        handleSrc1(InstanceOf.getSrc(obj));
        handleMember(InstanceOf.getType(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitIntIfCmp(joeq.Compiler.Quad.Quad)
     */
    public void visitIntIfCmp(Quad obj) {
        super.visitIntIfCmp(obj);
        handleSrc1(IntIfCmp.getSrc1(obj));
        handleSrc2(IntIfCmp.getSrc2(obj));
        handleTarget(IntIfCmp.getTarget(obj));
        // todo: condition code.
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitInvoke(joeq.Compiler.Quad.Quad)
     */
    public void visitInvoke(Quad obj) {
        super.visitInvoke(obj);
        handleDest(Invoke.getDest(obj));
        handleMember(Invoke.getMethod(obj));
        // todo: parameter list
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitJsr(joeq.Compiler.Quad.Quad)
     */
    public void visitJsr(Quad obj) {
        super.visitJsr(obj);
        handleDest(Jsr.getDest(obj));
        handleTarget(Jsr.getTarget(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitLookupSwitch(joeq.Compiler.Quad.Quad)
     */
    public void visitLookupSwitch(Quad obj) {
        super.visitLookupSwitch(obj);
        // todo
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitMonitor(joeq.Compiler.Quad.Quad)
     */
    public void visitMonitor(Quad obj) {
        super.visitMonitor(obj);
        handleSrc1(Monitor.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitMove(joeq.Compiler.Quad.Quad)
     */
    public void visitMove(Quad obj) {
        super.visitMove(obj);
        handleDest(Move.getDest(obj));
        handleSrc1(Move.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitNew(joeq.Compiler.Quad.Quad)
     */
    public void visitNew(Quad obj) {
        super.visitNew(obj);
        handleDest(New.getDest(obj));
        handleMember(New.getType(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitNewArray(joeq.Compiler.Quad.Quad)
     */
    public void visitNewArray(Quad obj) {
        super.visitNewArray(obj);
        handleDest(NewArray.getDest(obj));
        handleSrc1(NewArray.getSize(obj));
        handleMember(NewArray.getType(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitNullCheck(joeq.Compiler.Quad.Quad)
     */
    public void visitNullCheck(Quad obj) {
        super.visitNullCheck(obj);
        handleDest((RegisterOperand) NullCheck.getDest(obj));
        handleSrc1(NullCheck.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitPutfield(joeq.Compiler.Quad.Quad)
     */
    public void visitPutfield(Quad obj) {
        super.visitPutfield(obj);
        handleSrc1(Putfield.getBase(obj));
        handleSrc2(Putfield.getSrc(obj));
        handleMember(Putfield.getField(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitPutstatic(joeq.Compiler.Quad.Quad)
     */
    public void visitPutstatic(Quad obj) {
        super.visitPutstatic(obj);
        handleSrc2(Putstatic.getSrc(obj));
        handleMember(Putstatic.getField(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitRet(joeq.Compiler.Quad.Quad)
     */
    public void visitRet(Quad obj) {
        super.visitRet(obj);
        handleSrc1(Ret.getTarget(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitReturn(joeq.Compiler.Quad.Quad)
     */
    public void visitReturn(Quad obj) {
        super.visitReturn(obj);
        handleSrc1(Return.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitSpecial(joeq.Compiler.Quad.Quad)
     */
    public void visitSpecial(Quad obj) {
        super.visitSpecial(obj);
        // todo.
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitStoreCheck(joeq.Compiler.Quad.Quad)
     */
    public void visitStoreCheck(Quad obj) {
        super.visitStoreCheck(obj);
        handleSrc1(StoreCheck.getRef(obj));
        handleSrc2(StoreCheck.getElement(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitTableSwitch(joeq.Compiler.Quad.Quad)
     */
    public void visitTableSwitch(Quad obj) {
        super.visitTableSwitch(obj);
        // todo.
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitTypeCheck(joeq.Compiler.Quad.Quad)
     */
    public void visitTypeCheck(Quad obj) {
        super.visitTypeCheck(obj);
        handleDest((RegisterOperand)ZeroCheck.getDest(obj));
        handleSrc1(ZeroCheck.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitUnary(joeq.Compiler.Quad.Quad)
     */
    public void visitUnary(Quad obj) {
        super.visitUnary(obj);
        handleDest(Unary.getDest(obj));
        handleSrc1(Unary.getSrc(obj));
    }
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.QuadVisitor#visitZeroCheck(joeq.Compiler.Quad.Quad)
     */
    public void visitZeroCheck(Quad obj) {
        super.visitZeroCheck(obj);
        handleDest((RegisterOperand)ZeroCheck.getDest(obj));
        handleSrc1(ZeroCheck.getSrc(obj));
    }
    
    BDDDomain makeDomain(String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.ControlFlowGraphVisitor#visitCFG(joeq.Compiler.Quad.ControlFlowGraph)
     */
    public void visitCFG(ControlFlowGraph cfg) {
        QuadIterator i = new QuadIterator(cfg);
        while (i.hasNext()) {
            Quad q = i.nextQuad();
            currentQuad = bdd.one();
            int quadID = quadMap.get(q);
            //System.out.println("Quad id: "+quadID);
            currentQuad.andWith(quad.ithVar(quadID));
            int opID = opMap.get(q.getOperator());
            currentQuad.andWith(opc.ithVar(opID));
            q.accept(this);
            BDD succ = bdd.zero();
            for (Iterator j = i.successors(); j.hasNext(); ) {
                Quad q2 = (Quad) j.next();
                int quad2ID = quadMap.get(q2);
                succ.orWith(fallthrough.ithVar(quad2ID));
            }
            currentQuad.andWith(succ);
            //printQuad(currentQuad);
            allQuads.orWith(currentQuad);
        }
        try {
            print();
            dump();
        } catch (IOException x) {
        }
    }
    
    public void dump() throws IOException {
        bdd.save("quads.bdd", allQuads);
        dumpTuples("quads.tuples", allQuads);
        dumpMap(quadMap, "quads.map");
        dumpMap(opMap, "op.map");
        dumpMap(regMap, "reg.map");
        dumpMap(memberMap, "member.map");
    }
    
    void dumpMap(IndexMap map, String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        for (int i = 0; i < map.size(); ++i) {
            Object o = map.get(i);
            dos.writeBytes(o + "\n");
        }
        dos.close();
    }
    
    void dumpTuples(String fileName, BDD allQ) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        for (int i = 0; i < quadMap.size(); ++i) {
            BDD q = quad.ithVar(i).andWith(allQ.id());
            while (!q.isZero()) {
                long[] v = q.scanAllVar();
                BDD t = bdd.one();
                for (int j = 0; j < v.length; ++j) {
                    BDDDomain d = bdd.getDomain(j);
                    if (quantifyOtherDomains(q, d).isOne()) {
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                    } else {
                        dos.writeBytes(v[j]+" ");
                        t.andWith(bdd.getDomain(j).ithVar(v[j]));
                    }
                }
                q.applyWith(t, BDDFactory.diff);
                dos.writeBytes("\n");
            }
        }
        dos.close();
    }
    
    BDD quantifyOtherDomains(BDD q, BDDDomain d) {
        BDD result = q.id();
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            if (i == d.getIndex()) continue;
            BDD r2 = result.exist(bdd.getDomain(i).set());
            result.free();
            result = r2;
        }
        return result;
    }

    void print() {
        for (int i = 0; i < quadMap.size(); ++i) {
            BDD q = quad.ithVar(i).andWith(allQuads.id());
            printQuad(q);
        }
    }
    
    void print2() {
        // quad, opc, dest, src1, src2, constant, fallthrough, target, member;
        BDD domains = opc.set().andWith(dest.set()).andWith(src1.set()).andWith(src2.set());
        domains.andWith(constant.set()).andWith(fallthrough.set()).andWith(target.set()).andWith(member.set());
        BDD quad_ids = allQuads.exist(domains);
        for (Iterator i = quad_ids.iterator(quad.set()); i.hasNext(); ) {
            BDD q = (BDD) i.next();
            q.andWith(allQuads.id());
            printQuad(q);
        }
    }
    
    void printQuad(BDD q) {
        long id = q.scanVar(quad);
        if (id == -1) return;
        System.out.println("Quad id "+id);
        System.out.println("        "+quadMap.get((int) id));
        System.out.println(q.toStringWithDomains());
    }
}

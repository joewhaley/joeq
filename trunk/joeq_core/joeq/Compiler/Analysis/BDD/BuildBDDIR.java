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
import joeq.Compiler.Quad.Operand.ConstOperand;
import joeq.Compiler.Quad.Operand.FieldOperand;
import joeq.Compiler.Quad.Operand.MethodOperand;
import joeq.Compiler.Quad.Operand.RegisterOperand;
import joeq.Compiler.Quad.Operand.TargetOperand;
import joeq.Compiler.Quad.Operand.TypeOperand;
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
    
    IndexMap methodMap;
    IndexMap opMap;
    IndexMap quadMap;
    IndexMap regMap;
    IndexMap memberMap;
    IndexMap constantMap;
    
    String varOrderDesc = "method_quadxtargetxfallthrough_member_constant_src2_opc_src1_dest";
    
    int methodBits = 14, quadBits = 18, opBits = 8, regBits = 7, constantBits = 13, memberBits = 14;

    BDDFactory bdd;
    BDDDomain method, quad, opc, dest, src1, src2, constant, fallthrough, target, member;
    BDD methodToQuad;
    BDD allQuads;
    BDD currentQuad;
    
    int totalQuads;
    
    boolean GLOBAL_QUAD_NUMBERS = true;
    
    public BuildBDDIR() {
        if (!GLOBAL_QUAD_NUMBERS) {
            quadBits = 13;
        }
        methodMap = new IndexMap("method");
        opMap = new IndexMap("op");
        quadMap = new IndexMap("quad");
        regMap = new IndexMap("reg");
        memberMap = new IndexMap("member");
        constantMap = new IndexMap("constant");
        bdd = BDDFactory.init(1000000, 50000);
        method = makeDomain("method", methodBits);
        quad = makeDomain("quad", quadBits);
        opc = makeDomain("opc", opBits);
        dest = makeDomain("dest", regBits);
        src1 = makeDomain("src1", regBits);
        src2 = makeDomain("src2", regBits);
        constant = makeDomain("constant", constantBits);
        fallthrough = makeDomain("fallthrough", quadBits);
        target = makeDomain("target", quadBits);
        member = makeDomain("member", memberBits);
        allQuads = bdd.zero();
        methodToQuad = bdd.zero();
        int [] varOrder = bdd.makeVarOrdering(true, varOrderDesc);
        bdd.setVarOrder(varOrder);
        bdd.setMaxIncrease(500000);
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
        int methodID = methodMap.get(cfg.getMethod());
        
        if (!GLOBAL_QUAD_NUMBERS) quadMap.clear();
        
        long time = System.currentTimeMillis();
        
        while (i.hasNext()) {
            Quad q = i.nextQuad();
            currentQuad = bdd.one();
            int quadID = quadMap.get(q)+1;
            //System.out.println("Quad id: "+quadID);
            currentQuad.andWith(quad.ithVar(quadID));
            if (!GLOBAL_QUAD_NUMBERS) {
                currentQuad.andWith(method.ithVar(methodID));
                methodToQuad.orWith(currentQuad.id());
            } else {
                methodToQuad.orWith(currentQuad.and(method.ithVar(methodID)));
            }
            int opID = opMap.get(q.getOperator())+1;
            currentQuad.andWith(opc.ithVar(opID));
            handleQuad(q);
            BDD succ = bdd.zero();
            for (Iterator j = i.successors(); j.hasNext(); ) {
                Quad q2 = (Quad) j.next();
                int quad2ID = quadMap.get(q2)+1;
                succ.orWith(fallthrough.ithVar(quad2ID));
            }
            currentQuad.andWith(succ);
            //printQuad(currentQuad);
            allQuads.orWith(currentQuad);
        }
        
        time = System.currentTimeMillis() - time;
        time += totalTime;
        System.out.println("Method: " + cfg.getMethod() + " time: " + time);
        int qSize = totalQuads;
        int nodes = allQuads.nodeCount();
        System.out.println("Quads: " +qSize+", nodes: "+nodes+", average: "+(float)nodes/qSize);
    }
    
    long totalTime;
    
    public String toString() {
        System.out.println("Total time spent building representation: "+totalTime);
        System.out.println("allQuads, node count: " + allQuads.nodeCount());
        System.out.println("methodToQuad, node count: " + methodToQuad.nodeCount());
        
        System.out.println("methodMap size: " + methodMap.size());
        System.out.println("opMap size: " + opMap.size());
        System.out.println("quadMap size: " + quadMap.size());
        System.out.println("regMap size: " + regMap.size());
        System.out.println("memberMap size: " + memberMap.size());
        System.out.println("constantMap size: " + constantMap.size());
        
        try {
            //print();
            dump();
        } catch (IOException x) {
        }
        return ("BuildBDDIR, node count: " + allQuads.nodeCount());
    }
    
    public static boolean ZERO_FIELDS = false;
    
    void handleQuad(Quad q) {
        int quadID=0, opcID=0, destID=0, src1ID=0, src2ID=0, constantID=0, fallthroughID=0, targetID=0, memberID=0;
        quadID = quadMap.get(q)+1;
        opcID = opMap.get(q.getOperator())+1;
        Iterator i = q.getDefinedRegisters().iterator();
        if (i.hasNext()) {
            destID = regMap.get(((RegisterOperand)i.next()).getRegister().toString())+1;
            Assert._assert(!i.hasNext());
        }
        i = q.getUsedRegisters().iterator();
        if (i.hasNext()) {
            src1ID = regMap.get(((RegisterOperand)i.next()).getRegister().toString())+1;
            if (i.hasNext()) {
                src2ID = regMap.get(((RegisterOperand)i.next()).getRegister().toString())+1;
            }
        }
        i = q.getAllOperands().iterator();
        while (i.hasNext()) {
            Operand op = (Operand) i.next();
            if (op instanceof RegisterOperand) continue;
            else if (op instanceof ConstOperand) {
                constantID = constantMap.get(((ConstOperand) op).getWrapped());
            } else if (op instanceof TargetOperand) {
                targetID = quadMap.get(((TargetOperand) op).getTarget().getQuad(0))+1;
            } else if (op instanceof FieldOperand) {
                memberID = memberMap.get(((FieldOperand) op).getField())+1;
            } else if (op instanceof MethodOperand) {
                memberID = memberMap.get(((MethodOperand) op).getMethod())+1;
            } else if (op instanceof TypeOperand) {
                memberID = memberMap.get(((TypeOperand) op).getType())+1;
            }
        }
        if (ZERO_FIELDS || quadID != 0) currentQuad.andWith(quad.ithVar(quadID));
        if (ZERO_FIELDS || opcID != 0) currentQuad.andWith(opc.ithVar(opcID));
        if (ZERO_FIELDS || destID != 0) currentQuad.andWith(dest.ithVar(destID));
        if (ZERO_FIELDS || src1ID != 0) currentQuad.andWith(src1.ithVar(src1ID));
        if (ZERO_FIELDS || src2ID != 0) currentQuad.andWith(src2.ithVar(src2ID));
        if (ZERO_FIELDS || constantID != 0) currentQuad.andWith(constant.ithVar(((long)constantID) & 0xFFFFFFFFL));
        //currentQuad.andWith(fallthrough.ithVar(fallthroughID));
        if (ZERO_FIELDS || targetID != 0) currentQuad.andWith(target.ithVar(targetID));
        if (ZERO_FIELDS || memberID != 0) currentQuad.andWith(member.ithVar(memberID));
        ++totalQuads;
    }
    
    public void dump() throws IOException {
        System.out.println("Var order: "+varOrderDesc);
        dumpMap(quadMap, "quad.map");
        dumpMap(opMap, "op.map");
        dumpMap(regMap, "reg.map");
        dumpMap(memberMap, "member.map");
        dumpMap(constantMap, "constant.map");
        dumpBDDConfig("bdd.cfg");
        dumpFieldDomains("fielddomains.cfg");
        dumpRelations("relations.cfg");
        System.out.print("Saving BDD...");
        bdd.save("cfg.bdd", allQuads);
        bdd.save("m2q.bdd", methodToQuad);
        System.out.println("done.");
        dumpTuples("cfg.tuples", allQuads);
        dumpTuples("m2q.tuples", methodToQuad);
    }
    
    void dumpBDDConfig(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        dos.writeBytes("method "+(1L<<methodBits)+"\n");
        dos.writeBytes("quad "+(1L<<quadBits)+"\n");
        dos.writeBytes("opc "+(1L<<opBits)+"\n");
        dos.writeBytes("dest "+(1L<<regBits)+"\n");
        dos.writeBytes("src1 "+(1L<<regBits)+"\n");
        dos.writeBytes("src2 "+(1L<<regBits)+"\n");
        dos.writeBytes("constant "+(1L<<constantBits)+"\n");
        dos.writeBytes("fallthrough "+(1L<<quadBits)+"\n");
        dos.writeBytes("target "+(1L<<quadBits)+"\n");
        dos.writeBytes("member "+(1L<<memberBits)+"\n");
        dos.close();
    }
    
    void dumpFieldDomains(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        dos.writeBytes("method "+(1L<<methodBits)+"\n");
        dos.writeBytes("quad "+(1L<<quadBits)+"\n");
        dos.writeBytes("op "+(1L<<opBits)+"\n");
        dos.writeBytes("reg "+(1L<<regBits)+"\n");
        dos.writeBytes("constant "+(1L<<constantBits)+"\n");
        dos.writeBytes("member "+(1L<<memberBits)+"\n");
        dos.close();
    }
    
    void dumpRelations(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        if (GLOBAL_QUAD_NUMBERS) {
            dos.writeBytes("cfg ( id : quad , op : op , dest : reg , src1 : reg , src2 : reg , const : constant , fallthrough : quad , target : quad , member : member )\n");
        } else {
            dos.writeBytes("cfg ( method : method , id : quad , op : op , dest : reg , src1 : reg , src2 : reg , const : constant , fallthrough : quad , target : quad , member : member )\n");
        }
        dos.writeBytes("m2q ( method : method , id : quad )\n");
        dos.close();
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
                    BDD r = quantifyOtherDomains(q, d);
                    if (r.isOne()) {
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                    } else {
                        dos.writeBytes(v[j]+" ");
                        t.andWith(bdd.getDomain(j).ithVar(v[j]));
                    }
                    r.free();
                }
                q.applyWith(t, BDDFactory.diff);
                dos.writeBytes("\n");
            }
            q.free();
        }
        dos.close();
    }
    
    BDD quantifyOtherDomains(BDD q, BDDDomain d) {
        BDD result = q.id();
        BDD set = bdd.one();
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            if (i == d.getIndex()) continue;
            set.andWith(bdd.getDomain(i).set());
        }
        BDD r2 = result.exist(set);
        result.free(); set.free();
        return r2;
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

// BuildBDDIR.java, created Mar 17, 2004 2:43:55 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.BDD;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.ControlFlowGraphVisitor;
import joeq.Compiler.Quad.Operand;
import joeq.Compiler.Quad.Operator;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.QuadIterator;
import joeq.Compiler.Quad.QuadVisitor;
import joeq.Compiler.Quad.Operand.ConstOperand;
import joeq.Compiler.Quad.Operand.FieldOperand;
import joeq.Compiler.Quad.Operand.MethodOperand;
import joeq.Compiler.Quad.Operand.RegisterOperand;
import joeq.Compiler.Quad.Operand.TargetOperand;
import joeq.Compiler.Quad.Operand.TypeOperand;
import joeq.Compiler.Quad.RegisterFactory.Register;
import joeq.Compiler.Quad.SSA.EnterSSA;
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
    //IndexMap regMap;
    IndexMap memberMap;
    IndexMap constantMap;
    
    String varOrderDesc = "method_quadxtargetxfallthrough_member_constant_src2_opc_src1_dest_srcs_srcNum";
    
    int methodBits = 14, quadBits = 18, opBits = 8, regBits = 7, constantBits = 13, memberBits = 14, varargsBits = 4;

    BDDFactory bdd;
    BDDDomain method, quad, opc, dest, src1, src2, constant, fallthrough, target, member, srcNum, srcs;
    BDD methodToQuad;
    BDD methodEntries;
    BDD nullConstant;
    BDD nonNullConstants;
    BDD allQuads;
    BDD currentQuad;
    
    int totalQuads;
    
    boolean ZERO_FIELDS = !System.getProperty("zerofields", "yes").equals("no");
    boolean GLOBAL_QUAD_NUMBERS = !System.getProperty("globalquadnumber", "yes").equals("no");
    boolean SSA = !System.getProperty("ssa", "no").equals("no");
    
    public BuildBDDIR() {
        if (!GLOBAL_QUAD_NUMBERS) {
            quadBits = 13;
        }
        if (SSA) {
            regBits = 11;
            varargsBits = 5;
        }
        methodMap = new IndexMap("method");
        opMap = new IndexMap("op");
        quadMap = new IndexMap("quad");
        //regMap = new IndexMap("reg");
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
        srcNum = makeDomain("srcNum", varargsBits);
        srcs = makeDomain("srcs", regBits);
        allQuads = bdd.zero();
        methodToQuad = bdd.zero();
        methodEntries = bdd.zero();
        nullConstant = bdd.zero();
        nonNullConstants = bdd.zero();
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
        if (SSA) {
            new EnterSSA().visitCFG(cfg);
        }
        QuadIterator i = new QuadIterator(cfg);
        int methodID = methodMap.get(cfg.getMethod());
        
        if (!GLOBAL_QUAD_NUMBERS) quadMap.clear();
        
        long time = System.currentTimeMillis();
        
        boolean firstQuad = true;
        
        while (i.hasNext()) {
            Quad q = i.nextQuad();
            currentQuad = bdd.one();
            int quadID = getQuadID(q);
            //System.out.println("Quad id: "+quadID);
            
            // first quad visited is the entry point
            if (firstQuad) {
                methodEntries.orWith(method.ithVar(methodID).and(quad.ithVar(quadID)));
                firstQuad = false;
            }
            
            currentQuad.andWith(quad.ithVar(quadID));
            if (!GLOBAL_QUAD_NUMBERS) {
                currentQuad.andWith(method.ithVar(methodID));
                methodToQuad.orWith(currentQuad.id());
            } else {
                methodToQuad.orWith(currentQuad.and(method.ithVar(methodID)));
            }
            int opID = getOpID(q.getOperator());
            currentQuad.andWith(opc.ithVar(opID));
            handleQuad(q);
            BDD succ = bdd.zero();
            Iterator j = i.successors();
            if (!j.hasNext()) {
                succ.orWith(fallthrough.ithVar(0));
            } else do {
                Quad q2 = (Quad) j.next();
                int quad2ID = getQuadID(q2);
                succ.orWith(fallthrough.ithVar(quad2ID));
            } while (j.hasNext());
            currentQuad.andWith(succ);
            //printQuad(currentQuad);
            allQuads.orWith(currentQuad);
        }
        
        time = System.currentTimeMillis() - time;
        totalTime += time;
        System.out.println("Method: " + cfg.getMethod() + " time: " + time);
        int qSize = totalQuads;
        int nodes = allQuads.nodeCount();
        System.out.println("Quads: " +qSize+", nodes: "+nodes+", average: "+(float)nodes/qSize);
    }
    
    long totalTime;
    
    public String toString() {
        buildNullConstantBdds();
        
        System.out.println("Total time spent building representation: "+totalTime);
        System.out.println("allQuads, node count: " + allQuads.nodeCount());
        System.out.println("methodToQuad, node count: " + methodToQuad.nodeCount());
        
        System.out.println("methodMap size: " + methodMap.size());
        System.out.println("opMap size: " + opMap.size());
        System.out.println("quadMap size: " + quadMap.size());
        //System.out.println("regMap size: " + regMap.size());
        System.out.println("memberMap size: " + memberMap.size());
        System.out.println("constantMap size: " + constantMap.size());
        
        try {
            //print();
            dump();
        } catch (IOException x) {
        }
        return ("BuildBDDIR, node count: " + allQuads.nodeCount());
    }
    
    public int getRegisterID(Register r) {
        return r.getNumber()+1;
    }
    
    public int getConstantID(Object c) {
        return constantMap.get(c)+1;
    }
    
    public int getQuadID(Quad r) {
        return quadMap.get(r)+1;
    }
    
    public int getMemberID(Object r) {
        return memberMap.get(r)+1;
    }
    
    public int getOpID(Operator r) {
        return opMap.get(r)+1;
    }
    
    void handleQuad(Quad q) {
        int quadID=0, opcID=0, destID=0, src1ID=0, src2ID=0, constantID=0, fallthroughID=0, targetID=0, memberID=0;
        List srcsID = null;
        quadID = getQuadID(q);
        opcID = getOpID(q.getOperator());
        Iterator i = q.getDefinedRegisters().iterator();
        if (i.hasNext()) {
            destID = getRegisterID(((RegisterOperand)i.next()).getRegister());
            Assert._assert(!i.hasNext());
        }
        i = q.getUsedRegisters().iterator();
        if (i.hasNext()) {
            RegisterOperand rop;
            rop = (RegisterOperand) i.next();
            if (rop != null) src1ID = getRegisterID(rop.getRegister());
            if (i.hasNext()) {
                rop = (RegisterOperand) i.next();
                if (rop != null) src2ID = getRegisterID(rop.getRegister());
                if (i.hasNext()) {
                    srcsID = new LinkedList();
                    do {
                        rop = (RegisterOperand) i.next();
                        if (rop != null) srcsID.add(new Integer(getRegisterID(rop.getRegister())));
                    } while (i.hasNext());
                }
            }
        }
        i = q.getAllOperands().iterator();
        while (i.hasNext()) {
            Operand op = (Operand) i.next();
            if (op instanceof RegisterOperand) continue;
            else if (op instanceof ConstOperand) {
                constantID = getConstantID(((ConstOperand) op).getWrapped());
            } else if (op instanceof TargetOperand) {
                targetID = getQuadID(((TargetOperand) op).getTarget().getQuad(0));
            } else if (op instanceof FieldOperand) {
                memberID = getMemberID(((FieldOperand) op).getField());
            } else if (op instanceof MethodOperand) {
                memberID = getMemberID(((MethodOperand) op).getMethod());
            } else if (op instanceof TypeOperand) {
                memberID = getMemberID(((TypeOperand) op).getType());
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
        if (srcsID != null) {
            BDD temp = bdd.zero();
            int j = 1;
            for (i = srcsID.iterator(); i.hasNext(); ++j) {
                int srcID = ((Integer) i.next()).intValue();
                if (ZERO_FIELDS || srcID != 0) {
                    BDD temp2 = srcNum.ithVar(j);
                    temp2.andWith(srcs.ithVar(srcID));
                    temp.orWith(temp2);
                }
            }
            if (!temp.isZero())
                currentQuad.andWith(temp);
            else
                temp.free();
        } else if (ZERO_FIELDS) {
            BDD temp2 = srcNum.ithVar(0);
            temp2.andWith(srcs.ithVar(0));
            currentQuad.andWith(temp2);
        }
        ++totalQuads;
    }
    
    public void dump() throws IOException {
        System.out.println("Var order: "+varOrderDesc);
        dumpMap(quadMap, "quad.map");
        dumpMap(opMap, "op.map");
        //dumpMap(regMap, "reg.map");
        dumpMap(memberMap, "member.map");
        dumpMap(constantMap, "constant.map");
        dumpBDDConfig("bdd.cfg");
        dumpFieldDomains("fielddomains.cfg");
        dumpRelations("relations.cfg");
        System.out.print("Saving BDD...");
        bdd.save("cfg.bdd", allQuads);
        bdd.save("m2q.bdd", methodToQuad);
        bdd.save("entries.bdd", methodEntries);
        bdd.save("nullconstant.bdd", nullConstant);
        bdd.save("nonnullconstants.bdd", nonNullConstants);
        System.out.println("done.");
        dumpTuples("cfg.tuples", allQuads);
        dumpTuples("m2q.tuples", methodToQuad);
        dumpTuples("entries.tuples", methodEntries);
        dumpTuples("nullconstant.tuples", nullConstant);
        dumpTuples("nonnullconstants.tuples", nonNullConstants);
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
        dos.writeBytes("srcNum "+(1L<<varargsBits)+"\n");
        dos.writeBytes("srcs "+(1L<<regBits)+"\n");
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
        dos.writeBytes("varargs "+(1L<<varargsBits)+"\n");
        dos.close();
    }
    
    void dumpRelations(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        if (GLOBAL_QUAD_NUMBERS) {
            dos.writeBytes("cfg ( id : quad , op : op , dest : reg , src1 : reg , src2 : reg , const : constant , fallthrough : quad , target : quad , member : member , srcNum : varargs , srcs : reg )\n");
        } else {
            dos.writeBytes("cfg ( method : method , id : quad , op : op , dest : reg , src1 : reg , src2 : reg , const : constant , fallthrough : quad , target : quad , member : member , srcNum : varargs , srcs : reg )\n");
        }
        dos.writeBytes("m2q ( method : method , id : quad )\n");
        dos.writeBytes("entries ( method : method , entry : quad )\n");
        dos.writeBytes("nullconstant ( constant : constant )\n");
        dos.writeBytes("nonnullconstant ( constant : constant )\n");
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
        int[] a = allQ.support().scanSetDomains();
        BDD allDomains = bdd.one();
        System.out.print(fileName+" domains {");
        for (int i = 0; i < a.length; ++i) {
            BDDDomain d = bdd.getDomain(i);
            System.out.print(" "+d.toString());
            allDomains.andWith(d.set());
        }
        System.out.println(" ) = "+allQ.nodeCount()+" nodes");
        int lines = 0;
        for (int i = 0; i < quadMap.size(); ++i) {
            BDD q = quad.ithVar(i).andWith(allQ.id());
            while (!q.isZero()) {
                BDD sat = q.satOne(allDomains, bdd.zero());
                BDD sup = q.support();
                int[] b = sup.scanSetDomains();
                sup.free();
                long[] v = sat.scanAllVar();
                sat.free();
                BDD t = bdd.one();
                for (int j = 0, k = 0; j < bdd.numberOfDomains(); ++j) {
                    BDDDomain d = bdd.getDomain(j);
                    if (k >= a.length || a[k] != j) {
                        Assert._assert(v[j] == 0, "v["+j+"] is "+v[j]);
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                        continue;
                    } else {
                        ++k;
                    }
                    if (v[j] == 0) {
                        BDD qs = q.support();
                        qs.orWith(d.set());
                        boolean contains = qs.isOne();
                        qs.free();
                        if (!contains) {
                            dos.writeBytes("* ");
                            t.andWith(d.domain());
                            continue;
                        }
                    }
                    dos.writeBytes(v[j]+" ");
                    t.andWith(d.ithVar(v[j]));
                }
                q.applyWith(t, BDDFactory.diff);
                dos.writeBytes("\n");
                ++lines;
            }
            q.free();
        }
        dos.close();
        System.out.println("Done printing "+lines+" lines.");
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
            q.free();
        }
    }
    
    void printQuad(BDD q) {
        long id = q.scanVar(quad);
        if (id == -1) return;
        System.out.println("Quad id "+id);
        System.out.println("        "+quadMap.get((int) id));
        System.out.println(q.toStringWithDomains());
    }
    
    void buildNullConstantBdds() {
        for (int i = 0; i < constantMap.size(); ++i) {
            Object c = constantMap.get(i);
            if (c == null) {
                nullConstant.orWith(constant.ithVar(i));
            }
            else if (!(c instanceof Integer) &&
                     !(c instanceof Float) &&
                     !(c instanceof Long) &&
                     !(c instanceof Double)) {
                nonNullConstants.orWith(constant.ithVar(i));                    
            }
        }
    }
}

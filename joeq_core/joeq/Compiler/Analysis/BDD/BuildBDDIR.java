// BuildBDDIR.java, created Mar 17, 2004 2:43:55 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.BDD;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import joeq.Class.jq_Method;
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
    
    String dumpDir = System.getProperty("bdddumpdir", "");
    boolean DUMP_TUPLES = !System.getProperty("dumptuples", "yes").equals("no");
    
    String varOrderDesc = "method_quadxtargetxfallthrough_member_constant_opc_srcs_dest_srcNum";
    
    int methodBits = 14, quadBits = 18, opBits = 8, regBits = 7, constantBits = 13, memberBits = 14, varargsBits = 4;

    BDDFactory bdd;
    BDDDomain method, quad, opc, dest, src1, src2, constant, fallthrough, target, member, srcNum, srcs;
    BDD methodToQuad;
    BDD methodEntries;
    BDD nullConstant;
    BDD nonNullConstants;
    BDD allQuads;
    BDD currentQuad;
    
    Object theDummyObject;
    
    int totalQuads;
    
    boolean ZERO_FIELDS = !System.getProperty("zerofields", "yes").equals("no");
    boolean GLOBAL_QUAD_NUMBERS = !System.getProperty("globalquadnumber", "yes").equals("no");
    boolean SSA = !System.getProperty("ssa", "no").equals("no");
    boolean USE_SRC12 = !System.getProperty("src12", "no").equals("no");
    
    public BuildBDDIR() {
        if (!GLOBAL_QUAD_NUMBERS) {
            quadBits = 13;
        }
        if (SSA) {
            regBits = 11;
            varargsBits = 6;
            int index = varOrderDesc.indexOf("xtargetxfallthrough");
            varOrderDesc = varOrderDesc.substring(0, index) + varOrderDesc.substring(index + "xtargetxfallthrough".length());
            
            varOrderDesc = "method_memberxquad_constant_opc_srcs_dest_srcNum";
        }
        if (USE_SRC12) {
            int index = varOrderDesc.indexOf("_srcs");
            varOrderDesc = varOrderDesc.substring(0, index) + "_src2_src1" + varOrderDesc.substring(index);
        }
        theDummyObject = new Object();
        methodMap = new IndexMap("method");
        methodMap.get(theDummyObject);
        loadOpMap();
        quadMap = new IndexMap("quad");
        quadMap.get(theDummyObject);
        //regMap = new IndexMap("reg");
        memberMap = new IndexMap("member");
        memberMap.get(theDummyObject);
        constantMap = new IndexMap("constant");
        constantMap.get(theDummyObject);
        bdd = BDDFactory.init(1000000, 50000);
        method = makeDomain("method", methodBits);
        quad = makeDomain("quad", quadBits);
        opc = makeDomain("opc", opBits);
        dest = makeDomain("dest", regBits);
        if (USE_SRC12) {
            src1 = makeDomain("src1", regBits);
            src2 = makeDomain("src2", regBits);
        }
        constant = makeDomain("constant", constantBits);
        if (!SSA) {
            fallthrough = makeDomain("fallthrough", quadBits);
            target = makeDomain("target", quadBits);
        }
        member = makeDomain("member", memberBits);
        srcNum = makeDomain("srcNum", varargsBits);
        srcs = makeDomain("srcs", regBits);
        allQuads = bdd.zero();
        methodToQuad = bdd.zero();
        methodEntries = bdd.zero();
        nullConstant = bdd.zero();
        nonNullConstants = bdd.zero();
        System.out.println("Using variable ordering "+varOrderDesc);
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
    
    void loadOpMap() {
        String fileName = "op.map";
        try {
            DataInputStream in = new DataInputStream(new FileInputStream(fileName));
            opMap = IndexMap.loadStringMap("op", in);
            in.close();
        } catch (IOException x) {
            System.out.println("Cannot load op map "+fileName);
            opMap = new IndexMap("op");
            opMap.get(new Object());
        }
    }
    
    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.ControlFlowGraphVisitor#visitCFG(joeq.Compiler.Quad.ControlFlowGraph)
     */
    public void visitCFG(ControlFlowGraph cfg) {
        if (SSA) {
            new EnterSSA().visitCFG(cfg);
        }
        QuadIterator i = new QuadIterator(cfg);
        int methodID = getMethodID(cfg.getMethod());
        
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
            if (!SSA) {
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
            }
            //printQuad(currentQuad);
            allQuads.orWith(currentQuad);
        }
        
        time = System.currentTimeMillis() - time;
        totalTime += time;
        System.out.println("Method: " + cfg.getMethod() + " time: " + time);
        int qSize = totalQuads;
        //int nodes = allQuads.nodeCount();
        //System.out.println("Quads: " +qSize+", nodes: "+nodes+", average:
        // "+(float)nodes/qSize);
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
    
    public int getMethodID(jq_Method m) {
        int x = methodMap.get(m);
        Assert._assert(x > 0);
        return x;
    }
    
    public int getRegisterID(Register r) {
        int x = r.getNumber() + 1;
        return x;
    }
    
    public int getConstantID(Object c) {
        int x = constantMap.get(c);
        Assert._assert(x > 0);
        return x;
    }
    
    public int getQuadID(Quad r) {
        int x = quadMap.get(r);
        Assert._assert(x > 0);
        return x;
    }
    
    public int getMemberID(Object r) {
        int x = memberMap.get(r);
        Assert._assert(x > 0);
        return x;
    }
    
    public int getOpID(Operator r) {
        int x = opMap.get(r.toString());
        Assert._assert(x > 0);
        return x;
    }
    
    void handleQuad(Quad q) {
        int quadID=0, opcID=0, destID=0, src1ID=0, src2ID=0, constantID=0, targetID=0, memberID=0;
        List srcsID = null;
        quadID = getQuadID(q);
        opcID = getOpID(q.getOperator());
        Iterator i = q.getDefinedRegisters().iterator();
        if (i.hasNext()) {
            destID = getRegisterID(((RegisterOperand)i.next()).getRegister());
            Assert._assert(!i.hasNext());
        }
        i = q.getUsedRegisters().iterator();
        if (USE_SRC12 && i.hasNext()) {
            RegisterOperand rop;
            rop = (RegisterOperand) i.next();
            if (rop != null) src1ID = getRegisterID(rop.getRegister());
            if (i.hasNext()) {
                rop = (RegisterOperand) i.next();
                if (rop != null) src2ID = getRegisterID(rop.getRegister());
            }
        }
        if (i.hasNext()) {
            srcsID = new LinkedList();
            do {
                RegisterOperand rop = (RegisterOperand) i.next();
                if (rop != null) srcsID.add(new Integer(getRegisterID(rop.getRegister())));
            } while (i.hasNext());
        }
        i = q.getAllOperands().iterator();
        while (i.hasNext()) {
            Operand op = (Operand) i.next();
            if (op instanceof RegisterOperand) continue;
            else if (op instanceof ConstOperand) {
                constantID = getConstantID(((ConstOperand) op).getWrapped());
            } else if (op instanceof TargetOperand) {
                if (!SSA)
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
        if (USE_SRC12) {
            if (ZERO_FIELDS || src1ID != 0) currentQuad.andWith(src1.ithVar(src1ID));
            if (ZERO_FIELDS || src2ID != 0) currentQuad.andWith(src2.ithVar(src2ID));
        }
        if (ZERO_FIELDS || constantID != 0) currentQuad.andWith(constant.ithVar(((long)constantID) & 0xFFFFFFFFL));
        if (!SSA) {
            if (ZERO_FIELDS || targetID != 0) currentQuad.andWith(target.ithVar(targetID));
        }
        if (ZERO_FIELDS || memberID != 0) currentQuad.andWith(member.ithVar(memberID));
        if (srcsID != null && !srcsID.isEmpty()) {
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
        dumpMap(quadMap, dumpDir+"quad.map");
        dumpMap(opMap, dumpDir+"op.map");
        //dumpMap(regMap, dumpDir+"reg.map");
        dumpMap(memberMap, dumpDir+"member.map");
        dumpMap(constantMap, dumpDir+"constant.map");
        
        String relationName;
        if (SSA) {
            relationName = "ssa";
        }
        else {
            relationName = "cfg";
        }
        
        dumpBDDConfig(dumpDir+"bdd."+relationName);
        dumpFieldDomains(dumpDir+"fielddomains."+relationName);
        dumpRelations(dumpDir+"relations."+relationName);            
        
        System.out.print("Saving BDDs...");
        bdd.save(dumpDir+relationName+".bdd", allQuads);
        bdd.save(dumpDir+"m2q.bdd", methodToQuad);
        bdd.save(dumpDir+"entries.bdd", methodEntries);
        bdd.save(dumpDir+"nullconstant.bdd", nullConstant);
        bdd.save(dumpDir+"nonnullconstants.bdd", nonNullConstants);
        System.out.println("done.");
        
        if (DUMP_TUPLES) {
            System.out.println("Saving tuples....");
            dumpTuples(bdd, dumpDir+relationName+".tuples", allQuads);
            dumpTuples(bdd, dumpDir+"m2q.tuples", methodToQuad);
            dumpTuples(bdd, dumpDir+"entries.tuples", methodEntries);
            dumpTuples(bdd, dumpDir+"nullconstant.tuples", nullConstant);
            dumpTuples(bdd, dumpDir+"nonnullconstants.tuples", nonNullConstants);
            System.out.println("done.");
        }
    }
    
    void dumpBDDConfig(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        for (int i = 0; i < bdd.numberOfDomains(); ++i) {
            BDDDomain d = bdd.getDomain(i);
            dos.writeBytes(d.getName()+" "+d.size()+"\n");
        }
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
    
    void dumpRelation(DataOutputStream dos, String name, BDD relation) throws IOException {
        int[] a = relation.support().scanSetDomains();
        dos.writeBytes(name+" ( ");
        for (int i = 0; i < a.length; ++i) {
            if (i > 0) dos.writeBytes(", ");
            BDDDomain d = bdd.getDomain(a[i]);
            dos.writeBytes(d.toString()+" : ");
            if (d == quad || d == fallthrough || d == target) dos.writeBytes("quad ");
            else if (d == method) dos.writeBytes("method ");
            else if (d == opc) dos.writeBytes("op ");
            else if (d == dest || d == srcs || d == src1 || d == src2) dos.writeBytes("reg ");
            else if (d == constant) dos.writeBytes("constant ");
            else if (d == member) dos.writeBytes("member ");
            else if (d == srcNum) dos.writeBytes("varargs ");
            else dos.writeBytes("??? ");
        }
        dos.writeBytes(")\n");
    }
    
    void dumpRelations(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        dumpRelation(dos, "m2q", methodToQuad);
        if (SSA) {
            dumpRelation(dos, "ssa", allQuads);
        } else {
            dumpRelation(dos, "cfg", allQuads);
        }
        dumpRelation(dos, "entries", methodEntries);
        dumpRelation(dos, "nullconstant", nullConstant);
        dumpRelation(dos, "nonnullconstants", nonNullConstants);
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
    
    public static void dumpTuples(BDDFactory bdd, String fileName, BDD relation) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        if (relation.isZero()) {
            dos.close();
            return;
        }
        Assert._assert(!relation.isOne());
        BDD rsup = relation.support();
        int[] a = rsup.scanSetDomains();
        rsup.free();
        BDD allDomains = bdd.one();
        System.out.print(fileName+" domains {");
        dos.writeBytes("#");
        for (int i = 0; i < a.length; ++i) {
            BDDDomain d = bdd.getDomain(a[i]);
            System.out.print(" "+d.toString());
            dos.writeBytes(" "+d.toString()+":"+d.varNum());
            allDomains.andWith(d.set());
        }
        dos.writeBytes("\n");
        System.out.println(" ) = "+relation.nodeCount()+" nodes");
        BDDDomain primaryDomain = bdd.getDomain(a[0]);
        int lines = 0;
        BDD foo = relation.exist(allDomains.exist(primaryDomain.set()));
        for (Iterator i = foo.iterator(primaryDomain.set()); i.hasNext(); ) {
            BDD q = (BDD) i.next();
            q.andWith(relation.id());
            while (!q.isZero()) {
                BDD sat = q.satOne(allDomains, bdd.zero());
                BDD sup = q.support();
                int[] b = sup.scanSetDomains();
                sup.free();
                long[] v = sat.scanAllVar();
                sat.free();
                BDD t = bdd.one();
                for (int j = 0, k = 0, l = 0; j < bdd.numberOfDomains(); ++j) {
                    BDDDomain d = bdd.getDomain(j);
                    if (k >= a.length || a[k] != j) {
                        Assert._assert(v[j] == 0, "v["+j+"] is "+v[j]);
                        //dos.writeBytes("* ");
                        t.andWith(d.domain());
                        continue;
                    } else {
                        ++k;
                    }
                    if (l >= b.length || b[l] != j) {
                        Assert._assert(v[j] == 0, "v["+j+"] is "+v[j]);
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                        continue;
                    } else {
                        ++l;
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
                     !(c instanceof Double) &&
                     c != theDummyObject) {
                nonNullConstants.orWith(constant.ithVar(i));                    
            }
        }
    }
}

// Comparisons.java, created Sun Feb  8 16:38:30 PST 2004 by gback
// Copyright (C) 2003 Godmar Back <gback@stanford.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
import Compil3r.Analysis.IPA.*;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.InputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Member;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operator;
import Compil3r.Quad.Operand;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operand.ConditionOperand;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.QuadVisitor;
import Compil3r.Quad.BasicBlockVisitor;
import Compil3r.Quad.ControlFlowGraph;
import Util.Assert;
import Util.Strings;
import Util.IO.SourceLister;
import Util.Collections.Pair;
import Util.Collections.IndexMap;
import Util.Graphs.PathNumbering;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;

/**
 * Do analysis of types stored in collections.
 * 
 * @author Godmar Back
 * @version $Id$
 */
public class Comparisons {

    PAResults res;
    PAProxy r;
    MethodSummary ms;
    int totalcmps;
    int opmissed, notregcmps;
    HashSet missednodes;
    IndexMap cmpLoc;    // for comparison locations, we abuse F here.

    public Comparisons(PAProxy r, PAResults res) {
        this.r = r;
        this.res = res;
    }

    public TypedBDD getComparisons(final boolean trace) {
        final PAProxy r = this.r;
        final TypedBDD cmp = (TypedBDD)r.bdd.zero();
        opmissed = 0;
        totalcmps = 0;
        notregcmps = 0;
        missednodes = new HashSet();
        cmpLoc = new IndexMap("CIndex");

        BasicBlockVisitor bbv = new BasicBlockVisitor() {
            public void visitBasicBlock(final BasicBlock bb) {
                QuadVisitor qv = new QuadVisitor.EmptyVisitor() {
                    public void visitIntIfCmp(Quad quad) {
                        if (quad.getOperator() == Operator.IntIfCmp.IFCMP_A.INSTANCE) {
                            if (trace) System.out.println("visiting IntIfCmp: " + quad);
                            totalcmps++;
                            Operand op1 = Operator.IntIfCmp.getSrc1(quad);
                            Operand op2 = Operator.IntIfCmp.getSrc2(quad);
                            ConditionOperand c = Operator.IntIfCmp.getCond(quad);
                            if (!(op1 instanceof RegisterOperand) || !(op2 instanceof RegisterOperand)) {
                                notregcmps++;
                                return;
                            }
                            TypedBDD v1cmp = (TypedBDD)r.bdd.zero();
                            Register reg1 = ((RegisterOperand)op1).getRegister();
                            Collection c1 = ms.getRegisterAtLocation(bb, quad, reg1);
                            Register reg2 = ((RegisterOperand)op2).getRegister();
                            Collection c2 = ms.getRegisterAtLocation(bb, quad, reg2);

                            // first make sure we have all nodes in the map, or else we can't
                            // say anything about this comparison
                            int _oldmsize = missednodes.size();
                            for (Iterator i = c1.iterator(); i.hasNext(); ) {
                                Node n = (Node) i.next();
                                if (!r.Vmap.contains(n) && missednodes.add(n) && trace) {
                                    System.out.println("not found in map " + n.toString_long() + " in " + ms.getMethod());
                                }
                            }
                            for (Iterator i = c2.iterator(); i.hasNext(); ) {
                                Node n = (Node) i.next();
                                if (!r.Vmap.contains(n) && missednodes.add(n) && trace) {
                                    System.out.println("not found in map " + n.toString_long() + " in " + ms.getMethod());
                                } 
                            }
                            int unknowns = missednodes.size() - _oldmsize;
                            if (unknowns > 0) {
                                opmissed++;
                                return;
                            }

                            for (Iterator i = c1.iterator(); i.hasNext(); ) {
                                Node n = (Node) i.next();
                                v1cmp.orWith(r.V1.ithVar(r.Vmap.get(n)));
                            }

                            TypedBDD v2cmp = (TypedBDD)r.bdd.zero();
                            for (Iterator i = c2.iterator(); i.hasNext(); ) {
                                Node n = (Node) i.next();
                                v2cmp.orWith(r.V2.ithVar(r.Vmap.get(n)));
                            }
                            if (trace) 
                                System.out.println("Adding " + c1 + " vs " + c2);
                            QuadProgramLocation loc = new QuadProgramLocation(ms.getMethod(), quad);
                            // abuse F
                            BDD loci = r.F.ithVar(cmpLoc.get(loc));
                            cmp.orWith(v1cmp.andWith(v2cmp).andWith(loci));
                        }
                    } 
                };
                bb.visitQuads(qv);
            }
        };

        for (Iterator i = r.Mmap.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() != null) {
                ControlFlowGraph cfg = CodeCache.getCode(m);
                ms = MethodSummary.getSummary(m);
                cfg.visitBasicBlocks(bbv);
            }
        }
        System.out.println("total number of IFCMP_A comparisons: " + totalcmps);
        System.out.println("number of comparisons that can't be analyzed: " + opmissed);
        System.out.println("number of comparisons that don't compare registers: " + notregcmps);
        System.out.println("number of node operands missed: " + missednodes.size());
        BDD v2 = cmp.exist(r.V1.set()).exist(r.F.set());    // V2
        BDDPairing v2tov1 = r.bdd.makePair(r.V2, r.V1);
        v2.replaceWith(v2tov1);         // V2->V1
//System.out.println(res.toString((TypedBDD)v2, 10));
        BDD vp2 = v2.and(r.vP);         // V1xV1cxH1xH1c
        v2.free();
        vp2.replaceWith(r.H1toH2);      // V1xV1cxH2xH2c
        vp2.replaceWith(r.V1toV2);      // V2xV2cxH2xH2c
        v2 = cmp.and(r.vP);             // FxV1xV2 x V1xV1cxH1xH1c -> FxV1xV1cxH1xH1cxV2
        v2.andWith(vp2);                // FxV1xV1cxH1xH1cxV2 x V2xV2cxH2xH2c -> FxV1xV1cxH1xH1cxV2xV2cxH2xH2c
        BDD sameVc = r.V1c.buildEquals(r.V2c);
        BDD sameHc = r.H1c.buildEquals(r.H2c);
        v2.andWith(sameVc);
        v2.andWith(sameHc);
        v2 = v2.exist(r.V1c.set()).exist(r.V2c.set()).exist(r.H1c.set()).exist(r.H2c.set());
        TypedBDD sameH = (TypedBDD)r.H1.buildEquals(r.H2);
        res.storeBDD("sameh", sameH);

        TypedBDD eq = (TypedBDD)v2.and(sameH); // FxV1xV2xH1xH2
        res.storeBDD("eq", eq);

        BDD v1v2h1h2 = r.V1.set().andWith(r.V2.set()).andWith(r.H1.set()).andWith(r.H2.set());
        TypedBDD allcmp = (TypedBDD)cmp.exist(r.V1.set().andWith(r.V2.set()));    // F
        TypedBDD haveeq = (TypedBDD)eq.exist(v1v2h1h2); // FxV1xV2xH1xH2 -> F
        res.storeBDD("haveeq", haveeq);

        TypedBDD alwaysfalse = (TypedBDD)allcmp.apply(haveeq, BDDFactory.diff);
        long []af = r.F.getVarIndices(alwaysfalse);
        System.out.println("Number of always false comparisons: " + af.length);
        SourceLister l = new SourceLister();
        for (int i = 0; i < af.length; i++) {
            ProgramLocation pl = (ProgramLocation)cmpLoc.get((int)af[i]);
            System.out.println("F(" + i + ") " + pl.toStringLong() + "\n" + l.list(pl));
        }

/*
        TypedBDD ne = (TypedBDD)v2.apply(eq, BDDFactory.diff);
        res.storeBDD("ne", ne);

        TypedBDD havene = (TypedBDD)ne.exist(v1v2h1h2); // FxV1xV2xH1xH2
        res.storeBDD("havene", havene);
*/
        cmp.free();
        return (TypedBDD)v2;
    }
}

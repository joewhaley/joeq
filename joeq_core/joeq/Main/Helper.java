// Helper.java, created Thu Jan 16 10:53:32 2003 by mcmartin
// Copyright (C) 2001-3 mcmartin
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Main;

import java.util.Iterator;
import java.util.LinkedList;

import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_MethodVisitor;
import joeq.Class.jq_Type;
import joeq.Class.jq_TypeVisitor;
import joeq.Compiler.Quad.BasicBlock;
import joeq.Compiler.Quad.BasicBlockVisitor;
import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.ControlFlowGraphVisitor;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.QuadVisitor;

/**
 * @author  Michael Martin <mcmartin@stanford.edu>
 * @version $Id$
 */
public class Helper {
    static {
        HostedVM.initialize();
    }

    public static jq_Type load(String classname) {
        jq_Type c = jq_Type.parseType(classname);
        c.load();
        c.prepare();
        return c;
    }

    public static jq_Type[] loadPackage(String packagename) {
        return loadPackages(packagename, false);
    }

    public static jq_Type[] loadPackages(String packagename) {
        return loadPackages(packagename, true);
    }

    public static jq_Type[] loadPackages(
        String packagename,
        boolean recursive) {
        String canonicalPackageName = packagename.replace('.', '/');
        if (!canonicalPackageName.equals("") && 
            !canonicalPackageName.endsWith("/"))
            canonicalPackageName += '/';
        Iterator i =
            joeq.Class.PrimordialClassLoader.loader.listPackage(
                canonicalPackageName,
                recursive);
        if (!i.hasNext()) {
            System.err.println(
                "Package " + canonicalPackageName + " not found.");
        }
        LinkedList ll = new LinkedList();

        while (i.hasNext()) {
            String c = (String) i.next();
            c = c.substring(0, c.length() - 6);
            ll.add(Helper.load(c));
        }

        return (jq_Class[]) ll.toArray(new jq_Class[0]);
    }

    public static void runPass(jq_Class c, jq_TypeVisitor tv) {
        c.accept(tv);
    }

    public static void runPass(jq_Class c, jq_MethodVisitor mv) {
        runPass(c, new jq_MethodVisitor.DeclaredMethodVisitor(mv));
    }

    public static void runPass(jq_Class c, ControlFlowGraphVisitor cfgv) {
        runPass(c, new ControlFlowGraphVisitor.CodeCacheVisitor(cfgv));
    }

    public static void runPass(jq_Class c, BasicBlockVisitor bbv) {
        runPass(c, new BasicBlockVisitor.AllBasicBlockVisitor(bbv));
    }

    public static void runPass(jq_Class c, QuadVisitor qv) {
        runPass(c, new QuadVisitor.AllQuadVisitor(qv));
    }

    public static void runPass(jq_Method m, jq_MethodVisitor mv) {
        m.accept(mv);
    }

    public static void runPass(jq_Method m, ControlFlowGraphVisitor cfgv) {
        runPass(m, new ControlFlowGraphVisitor.CodeCacheVisitor(cfgv));
    }

    public static void runPass(jq_Method m, BasicBlockVisitor bbv) {
        runPass(m, new BasicBlockVisitor.AllBasicBlockVisitor(bbv));
    }

    public static void runPass(jq_Method m, QuadVisitor qv) {
        runPass(m, new QuadVisitor.AllQuadVisitor(qv));
    }

    public static void runPass(
        ControlFlowGraph c,
        ControlFlowGraphVisitor cfgv) {
        cfgv.visitCFG(c);
    }

    public static void runPass(ControlFlowGraph c, BasicBlockVisitor bbv) {
        runPass(c, new BasicBlockVisitor.AllBasicBlockVisitor(bbv));
    }

    public static void runPass(ControlFlowGraph c, QuadVisitor qv) {
        runPass(c, new QuadVisitor.AllQuadVisitor(qv));
    }

    public static void runPass(BasicBlock b, BasicBlockVisitor bbv) {
        bbv.visitBasicBlock(b);
    }

    public static void runPass(BasicBlock b, QuadVisitor qv) {
        runPass(b, new QuadVisitor.AllQuadVisitor(qv));
    }

    public static void runPass(Quad q, QuadVisitor qv) {
        q.accept(qv);
    }
}

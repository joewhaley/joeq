package Main;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;
import Clazz.jq_Type;
import Clazz.jq_TypeVisitor;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.BasicBlockVisitor;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadVisitor;

public class Helper {
    static {
	HostedVM.initialize();
    }

    public static jq_Class load(String classname) {
	jq_Class c = (jq_Class) jq_Type.parseType(classname);
	c.load(); c.prepare();
	return c;
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

    public static void runPass(ControlFlowGraph c, ControlFlowGraphVisitor cfgv) {
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

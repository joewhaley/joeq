/*
 * ControlFlowGraphVisitor.java
 *
 * Created on February 11, 2002, 12:14 AM
 */

package Compil3r.Quad;
import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;

/**
 *
 * @author  John Whaley
 * @version 
 */
public interface ControlFlowGraphVisitor {
    public void visitCFG(ControlFlowGraph cfg);
    
    public class CodeCacheVisitor extends jq_MethodVisitor.EmptyVisitor {
        private final ControlFlowGraphVisitor bbv;
        boolean trace;
        public CodeCacheVisitor(ControlFlowGraphVisitor bbv) { this.bbv = bbv; }
        public CodeCacheVisitor(ControlFlowGraphVisitor bbv, boolean trace) { this.bbv = bbv; this.trace = trace; }
        public void visitMethod(jq_Method m) {
            if (m.getBytecode() == null) return;
            if (trace) System.out.println(m.toString());
            ControlFlowGraph cfg = Compil3r.Quad.CodeCache.getCode(m);
            bbv.visitCFG(cfg);
        }
    }

}

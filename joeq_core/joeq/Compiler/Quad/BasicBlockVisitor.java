/*
 * BasicBlockVisitor.java
 *
 * Created on January 9, 2002, 8:47 AM
 *
 */

package Compil3r.Quad;
import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;

/**
 * Interface for the basic block visitor design pattern.
 * Make your visitor object implement this class in order to visit 
 * @author  John Whaley
 * @see BasicBlock
 * @version $Id$
 */
public interface BasicBlockVisitor {
    
    /** Visit a basic block.
     * @param bb  basic block to visit */
    public void visitBasicBlock(BasicBlock bb);

    /**
     * Empty basic block visitor for easy subclassing.
     */
    public static class EmptyVisitor implements BasicBlockVisitor {
        /** Visit a basic block.
         * @param bb  basic block to visit */
        public void visitBasicBlock(BasicBlock bb) {}
    }
    
    /**
     * Method visitor that visits all basic blocks in the method with a given
     * basic block visitor.
     * @see  jq_Method
     * @see  jq_MethodVisitor
     */
    public static class AllBasicBlockVisitor extends jq_MethodVisitor.EmptyVisitor {
        private final BasicBlockVisitor bbv;
        boolean trace;
        /** Construct a new AllBasicBlockVisitor.
         * @param bbv  basic block visitor to visit each basic block with. */
        public AllBasicBlockVisitor(BasicBlockVisitor bbv) { this.bbv = bbv; }
        /** Construct a new AllBasicBlockVisitor and set the trace flag to be the specified value.
         * @param bbv  basic block visitor to visit each basic block with.
         * @param trace  value of the trace flag */
        public AllBasicBlockVisitor(BasicBlockVisitor bbv, boolean trace) { this.bbv = bbv; this.trace = trace; }
        /** Convert the given method to quad format and visit each of its basic blocks
         * with the basic block visitor specified in the constructor.
         * Skips native and abstract methods.
         * @see  Compil3r.Quad.CodeCache
         * @param m  method to visit
         */
        public void visitMethod(jq_Method m) {
            if (m.isNative() || m.isAbstract()) return;
            if (trace) System.out.println(m.toString());
            ControlFlowGraph cfg = Compil3r.Quad.CodeCache.getCode(m);
            cfg.visitBasicBlocks(bbv);
        }
    }
    
}

/*
 * BasicBlockVisitor.java
 *
 * Created on January 9, 2002, 8:47 AM
 *
 */

package Compil3r.Quad;

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
    public static class AllBasicBlockVisitor implements ControlFlowGraphVisitor {
        private final BasicBlockVisitor bbv;
        boolean trace;
        /** Construct a new AllBasicBlockVisitor.
         * @param bbv  basic block visitor to visit each basic block with. */
        public AllBasicBlockVisitor(BasicBlockVisitor bbv) { this.bbv = bbv; }
        /** Construct a new AllBasicBlockVisitor and set the trace flag to be the specified value.
         * @param bbv  basic block visitor to visit each basic block with.
         * @param trace  value of the trace flag */
        public AllBasicBlockVisitor(BasicBlockVisitor bbv, boolean trace) { this.bbv = bbv; this.trace = trace; }
        /** Visit each of the basic blocks in the given control flow graph.
         * @param cfg  control flow graph to visit
         */
        public void visitCFG(ControlFlowGraph cfg) {
            cfg.visitBasicBlocks(bbv);
        }
    }
    
}

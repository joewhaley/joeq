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
     * Method visitor that visits all basic blocks in the method with a given
     * basic block visitor.
     * @see  jq_Method
     * @see  jq_MethodVisitor
     */
    public class AllBasicBlockVisitor extends jq_MethodVisitor.EmptyVisitor {
        private final BasicBlockVisitor bbv;
        /** Construct a new AllBasicBlockVisitor.
         * @param bbv  basic block visitor to visit each basic block with. */
        public AllBasicBlockVisitor(BasicBlockVisitor bbv) { this.bbv = bbv; }
        /** Convert the given method to quad format and visit each of its basic blocks
         * with the basic block visitor specified in the constructor.
         * Skips native and abstract methods.
         * @see  BytecodeToQuad
         * @param m  method to visit
         */
        public void visitMethod(jq_Method m) {
            if (m.isNative() || m.isAbstract()) return;
            BytecodeToQuad b2q = new BytecodeToQuad(m);
            ControlFlowGraph cfg = b2q.convert();
            cfg.visitBasicBlocks(bbv);
        }
    }
    
}

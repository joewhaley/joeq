/*
 * BasicBlockVisitor.java
 *
 * Created on January 9, 2002, 8:47 AM
 *
 * @author  Administrator
 * @version 
 */

package Compil3r.Quad;

import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;

public interface BasicBlockVisitor {
    
    /** Visit a basic block. */
    public void visitBasicBlock(BasicBlock bb);

    class AllBasicBlockVisitor extends jq_MethodVisitor.EmptyVisitor {
        final BasicBlockVisitor bbv;
        public AllBasicBlockVisitor(BasicBlockVisitor bbv) { this.bbv = bbv; }
        public void visitMethod(jq_Method m) {
            if (m.isNative() || m.isAbstract()) return;
            BytecodeToQuad b2q = new BytecodeToQuad(m);
            ControlFlowGraph cfg = b2q.convert();
            cfg.visitBasicBlocks(bbv);
        }
    }
    
}

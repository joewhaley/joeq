// ControlFlowGraphVisitor.java, created Mon Feb 11  0:24:01 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;

/**
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public interface ControlFlowGraphVisitor {
    void visitCFG(ControlFlowGraph cfg);
    
    class CodeCacheVisitor extends jq_MethodVisitor.EmptyVisitor {
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

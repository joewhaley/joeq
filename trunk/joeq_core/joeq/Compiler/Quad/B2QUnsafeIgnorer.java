// B2QUnsafeIgnorer.java, created Mon Dec 23 23:00:34 2002 by mcmartin
// Copyright (C) 2001-3 mcmartin
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Quad;

import joeq.Clazz.jq_Method;

/*
 * @author  Michael Martin <mcmartin@stanford.edu>
 * @version $Id$
 */
class B2QUnsafeIgnorer implements BytecodeToQuad.UnsafeHelper {
    public boolean isUnsafe(jq_Method m) {
        return false;
    }
    public boolean endsBB(jq_Method m) {
        return false;
    }
    public boolean handleMethod(BytecodeToQuad b2q, ControlFlowGraph quad_cfg, BytecodeToQuad.AbstractState current_state, jq_Method m, Operator.Invoke oper) {
        return false;
    }
}

// NullDelegates.java, created Mon Dec 23 20:00:01 2002 by mcmartin
// Copyright (C) 2001-3 Michael Martin <mcmartin@stanford.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Quad;

import joeq.Interpreter.QuadInterpreter;

/**
 * @author Michael Martin <mcmartin@stanford.edu>
 * @version $Id$
 */
class NullDelegates {
    static class Op implements joeq.Compil3r.Quad.Operator.Delegate {
        public void interpretGetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) { }
        public void interpretSetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) { }
        public void interpretMonitorEnter(Operator.Monitor op, Quad q, QuadInterpreter s) { }
        public void interpretMonitorExit(Operator.Monitor op, Quad q, QuadInterpreter s) { }
    }
}

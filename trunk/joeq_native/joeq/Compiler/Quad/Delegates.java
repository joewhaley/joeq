// Delegates.java, created Mon Dec 23 20:00:01 2002 by mcmartin
// Copyright (C) 2001-3 mcmartin
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Quad;

import joeq.Compil3r.Quad.Operand.RegisterOperand;
import joeq.Interpreter.QuadInterpreter;
import joeq.Main.jq;
import joeq.Run_Time.Unsafe;

/*
 * @author  Michael Martin <mcmartin@stanford.edu>
 * @version $Id$
 */
class Delegates {
    static class Op implements Operator.Delegate {
        public void interpretGetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) {
            if (jq.RunningNative)
                s.putReg_A(((RegisterOperand)Operator.Special.getOp1(q)).getRegister(), Unsafe.getThreadBlock());
        }
        public void interpretSetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) {
            Scheduler.jq_Thread o = (Scheduler.jq_Thread)Operator.getObjectOpValue(Operator.Special.getOp2(q), s);
            if (jq.RunningNative)
                Unsafe.setThreadBlock(o);
        }
        public void interpretMonitorEnter(Operator.Monitor op, Quad q, QuadInterpreter s) {
            Object o = Operator.getObjectOpValue(Operator.Monitor.getSrc(q), s);
            if (jq.RunningNative)
                Run_Time.Monitor.monitorenter(o);
        }
        public void interpretMonitorExit(Operator.Monitor op, Quad q, QuadInterpreter s) {
            Object o = Operator.getObjectOpValue(Operator.Monitor.getSrc(q), s);
            if (jq.RunningNative)
                Run_Time.Monitor.monitorexit(o);
        }
    }
}

package Compil3r.Quad;

import Compil3r.Quad.Operand.RegisterOperand;
import Interpreter.QuadInterpreter;
import Main.jq;
import Run_Time.Unsafe;

class Delegates {
    static class Op implements Operator.Delegate {
	public void interpretGetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) {
	    if (jq.RunningNative)
		s.putReg_A(((RegisterOperand)op.getOp1(q)).getRegister(), Unsafe.getThreadBlock());
	}
	public void interpretSetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) {
	    Scheduler.jq_Thread o = (Scheduler.jq_Thread)op.getObjectOpValue(op.getOp2(q), s);
	    if (jq.RunningNative)
		Unsafe.setThreadBlock(o);
	}
	public void interpretMonitorEnter(Operator.Monitor op, Quad q, QuadInterpreter s) {
	    Object o = op.getObjectOpValue(op.getSrc(q), s);
	    if (jq.RunningNative)
		Run_Time.Monitor.monitorenter(o);
	}
	public void interpretMonitorExit(Operator.Monitor op, Quad q, QuadInterpreter s) {
	    Object o = op.getObjectOpValue(op.getSrc(q), s);
	    if (jq.RunningNative)
		Run_Time.Monitor.monitorexit(o);
	}
    }
}

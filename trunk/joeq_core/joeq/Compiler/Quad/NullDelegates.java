package Compil3r.Quad;
import Interpreter.QuadInterpreter;

class NullDelegates {
    static class Op implements Compil3r.Quad.Operator.Delegate {
	public void interpretGetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) { }
	public void interpretSetThreadBlock(Operator.Special op, Quad q, QuadInterpreter s) { }
	public void interpretMonitorEnter(Operator.Monitor op, Quad q, QuadInterpreter s) { }
	public void interpretMonitorExit(Operator.Monitor op, Quad q, QuadInterpreter s) { }
    }
}

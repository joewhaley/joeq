package Compil3r.Quad;

import Clazz.jq_Method;

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

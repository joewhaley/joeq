package Compil3r.Quad;

import Clazz.jq_Method;
import Clazz.jq_Primitive;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operator.Unary;
import Compil3r.Quad.Operator.Special;
import Memory.StackAddress;
import Memory.CodeAddress;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;

class B2QUnsafeHandler implements BytecodeToQuad.UnsafeHelper {
    public boolean isUnsafe(jq_Method m) {
	return m.getDeclaringClass() == Unsafe._class;
    }
    public boolean endsBB(jq_Method m) {
	return m == Unsafe._longJump;
    }
    public boolean handleMethod(BytecodeToQuad b2q, ControlFlowGraph quad_cfg, BytecodeToQuad.AbstractState current_state, jq_Method m, Operator.Invoke oper) {
        Quad q;
        if (m == Unsafe._floatToIntBits) {
            Operand op = current_state.pop_F();
            RegisterOperand res = b2q.getStackRegister(jq_Primitive.INT);
            q = Unary.create(quad_cfg.getNewQuadID(), Unary.FLOAT_2INTBITS.INSTANCE, res, op);
            current_state.push_I(res);
        } else if (m == Unsafe._intBitsToFloat) {
            Operand op = current_state.pop_I();
            RegisterOperand res = b2q.getStackRegister(jq_Primitive.FLOAT);
            q = Unary.create(quad_cfg.getNewQuadID(), Unary.INTBITS_2FLOAT.INSTANCE, res, op);
            current_state.push_F(res);
        } else if (m == Unsafe._doubleToLongBits) {
            Operand op = current_state.pop_D();
            RegisterOperand res = b2q.getStackRegister(jq_Primitive.LONG);
            q = Unary.create(quad_cfg.getNewQuadID(), Unary.DOUBLE_2LONGBITS.INSTANCE, res, op);
            current_state.push_L(res);
        } else if (m == Unsafe._longBitsToDouble) {
            Operand op = current_state.pop_L();
            RegisterOperand res = b2q.getStackRegister(jq_Primitive.DOUBLE);
            q = Unary.create(quad_cfg.getNewQuadID(), Unary.LONGBITS_2DOUBLE.INSTANCE, res, op);
            current_state.push_D(res);
        } else if (m == Unsafe._getThreadBlock) {
            RegisterOperand res = b2q.getStackRegister(jq_Thread._class);
            q = Special.create(quad_cfg.getNewQuadID(), Special.GET_THREAD_BLOCK.INSTANCE, res);
            current_state.push_A(res);
        } else if (m == Unsafe._setThreadBlock) {
            Operand loc = current_state.pop_A();
            q = Special.create(quad_cfg.getNewQuadID(), Special.SET_THREAD_BLOCK.INSTANCE, loc);
        } else if (m == Unsafe._longJump) {
            Operand eax = current_state.pop_I();
            Operand sp = current_state.pop(StackAddress._class);
            Operand fp = current_state.pop(StackAddress._class);
            Operand ip = current_state.pop(CodeAddress._class);
            q = Special.create(quad_cfg.getNewQuadID(), Special.LONG_JUMP.INSTANCE, ip, fp, sp, eax);
	} else {
	    return false;
	}
        b2q.appendQuad(q);
	return true;
    }
}

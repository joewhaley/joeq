/*
 * RegisterFactory.java
 *
 * Created on May 10, 2001, 8:34 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import Clazz.*;
import Compil3r.Quad.Operand.RegisterOperand;
import java.util.ArrayList;
import jq;

public class RegisterFactory {

    private ArrayList/*<Register>*/ local_I;
    private ArrayList/*<Register>*/ local_F;
    private ArrayList/*<Register>*/ local_L;
    private ArrayList/*<Register>*/ local_D;
    private ArrayList/*<Register>*/ local_A;
    private ArrayList/*<Register>*/ stack_I;
    private ArrayList/*<Register>*/ stack_F;
    private ArrayList/*<Register>*/ stack_L;
    private ArrayList/*<Register>*/ stack_D;
    private ArrayList/*<Register>*/ stack_A;
    
    /** Creates new RegisterFactory */
    public RegisterFactory(jq_Method m) {
        int nlocals = m.getMaxLocals();
        local_I = new ArrayList(nlocals);
        local_F = new ArrayList(nlocals);
        local_L = new ArrayList(nlocals);
        local_D = new ArrayList(nlocals);
        local_A = new ArrayList(nlocals);
        for (int i=0; i<nlocals; ++i) {
            local_I.add(i, new Register(i));
            local_F.add(i, new Register(i));
            local_L.add(i, new Register(i));
            local_D.add(i, new Register(i));
            local_A.add(i, new Register(i));
        }
        int nstack = m.getMaxStack();
        stack_I = new ArrayList(nstack);
        stack_F = new ArrayList(nstack);
        stack_L = new ArrayList(nstack);
        stack_D = new ArrayList(nstack);
        stack_A = new ArrayList(nstack);
        for (int i=0; i<nstack; ++i) {
            stack_I.add(i, new Register(i));
            stack_F.add(i, new Register(i));
            stack_L.add(i, new Register(i));
            stack_D.add(i, new Register(i));
            stack_A.add(i, new Register(i));
        }
    }

    Register getStack(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)stack_A.get(i);
        if (t.isIntLike()) return (Register)stack_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)stack_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)stack_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)stack_D.get(i);
        jq.UNREACHABLE();
        return null;
    }
    
    Register getLocal(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)local_A.get(i);
        if (t.isIntLike()) return (Register)local_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)local_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)local_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)local_D.get(i);
        jq.UNREACHABLE();
        return null;
    }
    
    boolean isLocal(Operand op, int index, jq_Type type) {
	if (index >= getLocalSize()) return false;
        if (op instanceof RegisterOperand) {
            Register r = ((RegisterOperand)op).getRegister();
            if (getLocal(index, type) == r) return true;
        }
        return false;
    }

    private int getLocalSize() { return local_I.size(); }

    RegisterOperand makeGuardReg() {
        return new RegisterOperand(new Register(-1), null);
    }
    
    public static class Register {
        private int id;
        Register(int id) { this.id = id; }
        public int getNumber() { return id; }
        public String toString() { return "R"+id; }
    }

}

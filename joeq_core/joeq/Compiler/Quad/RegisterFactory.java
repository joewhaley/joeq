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
            local_I.add(i, new Register(i, false));
            local_F.add(i, new Register(i, false));
            local_L.add(i, new Register(i, false));
            local_D.add(i, new Register(i, false));
            local_A.add(i, new Register(i, false));
        }
        int nstack = m.getMaxStack();
        stack_I = new ArrayList(nstack);
        stack_F = new ArrayList(nstack);
        stack_L = new ArrayList(nstack);
        stack_D = new ArrayList(nstack);
        stack_A = new ArrayList(nstack);
        for (int i=0; i<nstack; ++i) {
            stack_I.add(i, new Register(i, true));
            stack_F.add(i, new Register(i, true));
            stack_L.add(i, new Register(i, true));
            stack_D.add(i, new Register(i, true));
            stack_A.add(i, new Register(i, true));
        }
    }

    public Register getStack(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)stack_A.get(i);
        if (t.isIntLike()) return (Register)stack_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)stack_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)stack_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)stack_D.get(i);
        jq.UNREACHABLE();
        return null;
    }
    
    public Register getNewStack(int i, jq_Type t) {
        if (t.isReferenceType()) {
	    while (i >= stack_A.size())
		stack_A.add(i, new Register(i, true));
	    return (Register)stack_A.get(i);
	}
        if (t.isIntLike()) {
	    while (i >= stack_I.size())
		stack_I.add(i, new Register(i, true));
	    return (Register)stack_I.get(i);
	}
        if (t == jq_Primitive.FLOAT) {
	    while (i >= stack_F.size())
		stack_F.add(i, new Register(i, true));
	    return (Register)stack_F.get(i);
	}
        if (t == jq_Primitive.LONG) {
	    while (i >= stack_L.size())
		stack_L.add(i, new Register(i, true));
	    return (Register)stack_L.get(i);
	}
        if (t == jq_Primitive.DOUBLE) {
	    while (i >= stack_D.size())
		stack_D.add(i, new Register(i, true));
	    return (Register)stack_D.get(i);
	}
        jq.UNREACHABLE();
        return null;
    }
    
    public Register getLocal(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)local_A.get(i);
        if (t.isIntLike()) return (Register)local_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)local_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)local_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)local_D.get(i);
        jq.UNREACHABLE();
        return null;
    }
    
    public boolean isLocal(Operand op, int index, jq_Type type) {
	if (index >= getLocalSize(type)) return false;
        if (op instanceof RegisterOperand) {
            Register r = ((RegisterOperand)op).getRegister();
            if (getLocal(index, type) == r) return true;
        }
        return false;
    }
    
    public Register makeTempReg(jq_Type t) {
        if (t.isReferenceType()) {
            int i = stack_A.size();
            stack_A.add(i, new Register(i, true));
	    return (Register)stack_A.get(i);
	}
        if (t.isIntLike()) {
            int i = stack_I.size();
            stack_I.add(i, new Register(i, true));
	    return (Register)stack_I.get(i);
	}
        if (t == jq_Primitive.FLOAT) {
            int i = stack_F.size();
            stack_F.add(i, new Register(i, true));
	    return (Register)stack_F.get(i);
	}
        if (t == jq_Primitive.LONG) {
            int i = stack_L.size();
            stack_L.add(i, new Register(i, true));
	    return (Register)stack_L.get(i);
	}
        if (t == jq_Primitive.DOUBLE) {
            int i = stack_D.size();
            stack_D.add(i, new Register(i, true));
	    return (Register)stack_D.get(i);
	}
        jq.UNREACHABLE();
        return null;
    }

    public int getLocalSize(jq_Type t) {
        if (t.isReferenceType()) return local_A.size();
        if (t.isIntLike()) return local_I.size();
        if (t == jq_Primitive.FLOAT) return local_F.size();
        if (t == jq_Primitive.LONG) return local_L.size();
        if (t == jq_Primitive.DOUBLE) return local_D.size();
        jq.UNREACHABLE();
        return 0;
    }
    public int getStackSize(jq_Type t) {
        if (t.isReferenceType()) return stack_A.size();
        if (t.isIntLike()) return stack_I.size();
        if (t == jq_Primitive.FLOAT) return stack_F.size();
        if (t == jq_Primitive.LONG) return stack_L.size();
        if (t == jq_Primitive.DOUBLE) return stack_D.size();
        jq.UNREACHABLE();
        return 0;
    }

    public RegisterOperand makeGuardReg() {
        return new RegisterOperand(new Register(-1, true), null);
    }
    
    public static class Register {
        private int id; private boolean isTemp;
        Register(int id, boolean isTemp) { this.id = id; this.isTemp = isTemp; }
        public int getNumber() { return id; }
        public String toString() { return (isTemp?"T":"R")+id; }
    }

}

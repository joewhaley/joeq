// RegisterFactory.java, created Fri Jan 11 16:42:38 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Quad;

import java.util.ArrayList;
import java.util.Iterator;

import joeq.Clazz.jq_Method;
import joeq.Clazz.jq_Primitive;
import joeq.Clazz.jq_Type;
import joeq.Compil3r.Quad.Operand.RegisterOperand;
import joeq.Util.Assert;
import joeq.Util.Strings;
import joeq.Util.Collections.AppendIterator;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
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
    
    private RegisterFactory() { }

    /** Creates new RegisterFactory */
    public RegisterFactory(jq_Method m) {
        int nlocals = m.getMaxLocals();
        local_I = new ArrayList(nlocals);
        local_F = new ArrayList(nlocals);
        local_L = new ArrayList(nlocals);
        local_D = new ArrayList(nlocals);
        local_A = new ArrayList(nlocals);
        for (int i=0; i<nlocals; ++i) {
            local_I.add(new Register(i, false));
            local_F.add(new Register(i, false));
            local_L.add(new Register(i, false));
            local_D.add(new Register(i, false));
            local_A.add(new Register(i, false));
        }
        int nstack = m.getMaxStack();
        stack_I = new ArrayList(nstack);
        stack_F = new ArrayList(nstack);
        stack_L = new ArrayList(nstack);
        stack_D = new ArrayList(nstack);
        stack_A = new ArrayList(nstack);
        for (int i=0; i<nstack; ++i) {
            stack_I.add(new Register(i, true));
            stack_F.add(new Register(i, true));
            stack_L.add(new Register(i, true));
            stack_D.add(new Register(i, true));
            stack_A.add(new Register(i, true));
        }
    }

    public Register getStack(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)stack_A.get(i);
        if (t.isIntLike()) return (Register)stack_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)stack_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)stack_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)stack_D.get(i);
        Assert.UNREACHABLE();
        return null;
    }
    
    public Register getNewStack(int i, jq_Type t) {
        if (t.isReferenceType()) {
            while (i >= stack_A.size())
                stack_A.add(new Register(stack_A.size(), true));
            return (Register)stack_A.get(i);
        }
        if (t.isIntLike()) {
            while (i >= stack_I.size())
                stack_I.add(new Register(stack_I.size(), true));
            return (Register)stack_I.get(i);
        }
        if (t == jq_Primitive.FLOAT) {
            while (i >= stack_F.size())
                stack_F.add(new Register(stack_F.size(), true));
            return (Register)stack_F.get(i);
        }
        if (t == jq_Primitive.LONG) {
            while (i >= stack_L.size())
                stack_L.add(new Register(stack_L.size(), true));
            return (Register)stack_L.get(i);
        }
        if (t == jq_Primitive.DOUBLE) {
            while (i >= stack_D.size())
                stack_D.add(i, new Register(stack_D.size(), true));
            return (Register)stack_D.get(i);
        }
        Assert.UNREACHABLE();
        return null;
    }
    
    public Register getLocal(int i, jq_Type t) {
        if (t.isReferenceType()) return (Register)local_A.get(i);
        if (t.isIntLike()) return (Register)local_I.get(i);
        if (t == jq_Primitive.FLOAT) return (Register)local_F.get(i);
        if (t == jq_Primitive.LONG) return (Register)local_L.get(i);
        if (t == jq_Primitive.DOUBLE) return (Register)local_D.get(i);
        Assert.UNREACHABLE();
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
        Assert.UNREACHABLE();
        return null;
    }

    public int getLocalSize(jq_Type t) {
        if (t.isReferenceType()) return local_A.size();
        if (t.isIntLike()) return local_I.size();
        if (t == jq_Primitive.FLOAT) return local_F.size();
        if (t == jq_Primitive.LONG) return local_L.size();
        if (t == jq_Primitive.DOUBLE) return local_D.size();
        Assert.UNREACHABLE();
        return 0;
    }
    public int getStackSize(jq_Type t) {
        if (t.isReferenceType()) return stack_A.size();
        if (t.isIntLike()) return stack_I.size();
        if (t == jq_Primitive.FLOAT) return stack_F.size();
        if (t == jq_Primitive.LONG) return stack_L.size();
        if (t == jq_Primitive.DOUBLE) return stack_D.size();
        Assert.UNREACHABLE();
        return 0;
    }

    public static RegisterOperand makeGuardReg() {
        return new RegisterOperand(new Register(-1, true), null);
    }

    void renumberRegisterList(ArrayList list, int n) {
        Iterator i;
        for (i = list.iterator(); i.hasNext(); ) {
            Register r = (Register)i.next();
            r.setNumber(r.getNumber()+n);
        }
    }

    static void fillRegisters(ArrayList list, int offset, int n, boolean t) {
        Assert._assert(list.size() == 0);
        for (int i=0; i<n; ++i) {
            list.add(i, new Register(offset+i, t));
        }
    }

    public RegisterFactory deep_copy() {
        RegisterFactory that = new RegisterFactory();
        fillRegisters(that.local_I = new ArrayList(this.local_I.size()),
                      0, this.local_I.size(), false);
        fillRegisters(that.local_F = new ArrayList(this.local_F.size()),
                      0, this.local_F.size(), false);
        fillRegisters(that.local_L = new ArrayList(this.local_L.size()),
                      0, this.local_L.size(), false);
        fillRegisters(that.local_D = new ArrayList(this.local_D.size()),
                      0, this.local_D.size(), false);
        fillRegisters(that.local_A = new ArrayList(this.local_A.size()),
                      0, this.local_A.size(), false);
        fillRegisters(that.stack_I = new ArrayList(this.stack_I.size()),
                      0, this.stack_I.size(), true);
        fillRegisters(that.stack_F = new ArrayList(this.stack_F.size()),
                      0, this.stack_F.size(), true);
        fillRegisters(that.stack_L = new ArrayList(this.stack_L.size()),
                      0, this.stack_L.size(), true);
        fillRegisters(that.stack_D = new ArrayList(this.stack_D.size()),
                      0, this.stack_D.size(), true);
        fillRegisters(that.stack_A = new ArrayList(this.stack_A.size()),
                      0, this.stack_A.size(), true);
        return that;
    }

    public RegisterFactory merge(RegisterFactory from) {
        RegisterFactory that = new RegisterFactory();
        fillRegisters(that.local_I = new ArrayList(from.local_I.size()),
                      this.local_I.size(), from.local_I.size(), false);
        fillRegisters(that.local_F = new ArrayList(from.local_F.size()),
                      this.local_F.size(), from.local_F.size(), false);
        fillRegisters(that.local_L = new ArrayList(from.local_L.size()),
                      this.local_L.size(), from.local_L.size(), false);
        fillRegisters(that.local_D = new ArrayList(from.local_D.size()),
                      this.local_D.size(), from.local_D.size(), false);
        fillRegisters(that.local_A = new ArrayList(from.local_A.size()),
                      this.local_A.size(), from.local_A.size(), false);
        fillRegisters(that.stack_I = new ArrayList(from.stack_I.size()),
                      this.stack_I.size(), from.stack_I.size(), true);
        fillRegisters(that.stack_F = new ArrayList(from.stack_F.size()),
                      this.stack_I.size(), from.stack_F.size(), true);
        fillRegisters(that.stack_L = new ArrayList(from.stack_L.size()),
                      this.stack_I.size(), from.stack_L.size(), true);
        fillRegisters(that.stack_D = new ArrayList(from.stack_D.size()),
                      this.stack_I.size(), from.stack_D.size(), true);
        fillRegisters(that.stack_A = new ArrayList(from.stack_A.size()),
                      this.stack_I.size(), from.stack_A.size(), true);

        // append lists
        this.local_I.addAll(that.local_I);
        this.local_F.addAll(that.local_F);
        this.local_L.addAll(that.local_L);
        this.local_D.addAll(that.local_D);
        this.local_A.addAll(that.local_A);
        this.stack_I.addAll(that.stack_I);
        this.stack_F.addAll(that.stack_F);
        this.stack_L.addAll(that.stack_L);
        this.stack_D.addAll(that.stack_D);
        this.stack_A.addAll(that.stack_A);

        return that;
    }
    
    public int totalSize() {
        return local_I.size()+local_F.size()+local_L.size()+local_D.size()+local_A.size()+
               stack_I.size()+stack_F.size()+stack_L.size()+stack_D.size()+stack_A.size();
    }
    
    public Iterator iterator() {
        Iterator i = new AppendIterator(local_I.iterator(), local_F.iterator());
        i = new AppendIterator(i, local_L.iterator());
        i = new AppendIterator(i, local_D.iterator());
        i = new AppendIterator(i, local_A.iterator());
        i = new AppendIterator(i, stack_I.iterator());
        i = new AppendIterator(i, stack_F.iterator());
        i = new AppendIterator(i, stack_L.iterator());
        i = new AppendIterator(i, stack_D.iterator());
        i = new AppendIterator(i, stack_A.iterator());
        return i;
    }
    
    public String toString() {
        return "Local: (I="+local_I.size()+
                      ",F="+local_F.size()+
                      ",L="+local_L.size()+
                      ",D="+local_D.size()+
                      ",A="+local_A.size()+
             ") Stack: (I="+stack_I.size()+
                      ",F="+stack_F.size()+
                      ",L="+stack_L.size()+
                      ",D="+stack_D.size()+
                      ",A="+stack_A.size()+
                      ")";
    }

    public String fullDump() {
        StringBuffer sb = new StringBuffer();
        sb.append("Local_I: "+local_I);
        sb.append(Strings.lineSep+"Local_F: "+local_F);
        sb.append(Strings.lineSep+"Local_L: "+local_L);
        sb.append(Strings.lineSep+"Local_D: "+local_D);
        sb.append(Strings.lineSep+"Local_A: "+local_A);
        sb.append(Strings.lineSep+"Stack_I: "+stack_I);
        sb.append(Strings.lineSep+"Stack_F: "+stack_F);
        sb.append(Strings.lineSep+"Stack_L: "+stack_L);
        sb.append(Strings.lineSep+"Stack_D: "+stack_D);
        sb.append(Strings.lineSep+"Stack_A: "+stack_A);
        return sb.toString();
    }

    public static class Register {
        private int id; private boolean isTemp;
        Register(int id, boolean isTemp) { this.id = id; this.isTemp = isTemp; }
        public int getNumber() { return id; }
        public void setNumber(int id) { this.id = id; }
        public boolean isTemp() { return isTemp; }
        public String toString() { return (isTemp?"T":"R")+id; }
    }

}

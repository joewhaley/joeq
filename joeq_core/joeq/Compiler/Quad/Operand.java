/*
 * Operand.java
 *
 * Created on April 22, 2001, 12:33 AM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import Clazz.*;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import jq;

public interface Operand {

    public Quad getQuad();
    public void attachToQuad(Quad q);
    public Operand copy();
    public boolean isSimilar(Operand that);

    public static abstract class Util {
        public static boolean isNullConstant(Operand op) {
            return op instanceof AConstOperand && ((AConstOperand)op).getValue() == null;
        }
        public static boolean isConstant(Operand op) {
            return op instanceof AConstOperand ||
                   op instanceof IConstOperand ||
                   op instanceof FConstOperand ||
                   op instanceof LConstOperand ||
                   op instanceof DConstOperand;
        }
    }
    
    public static class RegisterOperand implements Operand {
        private Quad instruction;
        private Register register; private jq_Type type; private int flags;
        public static final int PRECISE_TYPE = 0x1;
        public Object scratchObject;
        public RegisterOperand(Register reg, jq_Type type) {
            this(reg, type, 0);
        }
        public RegisterOperand(Register reg, jq_Type type, int flags) {
            this.register = reg; this.type = type; this.flags = flags;
        }
        public Register getRegister() { return register; }
        public jq_Type getType() { return type; }
        public int getFlags() { return flags; }
        public void setFlags(int f) { flags = f; }
        public void meetFlags(int f) { flags &= f; }
        public boolean isExactType() { return (flags & PRECISE_TYPE) != 0; }
        public void clearExactType() { flags &= ~PRECISE_TYPE; }
        public boolean hasMoreConservativeFlags(RegisterOperand that) { return that.getFlags() == (getFlags() | that.getFlags()); }
        public Operand copy() { return new RegisterOperand(this.register, this.type, this.flags); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof RegisterOperand && ((RegisterOperand)that).getRegister() == this.getRegister(); }
        public String toString() { return register+" "+((type==null)?"<g>":type.shortName()); }
    }
    
    public static class AConstOperand implements Operand {
        private Quad instruction;
        private final Object value;
        public AConstOperand(Object v) { this.value = v; }
        //public int hashCode() { return System.identityHashCode(value); }
        //public boolean equals(Object that) { return equals((AConstOperand)that); }
        //public boolean equals(AConstOperand that) { return this.value == that.value; }
        public Object getValue() { return value; }
        public String toString() {
            if (value instanceof String) return "AConst: \""+value+"\"";
            return "AConst: "+value;
        }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public Operand copy() { return new AConstOperand(value); }
        public boolean isSimilar(Operand that) { return that instanceof AConstOperand && ((AConstOperand)that).getValue() == this.getValue(); }
    }
    
    public static class IConstOperand implements Operand {
        private Quad instruction;
        private final int value;
        public IConstOperand(int v) { this.value = v; }
        //public int hashCode() { return value; }
        //public boolean equals(Object that) { return equals((IConstOperand)that); }
        //public boolean equals(IConstOperand that) { return this.value == that.value; }
        public int getValue() { return value; }
        public String toString() { return "IConst: "+value; }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public Operand copy() { return new IConstOperand(value); }
        public boolean isSimilar(Operand that) { return that instanceof IConstOperand && ((IConstOperand)that).getValue() == this.getValue(); }
    }
    
    public static class FConstOperand implements Operand {
        private Quad instruction;
        private final float value;
        public FConstOperand(float v) { this.value = v; }
        //public int hashCode() { return Float.floatToRawIntBits(value); }
        //public boolean equals(Object that) { return equals((FConstOperand)that); }
        //public boolean equals(FConstOperand that) { return this.value == that.value; }
        public float getValue() { return value; }
        public String toString() { return "FConst: "+value; }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public Operand copy() { return new FConstOperand(value); }
        public boolean isSimilar(Operand that) { return that instanceof FConstOperand && ((FConstOperand)that).getValue() == this.getValue(); }
    }

    public static class LConstOperand implements Operand {
        private Quad instruction;
        private final long value;
        public LConstOperand(long v) { this.value = v; }
        //public int hashCode() { return (int)(value>>32) ^ (int)value; }
        //public boolean equals(Object that) { return equals((LConstOperand)that); }
        //public boolean equals(DConstOperand that) { return this.value == that.value; }
        public long getValue() { return value; }
        public String toString() { return "LConst: "+value; }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public Operand copy() { return new LConstOperand(value); }
        public boolean isSimilar(Operand that) { return that instanceof LConstOperand && ((LConstOperand)that).getValue() == this.getValue(); }
    }

    public static class DConstOperand implements Operand {
        private Quad instruction;
        private final double value;
        public DConstOperand(double v) { this.value = v; }
        //public int hashCode() { long v = Double.doubleToRawLongBits(value); return (int)(v>>32) ^ (int)v; }
        //public boolean equals(Object that) { return equals((DConstOperand)that); }
        //public boolean equals(DConstOperand that) { return this.value == that.value; }
        public double getValue() { return value; }
        public String toString() { return "DConst: "+value; }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public Operand copy() { return new DConstOperand(value); }
        public boolean isSimilar(Operand that) { return that instanceof DConstOperand && ((DConstOperand)that).getValue() == this.getValue(); }
    }

    public static class UnnecessaryGuardOperand implements Operand {
        private Quad instruction;
        public UnnecessaryGuardOperand() {}
        //public int hashCode() { return 67; }
        //public boolean equals(Object that) { return that instanceof UnnecessaryGuardOperand; }
        public String toString() { return "<no guard>"; }
        public Operand copy() { return new UnnecessaryGuardOperand(); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof UnnecessaryGuardOperand; }
    }
    
    public static class ConditionOperand implements Operand {
        private Quad instruction; byte condition;
        public ConditionOperand(byte c) { condition = c; }
        //public int hashCode() { return condition; }
        //public boolean equals(Object that) { return this.equals((ConditionOperand)that); }
        //public boolean equals(ConditionOperand that) { return this.condition == that.condition; }
        public byte getCondition() { return condition; }
        public String toString() { return BytecodeVisitor.cmpopnames[condition]; }
        public Operand copy() { return new UnnecessaryGuardOperand(); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof ConditionOperand && ((ConditionOperand)that).getCondition() == this.getCondition(); }
    }

    public static class FieldOperand implements Operand {
        private Quad instruction; jq_Field field;
        public FieldOperand(jq_Field f) { field = f; }
        public jq_Field getField() { return field; }
        public String toString() { return "."+field.getName(); }
        public Operand copy() { return new FieldOperand(field); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof FieldOperand && ((FieldOperand)that).getField() == this.getField(); }
    }
    
    public static class TypeOperand implements Operand {
        private Quad instruction; jq_Type type;
        TypeOperand(jq_Type f) { type = f; }
        public jq_Type getType() { return type; }
        public String toString() { return type.toString(); }
        public Operand copy() { return new TypeOperand(type); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof TypeOperand && ((TypeOperand)that).getType() == this.getType(); }
    }
    
    public static class TargetOperand implements Operand {
        private Quad instruction; BasicBlock target;
        public TargetOperand(BasicBlock t) { target = t; }
        public BasicBlock getTarget() { return target; }
        public String toString() { return target.toString(); }
        public Operand copy() { return new TargetOperand(target); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof TargetOperand && ((TargetOperand)that).getTarget() == this.getTarget(); }
    }
    
    public static class MethodOperand implements Operand {
        private Quad instruction; jq_Method target;
        public MethodOperand(jq_Method t) { target = t; }
        public jq_Method getMethod() { return target; }
        public String toString() { return target.toString(); }
        public Operand copy() { return new MethodOperand(target); }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return that instanceof MethodOperand && ((MethodOperand)that).getMethod() == this.getMethod(); }
    }
    
    public static class IntValueTableOperand implements Operand {
        private Quad instruction; int[] table;
        public IntValueTableOperand(int[] t) { table = t; }
        public void set(int i, int b) { table[i] = b; }
        public int get(int i) { return table[i]; }
	public int size() { return table.length; }
        public String toString() {
            StringBuffer sb = new StringBuffer("{ ");
            if (table.length > 0) {
                sb.append(table[0]);
                for (int i=1; i<table.length; ++i) {
                    sb.append(", ");
                    sb.append(table[i]);
                }
            }
            sb.append(" }");
            return sb.toString();
        }
        public Operand copy() {
            int[] t2 = new int[this.table.length];
            System.arraycopy(this.table, 0, t2, 0, t2.length);
            return new IntValueTableOperand(t2);
        }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return false; }
    }
    
    public static class BasicBlockTableOperand implements Operand {
        private Quad instruction; BasicBlock[] table;
        public BasicBlockTableOperand(BasicBlock[] t) { table = t; }
        public void set(int i, BasicBlock b) { table[i] = b; }
        public BasicBlock get(int i) { return table[i]; }
	public int size() { return table.length; }
        public String toString() {
            StringBuffer sb = new StringBuffer("{ ");
            if (table.length > 0) {
                sb.append(table[0]);
                for (int i=1; i<table.length; ++i) {
                    sb.append(", ");
                    sb.append(table[i]);
                }
            }
            sb.append(" }");
            return sb.toString();
        }
        public Operand copy() {
            BasicBlock[] t2 = new BasicBlock[this.table.length];
            System.arraycopy(this.table, 0, t2, 0, t2.length);
            return new BasicBlockTableOperand(t2);
        }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return false; }
    }
    
    public static class ParamListOperand implements Operand {
        private Quad instruction; RegisterOperand[] params;
        public ParamListOperand(RegisterOperand[] t) { params = t; }
        public void set(int i, RegisterOperand b) { params[i] = b; }
        public RegisterOperand get(int i) { return params[i]; }
        public int length() { return params.length; }
        public String toString() {
            StringBuffer sb = new StringBuffer("(");
            if (params.length > 0) {
                sb.append(params[0]);
                for (int i=1; i<params.length; ++i) {
                    sb.append(", ");
                    sb.append(params[i]);
                }
            }
            sb.append(")");
            return sb.toString();
        }
        public Operand copy() {
            RegisterOperand[] t2 = new RegisterOperand[this.params.length];
            System.arraycopy(this.params, 0, t2, 0, t2.length);
            return new ParamListOperand(t2);
        }
        public void attachToQuad(Quad q) { jq.assert(instruction == null); instruction = q; }
        public Quad getQuad() { return instruction; }
        public boolean isSimilar(Operand that) { return false; }
    }
    
}

/*
 * Operator.java
 *
 * Created on May 11, 2001, 12:17 AM
 *
 */

package Compil3r.Quad;

import Clazz.*;
import Main.jq;
import Bootstrap.PrimordialClassLoader;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operand.TargetOperand;
import Compil3r.Quad.Operand.ConditionOperand;
import Compil3r.Quad.Operand.IConstOperand;
import Compil3r.Quad.Operand.FConstOperand;
import Compil3r.Quad.Operand.LConstOperand;
import Compil3r.Quad.Operand.DConstOperand;
import Compil3r.Quad.Operand.AConstOperand;
import Compil3r.Quad.Operand.TypeOperand;
import Compil3r.Quad.Operand.FieldOperand;
import Compil3r.Quad.Operand.MethodOperand;
import Compil3r.Quad.Operand.ParamListOperand;
import Compil3r.Quad.Operand.IntValueTableOperand;
import Compil3r.Quad.Operand.BasicBlockTableOperand;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Util.Templates.UnmodifiableList;
import Interpreter.QuadInterpreter.State;
import Interpreter.QuadInterpreter.UninitializedReference;
import java.util.Set;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Operator {

    public void accept(Quad q, QuadVisitor qv) {
        qv.visitQuad(q);
    }
    public UnmodifiableList.jq_Class getThrownExceptions() {
        return noexceptions;
    }
    public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) {
        return noregisters;
    }
    public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) {
        return noregisters;
    }

    public abstract boolean hasSideEffects();
    
    public abstract void interpret(Quad q, State s);
    
    static int getIntOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return ((Number)s.getReg(((RegisterOperand)op).getRegister())).intValue();
	else
	    return ((IConstOperand)op).getValue();
    }
    
    static float getFloatOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return ((Float)s.getReg(((RegisterOperand)op).getRegister())).floatValue();
	else
	    return ((FConstOperand)op).getValue();
    }
    
    static long getLongOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return ((Long)s.getReg(((RegisterOperand)op).getRegister())).longValue();
	else
	    return ((LConstOperand)op).getValue();
    }
    
    static double getDoubleOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return ((Double)s.getReg(((RegisterOperand)op).getRegister())).doubleValue();
	else
	    return ((DConstOperand)op).getValue();
    }
    
    static Object getObjectOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return s.getReg(((RegisterOperand)op).getRegister());
	else
	    return ((AConstOperand)op).getValue();
    }

    static Object getWrappedOpValue(Operand op, State s) {
	if (op instanceof RegisterOperand)
	    return s.getReg(((RegisterOperand)op).getRegister());
	else if (op instanceof AConstOperand)
	    return ((AConstOperand)op).getValue();
	else if (op instanceof IConstOperand)
	    return new Integer(((IConstOperand)op).getValue());
	else if (op instanceof FConstOperand)
	    return new Float(((FConstOperand)op).getValue());
	else if (op instanceof LConstOperand)
	    return new Long(((LConstOperand)op).getValue());
	else if (op instanceof DConstOperand)
	    return new Double(((DConstOperand)op).getValue());
	jq.UNREACHABLE();
	return null;
    }

    public static UnmodifiableList.RegisterOperand getReg1(Quad q) {
        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1());
    }
    public static UnmodifiableList.RegisterOperand getReg1_check(Quad q) {
        if (q.getOp1() instanceof RegisterOperand)
            return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1());
        else
            return noregisters;
    }
    public static UnmodifiableList.RegisterOperand getReg2(Quad q) {
        if (q.getOp2() instanceof RegisterOperand)
            return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2());
        else
            return noregisters;
    }
    public static UnmodifiableList.RegisterOperand getReg3(Quad q) {
        if (q.getOp3() instanceof RegisterOperand)
            return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp3());
        else
            return noregisters;
    }
    protected static UnmodifiableList.RegisterOperand getReg12(Quad q) {
        if (q.getOp1() instanceof RegisterOperand) {
            if (q.getOp2() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2());
            else
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1());
        } else {
            if (q.getOp2() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2());
            else
                return noregisters;
        }
    }
    protected static UnmodifiableList.RegisterOperand getReg23(Quad q) {
        if (q.getOp2() instanceof RegisterOperand) {
            if (q.getOp3() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3());
            else
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2());
        } else {
            if (q.getOp3() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp3());
            else
                return noregisters;
        }
    }
    protected static UnmodifiableList.RegisterOperand getReg24(Quad q) {
        if (q.getOp2() instanceof RegisterOperand) {
            if (q.getOp4() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp4());
            else
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2());
        } else {
            if (q.getOp4() instanceof RegisterOperand)
                return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp4());
            else
                return noregisters;
        }
    }
    protected static UnmodifiableList.RegisterOperand getReg124(Quad q) {
        if (q.getOp1() instanceof RegisterOperand) {
            if (q.getOp2() instanceof RegisterOperand) {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp4());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2());
            } else {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp4());
                else
                    return noregisters;
            }
        } else return getReg24(q);
    }
    protected static UnmodifiableList.RegisterOperand getReg123(Quad q) {
        if (q.getOp1() instanceof RegisterOperand) {
            if (q.getOp2() instanceof RegisterOperand) {
                if (q.getOp3() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2());
            } else {
                if (q.getOp3() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp3());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1());
            }
        } else return getReg23(q);
    }
    protected static UnmodifiableList.RegisterOperand getReg234(Quad q) {
        if (q.getOp2() instanceof RegisterOperand) {
            if (q.getOp3() instanceof RegisterOperand) {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3(), (RegisterOperand)q.getOp4());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3());
            } else {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp4());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp2());
            }
        } else {
            if (q.getOp3() instanceof RegisterOperand) {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp3(), (RegisterOperand)q.getOp4());
                else
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp3());
            } else {
                if (q.getOp4() instanceof RegisterOperand)
                    return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp4());
                else
                    return noregisters;
            }
        }
    }
    protected static UnmodifiableList.RegisterOperand getReg1234(Quad q) {
        if (q.getOp1() instanceof RegisterOperand) {
            if (q.getOp2() instanceof RegisterOperand) {
                if (q.getOp3() instanceof RegisterOperand) {
                    if (q.getOp4() instanceof RegisterOperand)
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3(), (RegisterOperand)q.getOp4());
                    else
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp3());
                } else {
                    if (q.getOp4() instanceof RegisterOperand)
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2(), (RegisterOperand)q.getOp4());
                    else
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp2());
                }
            } else {
                if (q.getOp3() instanceof RegisterOperand) {
                    if (q.getOp4() instanceof RegisterOperand)
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp3(), (RegisterOperand)q.getOp4());
                    else
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp3());
                } else {
                    if (q.getOp4() instanceof RegisterOperand)
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1(), (RegisterOperand)q.getOp4());
                    else
                        return new UnmodifiableList.RegisterOperand((RegisterOperand)q.getOp1());
                }
            }
        } else return getReg234(q);
    }
    
    public static final UnmodifiableList.RegisterOperand noregisters;
    public static final UnmodifiableList.jq_Class noexceptions;
    public static final UnmodifiableList.jq_Class anyexception;
    public static final UnmodifiableList.jq_Class resolutionexceptions;
    public static final UnmodifiableList.jq_Class nullptrexception;
    public static final UnmodifiableList.jq_Class arrayboundsexception;
    public static final UnmodifiableList.jq_Class arithexception;
    public static final UnmodifiableList.jq_Class arraystoreexception;
    public static final UnmodifiableList.jq_Class negativesizeexception;
    public static final UnmodifiableList.jq_Class classcastexceptions;
    public static final UnmodifiableList.jq_Class illegalmonitorstateexception;
    static {
        noregisters = new UnmodifiableList.RegisterOperand( new RegisterOperand[0] );
        noexceptions = new UnmodifiableList.jq_Class( new jq_Class[0] );
        anyexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangThrowable()
            );
        resolutionexceptions = anyexception; // a little conservative
        nullptrexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangNullPointerException()
            );
        arrayboundsexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangArrayIndexOutOfBoundsException()
            );
        arraystoreexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangArrayStoreException()
            );
        negativesizeexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangNegativeArraySizeException()
            );
        arithexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangArithmeticException()
            );
        classcastexceptions = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangThrowable()
            );
        illegalmonitorstateexception = new UnmodifiableList.jq_Class(
            PrimordialClassLoader.getJavaLangIllegalMonitorStateException()
            );
    }
    
    public static abstract class Move extends Operator {
        
        public static Quad create(int id, Move operator, RegisterOperand dst, Operand src) {
            return new Quad(id, operator, dst, src);
        }
        public static Move getMoveOp(jq_Type type) {
            if (type.isReferenceType()) return MOVE_A.INSTANCE;
            if (type.isIntLike()) return MOVE_I.INSTANCE;
            if (type == jq_Primitive.FLOAT) return MOVE_F.INSTANCE;
            if (type == jq_Primitive.LONG) return MOVE_L.INSTANCE;
            if (type == jq_Primitive.DOUBLE) return MOVE_D.INSTANCE;
            jq.UNREACHABLE(); return null;
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public boolean hasSideEffects() { return false; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitMove(q);
            super.accept(q, qv);
        }
        
        public static class MOVE_I extends Move {
            public static final MOVE_I INSTANCE = new MOVE_I();
            private MOVE_I() { }
            public String toString() { return "MOVE_I"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), getIntOpValue(getSrc(q), s));
	    }
        }
        public static class MOVE_F extends Move {
            public static final MOVE_F INSTANCE = new MOVE_F();
            private MOVE_F() { }
            public String toString() { return "MOVE_F"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), getFloatOpValue(getSrc(q), s));
	    }
        }
        public static class MOVE_L extends Move {
            public static final MOVE_L INSTANCE = new MOVE_L();
            private MOVE_L() { }
            public String toString() { return "MOVE_L"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), getLongOpValue(getSrc(q), s));
	    }
        }
        public static class MOVE_D extends Move {
            public static final MOVE_D INSTANCE = new MOVE_D();
            private MOVE_D() { }
            public String toString() { return "MOVE_D"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), getDoubleOpValue(getSrc(q), s));
	    }
        }
        public static class MOVE_A extends Move {
            public static final MOVE_A INSTANCE = new MOVE_A();
            private MOVE_A() { }
            public String toString() { return "MOVE_A"; }
	    public void interpret(Quad q, State s) {
		s.putReg_A(getDest(q).getRegister(), getObjectOpValue(getSrc(q), s));
	    }
        }
    }

    public static abstract class Binary extends Operator {
        
        public static Quad create(int id, Binary operator, RegisterOperand dst, Operand src1, Operand src2) {
            return new Quad(id, operator, dst, src1, src2);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc1(Quad q) { return q.getOp2(); }
        public static Operand getSrc2(Quad q) { return q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc1(Quad q, Operand o) { q.setOp2(o); }
	public static void setSrc2(Quad q, Operand o) { q.setOp3(o); }
        public boolean hasSideEffects() { return false; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg23(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitBinary(q);
            super.accept(q, qv);
        }
        
        public static class ADD_I extends Binary {
            public static final ADD_I INSTANCE = new ADD_I();
            private ADD_I() { }
            public String toString() { return "ADD_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) + getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class SUB_I extends Binary {
            public static final SUB_I INSTANCE = new SUB_I();
            private SUB_I() { }
            public String toString() { return "SUB_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) - getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class MUL_I extends Binary {
            public static final MUL_I INSTANCE = new MUL_I();
            private MUL_I() { }
            public String toString() { return "MUL_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) * getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class DIV_I extends Binary {
            public static final DIV_I INSTANCE = new DIV_I();
            private DIV_I() { }
            public String toString() { return "DIV_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) / getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class REM_I extends Binary {
            public static final REM_I INSTANCE = new REM_I();
            private REM_I() { }
            public String toString() { return "REM_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) % getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class AND_I extends Binary {
            public static final AND_I INSTANCE = new AND_I();
            private AND_I() { }
            public String toString() { return "AND_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) & getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class OR_I extends Binary {
            public static final OR_I INSTANCE = new OR_I();
            private OR_I() { }
            public String toString() { return "OR_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) | getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class XOR_I extends Binary {
            public static final XOR_I INSTANCE = new XOR_I();
            private XOR_I() { }
            public String toString() { return "XOR_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) ^ getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class SHL_I extends Binary {
            public static final SHL_I INSTANCE = new SHL_I();
            private SHL_I() { }
            public String toString() { return "SHL_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) << getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class SHR_I extends Binary {
            public static final SHR_I INSTANCE = new SHR_I();
            private SHR_I() { }
            public String toString() { return "SHR_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) >> getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class USHR_I extends Binary {
            public static final USHR_I INSTANCE = new USHR_I();
            private USHR_I() { }
            public String toString() { return "USHR_I"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc1(q), s) >>> getIntOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
        public static class SHL_L extends Binary {
            public static final SHL_L INSTANCE = new SHL_L();
            private SHL_L() { }
            public String toString() { return "SHL_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) << getIntOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class SHR_L extends Binary {
            public static final SHR_L INSTANCE = new SHR_L();
            private SHR_L() { }
            public String toString() { return "SHR_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) >> getIntOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class USHR_L extends Binary {
            public static final USHR_L INSTANCE = new USHR_L();
            private USHR_L() { }
            public String toString() { return "USHR_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) >>> getIntOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class ADD_L extends Binary {
            public static final ADD_L INSTANCE = new ADD_L();
            private ADD_L() { }
            public String toString() { return "ADD_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) + getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class SUB_L extends Binary {
            public static final SUB_L INSTANCE = new SUB_L();
            private SUB_L() { }
            public String toString() { return "SUB_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) - getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class MUL_L extends Binary {
            public static final MUL_L INSTANCE = new MUL_L();
            private MUL_L() { }
            public String toString() { return "MUL_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) * getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class DIV_L extends Binary {
            public static final DIV_L INSTANCE = new DIV_L();
            private DIV_L() { }
            public String toString() { return "DIV_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) / getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class REM_L extends Binary {
            public static final REM_L INSTANCE = new REM_L();
            private REM_L() { }
            public String toString() { return "REM_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) % getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class AND_L extends Binary {
            public static final AND_L INSTANCE = new AND_L();
            private AND_L() { }
            public String toString() { return "AND_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) & getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class OR_L extends Binary {
            public static final OR_L INSTANCE = new OR_L();
            private OR_L() { }
            public String toString() { return "OR_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) | getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class XOR_L extends Binary {
            public static final XOR_L INSTANCE = new XOR_L();
            private XOR_L() { }
            public String toString() { return "XOR_L"; }
	    public void interpret(Quad q, State s) {
		long v = getLongOpValue(getSrc1(q), s) ^ getLongOpValue(getSrc2(q), s);
		s.putReg_L(getDest(q).getRegister(), v);
	    }
        }
        public static class ADD_F extends Binary {
            public static final ADD_F INSTANCE = new ADD_F();
            private ADD_F() { }
            public String toString() { return "ADD_F"; }
	    public void interpret(Quad q, State s) {
		float v = getFloatOpValue(getSrc1(q), s) + getFloatOpValue(getSrc2(q), s);
		s.putReg_F(getDest(q).getRegister(), v);
	    }
        }
        public static class SUB_F extends Binary {
            public static final SUB_F INSTANCE = new SUB_F();
            private SUB_F() { }
            public String toString() { return "SUB_F"; }
	    public void interpret(Quad q, State s) {
		float v = getFloatOpValue(getSrc1(q), s) - getFloatOpValue(getSrc2(q), s);
		s.putReg_F(getDest(q).getRegister(), v);
	    }
        }
        public static class MUL_F extends Binary {
            public static final MUL_F INSTANCE = new MUL_F();
            private MUL_F() { }
            public String toString() { return "MUL_F"; }
	    public void interpret(Quad q, State s) {
		float v = getFloatOpValue(getSrc1(q), s) * getFloatOpValue(getSrc2(q), s);
		s.putReg_F(getDest(q).getRegister(), v);
	    }
        }
        public static class DIV_F extends Binary {
            public static final DIV_F INSTANCE = new DIV_F();
            private DIV_F() { }
            public String toString() { return "DIV_F"; }
	    public void interpret(Quad q, State s) {
		float v = getFloatOpValue(getSrc1(q), s) / getFloatOpValue(getSrc2(q), s);
		s.putReg_F(getDest(q).getRegister(), v);
	    }
        }
        public static class REM_F extends Binary {
            public static final REM_F INSTANCE = new REM_F();
            private REM_F() { }
            public String toString() { return "REM_F"; }
	    public void interpret(Quad q, State s) {
		float v = getFloatOpValue(getSrc1(q), s) % getFloatOpValue(getSrc2(q), s);
		s.putReg_F(getDest(q).getRegister(), v);
	    }
        }
        public static class ADD_D extends Binary {
            public static final ADD_D INSTANCE = new ADD_D();
            private ADD_D() { }
            public String toString() { return "ADD_D"; }
	    public void interpret(Quad q, State s) {
		double v = getDoubleOpValue(getSrc1(q), s) + getDoubleOpValue(getSrc2(q), s);
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class SUB_D extends Binary {
            public static final SUB_D INSTANCE = new SUB_D();
            private SUB_D() { }
            public String toString() { return "SUB_D"; }
	    public void interpret(Quad q, State s) {
		double v = getDoubleOpValue(getSrc1(q), s) - getDoubleOpValue(getSrc2(q), s);
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class MUL_D extends Binary {
            public static final MUL_D INSTANCE = new MUL_D();
            private MUL_D() { }
            public String toString() { return "MUL_D"; }
	    public void interpret(Quad q, State s) {
		double v = getDoubleOpValue(getSrc1(q), s) * getDoubleOpValue(getSrc2(q), s);
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class DIV_D extends Binary {
            public static final DIV_D INSTANCE = new DIV_D();
            private DIV_D() { }
            public String toString() { return "DIV_D"; }
	    public void interpret(Quad q, State s) {
		double v = getDoubleOpValue(getSrc1(q), s) / getDoubleOpValue(getSrc2(q), s);
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class REM_D extends Binary {
            public static final REM_D INSTANCE = new REM_D();
            private REM_D() { }
            public String toString() { return "REM_D"; }
	    public void interpret(Quad q, State s) {
		double v = getDoubleOpValue(getSrc1(q), s) % getDoubleOpValue(getSrc2(q), s);
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class CMP_L extends Binary {
            public static final CMP_L INSTANCE = new CMP_L();
            private CMP_L() { }
            public String toString() { return "CMP_L"; }
	    public void interpret(Quad q, State s) {
		long v2 = getLongOpValue(getSrc1(q), s);
		long v1 = getLongOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), (v2>v1)?1:((v2==v1)?0:-1));
	    }
        }
        public static class CMP_F extends Binary {
            public static final CMP_F INSTANCE = new CMP_F();
            private CMP_F() { }
            public String toString() { return "CMP_F"; }
	    public void interpret(Quad q, State s) {
		float v2 = getFloatOpValue(getSrc1(q), s);
		float v1 = getFloatOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), (v2>v1)?1:((v2==v1)?0:-1));
	    }
        }
        public static class CMP_D extends Binary {
            public static final CMP_D INSTANCE = new CMP_D();
            private CMP_D() { }
            public String toString() { return "CMP_D"; }
	    public void interpret(Quad q, State s) {
		double v2 = getDoubleOpValue(getSrc1(q), s);
		double v1 = getDoubleOpValue(getSrc2(q), s);
		s.putReg_I(getDest(q).getRegister(), (v2>v1)?1:((v2==v1)?0:-1));
	    }
        }
    }

    public static abstract class Unary extends Operator {
        
        public static Quad create(int id, Unary operator, RegisterOperand dst, Operand src1) {
            return new Quad(id, operator, dst, src1);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public boolean hasSideEffects() { return false; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitUnary(q);
            super.accept(q, qv);
        }
        
        public static class NEG_I extends Unary {
            public static final NEG_I INSTANCE = new NEG_I();
            private NEG_I() { }
            public String toString() { return "NEG_I"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), -getIntOpValue(getSrc(q), s));
	    }
        }
        public static class NEG_F extends Unary {
            public static final NEG_F INSTANCE = new NEG_F();
            private NEG_F() { }
            public String toString() { return "NEG_F"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), -getFloatOpValue(getSrc(q), s));
	    }
        }
        public static class NEG_L extends Unary {
            public static final NEG_L INSTANCE = new NEG_L();
            private NEG_L() { }
            public String toString() { return "NEG_L"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), -getLongOpValue(getSrc(q), s));
	    }
        }
        public static class NEG_D extends Unary {
            public static final NEG_D INSTANCE = new NEG_D();
            private NEG_D() { }
            public String toString() { return "NEG_D"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), -getDoubleOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2LONG extends Unary {
            public static final INT_2LONG INSTANCE = new INT_2LONG();
            private INT_2LONG() { }
            public String toString() { return "INT_2LONG"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), (long)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2FLOAT extends Unary {
            public static final INT_2FLOAT INSTANCE = new INT_2FLOAT();
            private INT_2FLOAT() { }
            public String toString() { return "INT_2FLOAT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), (float)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2DOUBLE extends Unary {
            public static final INT_2DOUBLE INSTANCE = new INT_2DOUBLE();
            private INT_2DOUBLE() { }
            public String toString() { return "INT_2DOUBLE"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), (double)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class LONG_2INT extends Unary {
            public static final LONG_2INT INSTANCE = new LONG_2INT();
            private LONG_2INT() { }
            public String toString() { return "LONG_2INT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (int)getLongOpValue(getSrc(q), s));
	    }
        }
        public static class LONG_2FLOAT extends Unary {
            public static final LONG_2FLOAT INSTANCE = new LONG_2FLOAT();
            private LONG_2FLOAT() { }
            public String toString() { return "LONG_2FLOAT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), (float)getLongOpValue(getSrc(q), s));
	    }
        }
        public static class LONG_2DOUBLE extends Unary {
            public static final LONG_2DOUBLE INSTANCE = new LONG_2DOUBLE();
            private LONG_2DOUBLE() { }
            public String toString() { return "LONG_2DOUBLE"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), (double)getLongOpValue(getSrc(q), s));
	    }
        }
        public static class FLOAT_2INT extends Unary {
            public static final FLOAT_2INT INSTANCE = new FLOAT_2INT();
            private FLOAT_2INT() { }
            public String toString() { return "FLOAT_2INT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (int)getFloatOpValue(getSrc(q), s));
	    }
        }
        public static class FLOAT_2LONG extends Unary {
            public static final FLOAT_2LONG INSTANCE = new FLOAT_2LONG();
            private FLOAT_2LONG() { }
            public String toString() { return "FLOAT_2LONG"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), (long)getFloatOpValue(getSrc(q), s));
	    }
        }
        public static class FLOAT_2DOUBLE extends Unary {
            public static final FLOAT_2DOUBLE INSTANCE = new FLOAT_2DOUBLE();
            private FLOAT_2DOUBLE() { }
            public String toString() { return "FLOAT_2DOUBLE"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), (double)getFloatOpValue(getSrc(q), s));
	    }
        }
        public static class DOUBLE_2INT extends Unary {
            public static final DOUBLE_2INT INSTANCE = new DOUBLE_2INT();
            private DOUBLE_2INT() { }
            public String toString() { return "DOUBLE_2INT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (int)getDoubleOpValue(getSrc(q), s));
	    }
        }
        public static class DOUBLE_2LONG extends Unary {
            public static final DOUBLE_2LONG INSTANCE = new DOUBLE_2LONG();
            private DOUBLE_2LONG() { }
            public String toString() { return "DOUBLE_2LONG"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), (long)getDoubleOpValue(getSrc(q), s));
	    }
        }
        public static class DOUBLE_2FLOAT extends Unary {
            public static final DOUBLE_2FLOAT INSTANCE = new DOUBLE_2FLOAT();
            private DOUBLE_2FLOAT() { }
            public String toString() { return "DOUBLE_2FLOAT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), (float)getDoubleOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2BYTE extends Unary {
            public static final INT_2BYTE INSTANCE = new INT_2BYTE();
            private INT_2BYTE() { }
            public String toString() { return "INT_2BYTE"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (byte)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2CHAR extends Unary {
            public static final INT_2CHAR INSTANCE = new INT_2CHAR();
            private INT_2CHAR() { }
            public String toString() { return "INT_2CHAR"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (char)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class INT_2SHORT extends Unary {
            public static final INT_2SHORT INSTANCE = new INT_2SHORT();
            private INT_2SHORT() { }
            public String toString() { return "INT_2SHORT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), (short)getIntOpValue(getSrc(q), s));
	    }
        }
        public static class OBJECT_2INT extends Unary {
            public static final OBJECT_2INT INSTANCE = new OBJECT_2INT();
            private OBJECT_2INT() { }
            public String toString() { return "OBJECT_2INT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), Unsafe.addressOf(getObjectOpValue(getSrc(q), s)));
	    }
        }
        public static class INT_2OBJECT extends Unary {
            public static final INT_2OBJECT INSTANCE = new INT_2OBJECT();
            private INT_2OBJECT() { }
            public String toString() { return "INT_2OBJECT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_A(getDest(q).getRegister(), Unsafe.asObject(getIntOpValue(getSrc(q), s)));
	    }
        }
        public static class FLOAT_2INTBITS extends Unary {
            public static final FLOAT_2INTBITS INSTANCE = new FLOAT_2INTBITS();
            private FLOAT_2INTBITS() { }
            public String toString() { return "FLOAT_2INTBITS"; }
	    public void interpret(Quad q, State s) {
		s.putReg_I(getDest(q).getRegister(), Float.floatToRawIntBits(getFloatOpValue(getSrc(q), s)));
	    }
        }
        public static class INTBITS_2FLOAT extends Unary {
            public static final INTBITS_2FLOAT INSTANCE = new INTBITS_2FLOAT();
            private INTBITS_2FLOAT() { }
            public String toString() { return "INTBITS_2FLOAT"; }
	    public void interpret(Quad q, State s) {
		s.putReg_F(getDest(q).getRegister(), Float.intBitsToFloat(getIntOpValue(getSrc(q), s)));
	    }
        }
        public static class DOUBLE_2LONGBITS extends Unary {
            public static final DOUBLE_2LONGBITS INSTANCE = new DOUBLE_2LONGBITS();
            private DOUBLE_2LONGBITS() { }
            public String toString() { return "DOUBLE_2LONGBITS"; }
	    public void interpret(Quad q, State s) {
		s.putReg_L(getDest(q).getRegister(), Double.doubleToRawLongBits(getDoubleOpValue(getSrc(q), s)));
	    }
        }
        public static class LONGBITS_2DOUBLE extends Unary {
            public static final LONGBITS_2DOUBLE INSTANCE = new LONGBITS_2DOUBLE();
            private LONGBITS_2DOUBLE() { }
            public String toString() { return "LONGBITS_2DOUBLE"; }
	    public void interpret(Quad q, State s) {
		s.putReg_D(getDest(q).getRegister(), Double.longBitsToDouble(getLongOpValue(getSrc(q), s)));
	    }
        }
    }

    public static abstract class ALoad extends Operator {
        
        public static Quad create(int id, ALoad operator, RegisterOperand dst, Operand base, Operand ind, Operand guard) {
            return new Quad(id, operator, dst, base, ind, guard);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getBase(Quad q) { return q.getOp2(); }
        public static Operand getIndex(Quad q) { return q.getOp3(); }
        public static Operand getGuard(Quad q) { return q.getOp4(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setBase(Quad q, Operand o) { q.setOp2(o); }
	public static void setIndex(Quad q, Operand o) { q.setOp3(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return false; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg234(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitALoad(q);
            qv.visitArray(q);
            qv.visitLoad(q);
            super.accept(q, qv);
        }
        
        public static class ALOAD_I extends ALoad {
            public static final ALOAD_I INSTANCE = new ALOAD_I();
            private ALOAD_I() { }
            public String toString() { return "ALOAD_I"; }
	    public void interpret(Quad q, State s) {
		int[] a = (int[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_I(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_L extends ALoad {
            public static final ALOAD_L INSTANCE = new ALOAD_L();
            private ALOAD_L() { }
            public String toString() { return "ALOAD_L"; }
	    public void interpret(Quad q, State s) {
		long[] a = (long[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_L(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_F extends ALoad {
            public static final ALOAD_F INSTANCE = new ALOAD_F();
            private ALOAD_F() { }
            public String toString() { return "ALOAD_F"; }
	    public void interpret(Quad q, State s) {
		float[] a = (float[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_F(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_D extends ALoad {
            public static final ALOAD_D INSTANCE = new ALOAD_D();
            private ALOAD_D() { }
            public String toString() { return "ALOAD_D"; }
	    public void interpret(Quad q, State s) {
		double[] a = (double[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_D(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_A extends ALoad {
            public static final ALOAD_A INSTANCE = new ALOAD_A();
            private ALOAD_A() { }
            public String toString() { return "ALOAD_A"; }
	    public void interpret(Quad q, State s) {
		Object[] a = (Object[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_A(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_B extends ALoad {
            public static final ALOAD_B INSTANCE = new ALOAD_B();
            private ALOAD_B() { }
            public String toString() { return "ALOAD_B"; }
	    public void interpret(Quad q, State s) {
		Object a = getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		int v;
		if (a instanceof byte[]) v = ((byte[])a)[i];
		else v = ((boolean[])a)[i]?1:0;
		s.putReg_D(getDest(q).getRegister(), v);
	    }
        }
        public static class ALOAD_C extends ALoad {
            public static final ALOAD_C INSTANCE = new ALOAD_C();
            private ALOAD_C() { }
            public String toString() { return "ALOAD_C"; }
	    public void interpret(Quad q, State s) {
		char[] a = (char[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_I(getDest(q).getRegister(), a[i]);
	    }
        }
        public static class ALOAD_S extends ALoad {
            public static final ALOAD_S INSTANCE = new ALOAD_S();
            private ALOAD_S() { }
            public String toString() { return "ALOAD_S"; }
	    public void interpret(Quad q, State s) {
		short[] a = (short[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		s.putReg_I(getDest(q).getRegister(), a[i]);
	    }
        }
    }
    
    public static abstract class AStore extends Operator {
        
        public static Quad create(int id, AStore operator, Operand val, Operand base, Operand ind, Operand guard) {
            return new Quad(id, operator, val, base, ind, guard);
        }
        public static Operand getValue(Quad q) { return q.getOp1(); }
        public static Operand getBase(Quad q) { return q.getOp2(); }
        public static Operand getIndex(Quad q) { return q.getOp3(); }
        public static Operand getGuard(Quad q) { return q.getOp4(); }
	public static void setValue(Quad q, Operand o) { q.setOp1(o); }
	public static void setBase(Quad q, Operand o) { q.setOp2(o); }
	public static void setIndex(Quad q, Operand o) { q.setOp3(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1234(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitAStore(q);
            qv.visitArray(q);
            qv.visitStore(q);
            super.accept(q, qv);
        }
        
        public static class ASTORE_I extends AStore {
            public static final ASTORE_I INSTANCE = new ASTORE_I();
            private ASTORE_I() { }
            public String toString() { return "ASTORE_I"; }
	    public void interpret(Quad q, State s) {
		int[] a = (int[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		int v = getIntOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_L extends AStore {
            public static final ASTORE_L INSTANCE = new ASTORE_L();
            private ASTORE_L() { }
            public String toString() { return "ASTORE_L"; }
	    public void interpret(Quad q, State s) {
		long[] a = (long[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		long v = getLongOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_F extends AStore {
            public static final ASTORE_F INSTANCE = new ASTORE_F();
            private ASTORE_F() { }
            public String toString() { return "ASTORE_F"; }
	    public void interpret(Quad q, State s) {
		float[] a = (float[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		float v = getFloatOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_D extends AStore {
            public static final ASTORE_D INSTANCE = new ASTORE_D();
            private ASTORE_D() { }
            public String toString() { return "ASTORE_D"; }
	    public void interpret(Quad q, State s) {
		double[] a = (double[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		double v = getDoubleOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_A extends AStore {
            public static final ASTORE_A INSTANCE = new ASTORE_A();
            private ASTORE_A() { }
            public String toString() { return "ASTORE_A"; }
	    public void interpret(Quad q, State s) {
		Object[] a = (Object[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		Object v = getObjectOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_B extends AStore {
            public static final ASTORE_B INSTANCE = new ASTORE_B();
            private ASTORE_B() { }
            public String toString() { return "ASTORE_B"; }
	    public void interpret(Quad q, State s) {
		Object a = getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		if (a instanceof byte[]) ((byte[])a)[i] = (byte)getIntOpValue(getValue(q), s);
		else ((boolean[])a)[i] = getIntOpValue(getValue(q), s)!=0?true:false;
	    }
        }
        public static class ASTORE_C extends AStore {
            public static final ASTORE_C INSTANCE = new ASTORE_C();
            private ASTORE_C() { }
            public String toString() { return "ASTORE_C"; }
	    public void interpret(Quad q, State s) {
		char[] a = (char[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		char v = (char)getIntOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
        public static class ASTORE_S extends AStore {
            public static final ASTORE_S INSTANCE = new ASTORE_S();
            private ASTORE_S() { }
            public String toString() { return "ASTORE_S"; }
	    public void interpret(Quad q, State s) {
		short[] a = (short[])getObjectOpValue(getBase(q), s);
		int i = getIntOpValue(getIndex(q), s);
		short v = (short)getIntOpValue(getValue(q), s);
		a[i] = v;
	    }
        }
    }

    public static abstract class IntIfCmp extends Operator {
        public static Quad create(int id, IntIfCmp operator, Operand op0, Operand op1, ConditionOperand cond, TargetOperand target) {
            return new Quad(id, operator, op0, op1, cond, target);
        }
        public static Operand getSrc1(Quad q) { return q.getOp1(); }
        public static Operand getSrc2(Quad q) { return q.getOp2(); }
        public static ConditionOperand getCond(Quad q) { return (ConditionOperand)q.getOp3(); }
        public static TargetOperand getTarget(Quad q) { return (TargetOperand)q.getOp4(); }
	public static void setSrc1(Quad q, Operand o) { q.setOp1(o); }
	public static void setSrc2(Quad q, Operand o) { q.setOp2(o); }
	public static void setCond(Quad q, ConditionOperand o) { q.setOp3(o); }
	public static void setTarget(Quad q, TargetOperand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg12(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitIntIfCmp(q);
            qv.visitCondBranch(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class IFCMP_I extends IntIfCmp {
            public static final IFCMP_I INSTANCE = new IFCMP_I();
            private IFCMP_I() { }
            public String toString() { return "IFCMP_I"; }
	    public void interpret(Quad q, State s) {
		int s1 = getIntOpValue(getSrc1(q), s);
		int s2 = getIntOpValue(getSrc2(q), s);
		byte c = getCond(q).getCondition();
		boolean r;
		switch (c) {
		case BytecodeVisitor.CMP_EQ: r = s1 == s2; break;
		case BytecodeVisitor.CMP_NE: r = s1 != s2; break;
		case BytecodeVisitor.CMP_LT: r = s1 < s2; break;
		case BytecodeVisitor.CMP_GE: r = s1 >= s2; break;
		case BytecodeVisitor.CMP_LE: r = s1 <= s2; break;
		case BytecodeVisitor.CMP_GT: r = s1 > s2; break;
		case BytecodeVisitor.CMP_AE: r = Run_Time.MathSupport.ucmp(s1, s2); break;
		case BytecodeVisitor.CMP_UNCOND: r = true; break;
		default: jq.UNREACHABLE(); r = false; break;
		}
		if (r) s.branchTo(getTarget(q).getTarget());
	    }
        }
        public static class IFCMP_A extends IntIfCmp {
            public static final IFCMP_A INSTANCE = new IFCMP_A();
            private IFCMP_A() { }
            public String toString() { return "IFCMP_A"; }
	    public void interpret(Quad q, State s) {
		Object s1 = getObjectOpValue(getSrc1(q), s);
		Object s2 = getObjectOpValue(getSrc2(q), s);
		byte c = getCond(q).getCondition();
		boolean r;
		switch (c) {
		case BytecodeVisitor.CMP_EQ: r = s1 == s2; break;
		case BytecodeVisitor.CMP_NE: r = s1 != s2; break;
		case BytecodeVisitor.CMP_UNCOND: r = true; break;
		default: jq.UNREACHABLE(); r = false; break;
		}
		if (r) s.branchTo(getTarget(q).getTarget());
	    }
        }
    }
    
    public static abstract class Goto extends Operator {
        public static Quad create(int id, Goto operator, TargetOperand target) {
            return new Quad(id, operator, target);
        }
        public static TargetOperand getTarget(Quad q) { return (TargetOperand)q.getOp1(); }
	public static void setTarget(Quad q, TargetOperand o) { q.setOp1(o); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitGoto(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class GOTO extends Goto {
            public static final GOTO INSTANCE = new GOTO();
            private GOTO() { }
            public String toString() { return "GOTO"; }
	    public void interpret(Quad q, State s) {
		s.branchTo(getTarget(q).getTarget());
	    }
        }
    }
    
    public static abstract class Jsr extends Operator {
        public static Quad create(int id, Jsr operator, RegisterOperand loc, TargetOperand target, TargetOperand successor) {
            return new Quad(id, operator, loc, target, successor);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static TargetOperand getTarget(Quad q) { return (TargetOperand)q.getOp2(); }
        public static TargetOperand getSuccessor(Quad q) { return (TargetOperand)q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setTarget(Quad q, TargetOperand o) { q.setOp2(o); }
	public static void setSuccessor(Quad q, TargetOperand o) { q.setOp3(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitJsr(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class JSR extends Jsr {
            public static final JSR INSTANCE = new JSR();
            private JSR() { }
            public String toString() { return "JSR"; }
	    public void interpret(Quad q, State s) {
		BasicBlock bb = getSuccessor(q).getTarget();
		s.putReg(getDest(q).getRegister(), bb);
		s.branchTo(getTarget(q).getTarget());
	    }
        }
    }
    
    public static abstract class Ret extends Operator {
        public static Quad create(int id, Ret operator, RegisterOperand loc) {
            return new Quad(id, operator, loc);
        }
        public static RegisterOperand getTarget(Quad q) { return (RegisterOperand)q.getOp1(); }
	public static void setTarget(Quad q, RegisterOperand o) { q.setOp1(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitRet(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class RET extends Ret {
            public static final RET INSTANCE = new RET();
            private RET() { }
            public String toString() { return "RET"; }
	    public void interpret(Quad q, State s) {
		BasicBlock bb = (BasicBlock)s.getReg(getTarget(q).getRegister());
		s.branchTo(bb);
	    }
        }
    }
    
    public static abstract class TableSwitch extends Operator {
        public static Quad create(int id, TableSwitch operator, Operand val, IConstOperand low, TargetOperand def, int length) {
            return new Quad(id, operator, val, low, def, new BasicBlockTableOperand(new BasicBlock[length]));
        }
        public static void setTarget(Quad q, int i, BasicBlock t) {
            ((BasicBlockTableOperand)q.getOp4()).set(i, t);
        }
        public static Operand getSrc(Quad q) { return q.getOp1(); }
        public static IConstOperand getLow(Quad q) { return (IConstOperand)q.getOp2(); }
        public static TargetOperand getDefault(Quad q) { return (TargetOperand)q.getOp3(); }
        public static BasicBlock getTarget(Quad q, int i) { return ((BasicBlockTableOperand)q.getOp4()).get(i); }
        public static BasicBlockTableOperand getTargetTable(Quad q) { return (BasicBlockTableOperand)q.getOp4(); }
	public static void setSrc(Quad q, Operand o) { q.setOp1(o); }
	public static void setLow(Quad q, IConstOperand o) { q.setOp2(o); }
	public static void setDefault(Quad q, TargetOperand o) { q.setOp3(o); }
	public static void setTargetTable(Quad q, BasicBlockTableOperand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitTableSwitch(q);
            qv.visitCondBranch(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class TABLESWITCH extends TableSwitch {
            public static final TABLESWITCH INSTANCE = new TABLESWITCH();
            private TABLESWITCH() { }
            public String toString() { return "TABLESWITCH"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc(q), s);
		int lo = getLow(q).getValue();
		int hi = getTargetTable(q).size() + lo - 1;
		BasicBlock bb;
		if ((v < lo) || (v > hi))
		    bb = getDefault(q).getTarget();
		else
		    bb = getTarget(q, v-lo);
		s.branchTo(bb);
	    }
        }
    }
    
    public static abstract class LookupSwitch extends Operator {
        public static Quad create(int id, LookupSwitch operator, Operand val, TargetOperand def, int length) {
            return new Quad(id, operator, val, def, new IntValueTableOperand(new int[length]), new BasicBlockTableOperand(new BasicBlock[length]));
        }
        public static void setMatch(Quad q, int i, int t) {
            ((IntValueTableOperand)q.getOp3()).set(i, t);
        }
        public static void setTarget(Quad q, int i, BasicBlock t) {
            ((BasicBlockTableOperand)q.getOp4()).set(i, t);
        }
        public static Operand getSrc(Quad q) { return q.getOp1(); }
        public static TargetOperand getDefault(Quad q) { return (TargetOperand)q.getOp2(); }
        public static int getMatch(Quad q, int i) { return ((IntValueTableOperand)q.getOp3()).get(i); }
        public static BasicBlock getTarget(Quad q, int i) { return ((BasicBlockTableOperand)q.getOp4()).get(i); }
        public static IntValueTableOperand getValueTable(Quad q) { return (IntValueTableOperand)q.getOp3(); }
        public static BasicBlockTableOperand getTargetTable(Quad q) { return (BasicBlockTableOperand)q.getOp4(); }
	public static void setSrc(Quad q, Operand o) { q.setOp1(o); }
	public static void setDefault(Quad q, TargetOperand o) { q.setOp2(o); }
	public static void setValueTable(Quad q, IntValueTableOperand o) { q.setOp3(o); }
	public static void setTargetTable(Quad q, BasicBlockTableOperand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitLookupSwitch(q);
            qv.visitCondBranch(q);
            qv.visitBranch(q);
            super.accept(q, qv);
        }
        
        public static class LOOKUPSWITCH extends LookupSwitch {
            public static final LOOKUPSWITCH INSTANCE = new LOOKUPSWITCH();
            private LOOKUPSWITCH() { }
            public String toString() { return "LOOKUPSWITCH"; }
	    public void interpret(Quad q, State s) {
		int v = getIntOpValue(getSrc(q), s);
		IntValueTableOperand t = getValueTable(q);
		BasicBlock bb = getDefault(q).getTarget();
		for (int i=0; i<t.size(); ++i) {
		    if (v == t.get(i)) {
			bb = getTargetTable(q).get(i);
			break;
		    }
		}
		s.branchTo(bb);
	    }
        }
    }
    
    public static abstract class Return extends Operator {
        public static Quad create(int id, Return operator, Operand val) {
            return new Quad(id, operator, val);
        }
        public static Quad create(int id, Return operator) {
            return new Quad(id, operator);
        }
        public static Operand getSrc(Quad q) { return q.getOp1(); }
	public static void setSrc(Quad q, Operand o) { q.setOp1(o); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitReturn(q);
            super.accept(q, qv);
        }
        
	public void interpret(Quad q, State s) {
	    s.setReturnValue(getWrappedOpValue(getSrc(q), s));
	}

        public static class RETURN_V extends Return {
            public static final RETURN_V INSTANCE = new RETURN_V();
            private RETURN_V() { }
            public String toString() { return "RETURN_V"; }
	    public void interpret(Quad q, State s) {
		s.setReturnValue(null);
	    }
        }
        public static class RETURN_I extends Return {
            public static final RETURN_I INSTANCE = new RETURN_I();
            private RETURN_I() { }
            public String toString() { return "RETURN_I"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        }
        public static class RETURN_F extends Return {
            public static final RETURN_F INSTANCE = new RETURN_F();
            private RETURN_F() { }
            public String toString() { return "RETURN_F"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        }
        public static class RETURN_L extends Return {
            public static final RETURN_L INSTANCE = new RETURN_L();
            private RETURN_L() { }
            public String toString() { return "RETURN_L"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        }
        public static class RETURN_D extends Return {
            public static final RETURN_D INSTANCE = new RETURN_D();
            private RETURN_D() { }
            public String toString() { return "RETURN_D"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        }
        public static class RETURN_A extends Return {
            public static final RETURN_A INSTANCE = new RETURN_A();
            private RETURN_A() { }
            public String toString() { return "RETURN_A"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        }
        public static class THROW_A extends Return {
            public static final THROW_A INSTANCE = new THROW_A();
            private THROW_A() { }
            public String toString() { return "THROW_A"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return anyexception;
            }
	    public void interpret(Quad q, State s) {
		s.handleException((Throwable)getObjectOpValue(getSrc(q), s));
	    }
        }
    }

    public static abstract class Getstatic extends Operator {
        public static Quad create(int id, Getstatic operator, RegisterOperand dst, FieldOperand field) {
            return new Quad(id, operator, dst, field);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static FieldOperand getField(Quad q) { return (FieldOperand)q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setField(Quad q, FieldOperand o) { q.setOp2(o); }
        public boolean hasSideEffects() { return false; }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitGetstatic(q);
            qv.visitStaticField(q);
            qv.visitLoad(q);
            super.accept(q, qv);
        }
        
        public static class GETSTATIC_I extends Getstatic {
            public static final GETSTATIC_I INSTANCE = new GETSTATIC_I();
            private GETSTATIC_I() { }
            public String toString() { return "GETSTATIC_I"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), Reflection.getstatic_I(f));
            }
        }
        public static class GETSTATIC_F extends Getstatic {
            public static final GETSTATIC_F INSTANCE = new GETSTATIC_F();
            private GETSTATIC_F() { }
            public String toString() { return "GETSTATIC_F"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_F(getDest(q).getRegister(), Reflection.getstatic_F(f));
            }
        }
        public static class GETSTATIC_L extends Getstatic {
            public static final GETSTATIC_L INSTANCE = new GETSTATIC_L();
            private GETSTATIC_L() { }
            public String toString() { return "GETSTATIC_L"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_L(getDest(q).getRegister(), Reflection.getstatic_L(f));
            }
        }
        public static class GETSTATIC_D extends Getstatic {
            public static final GETSTATIC_D INSTANCE = new GETSTATIC_D();
            private GETSTATIC_D() { }
            public String toString() { return "GETSTATIC_D"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_D(getDest(q).getRegister(), Reflection.getstatic_D(f));
            }
        }
        public static class GETSTATIC_A extends Getstatic {
            public static final GETSTATIC_A INSTANCE = new GETSTATIC_A();
            private GETSTATIC_A() { }
            public String toString() { return "GETSTATIC_A"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_A(getDest(q).getRegister(), Reflection.getstatic_A(f));
            }
        }
        public static class GETSTATIC_Z extends Getstatic {
            public static final GETSTATIC_Z INSTANCE = new GETSTATIC_Z();
            private GETSTATIC_Z() { }
            public String toString() { return "GETSTATIC_Z"; }
	    public void interpret(Quad q, State s) {
		jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), Reflection.getstatic_Z(f)?1:0);
            }
        }
        public static class GETSTATIC_B extends Getstatic {
            public static final GETSTATIC_B INSTANCE = new GETSTATIC_B();
            private GETSTATIC_B() { }
            public String toString() { return "GETSTATIC_B"; }
	    public void interpret(Quad q, State s) {
		jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getstatic_B(f));
            }
        }
        public static class GETSTATIC_C extends Getstatic {
            public static final GETSTATIC_C INSTANCE = new GETSTATIC_C();
            private GETSTATIC_C() { }
            public String toString() { return "GETSTATIC_C"; }
	    public void interpret(Quad q, State s) {
		jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getstatic_C(f));
            }
        }
        public static class GETSTATIC_S extends Getstatic {
            public static final GETSTATIC_S INSTANCE = new GETSTATIC_S();
            private GETSTATIC_S() { }
            public String toString() { return "GETSTATIC_S"; }
	    public void interpret(Quad q, State s) {
		jq_StaticField f = (jq_StaticField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getstatic_S(f));
            }
        }
        public static class GETSTATIC_I_DYNLINK extends GETSTATIC_I {
            public static final GETSTATIC_I_DYNLINK INSTANCE = new GETSTATIC_I_DYNLINK();
            private GETSTATIC_I_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_I%"; }
        }
        public static class GETSTATIC_F_DYNLINK extends GETSTATIC_F {
            public static final GETSTATIC_F_DYNLINK INSTANCE = new GETSTATIC_F_DYNLINK();
            private GETSTATIC_F_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_F%"; }
        }
        public static class GETSTATIC_L_DYNLINK extends GETSTATIC_L {
            public static final GETSTATIC_L_DYNLINK INSTANCE = new GETSTATIC_L_DYNLINK();
            private GETSTATIC_L_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_L%"; }
        }
        public static class GETSTATIC_D_DYNLINK extends GETSTATIC_D {
            public static final GETSTATIC_D_DYNLINK INSTANCE = new GETSTATIC_D_DYNLINK();
            private GETSTATIC_D_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_D%"; }
        }
        public static class GETSTATIC_A_DYNLINK extends GETSTATIC_A {
            public static final GETSTATIC_A_DYNLINK INSTANCE = new GETSTATIC_A_DYNLINK();
            private GETSTATIC_A_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_A%"; }
        }
        public static class GETSTATIC_Z_DYNLINK extends GETSTATIC_Z {
            public static final GETSTATIC_Z_DYNLINK INSTANCE = new GETSTATIC_Z_DYNLINK();
            private GETSTATIC_Z_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_Z%"; }
        }
        public static class GETSTATIC_B_DYNLINK extends GETSTATIC_B {
            public static final GETSTATIC_B_DYNLINK INSTANCE = new GETSTATIC_B_DYNLINK();
            private GETSTATIC_B_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_B%"; }
        }
        public static class GETSTATIC_C_DYNLINK extends GETSTATIC_C {
            public static final GETSTATIC_C_DYNLINK INSTANCE = new GETSTATIC_C_DYNLINK();
            private GETSTATIC_C_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_C%"; }
        }
        public static class GETSTATIC_S_DYNLINK extends GETSTATIC_S {
            public static final GETSTATIC_S_DYNLINK INSTANCE = new GETSTATIC_S_DYNLINK();
            private GETSTATIC_S_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETSTATIC_S%"; }
        }
    }
    
    public static abstract class Putstatic extends Operator {
        public static Quad create(int id, Putstatic operator, Operand src, FieldOperand field) {
            return new Quad(id, operator, src, field);
        }
        public static Operand getSrc(Quad q) { return q.getOp1(); }
        public static FieldOperand getField(Quad q) { return (FieldOperand)q.getOp2(); }
	public static void setSrc(Quad q, Operand o) { q.setOp1(o); }
	public static void setField(Quad q, FieldOperand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitPutstatic(q);
            qv.visitStaticField(q);
            qv.visitStore(q);
            super.accept(q, qv);
        }
        
        public static class PUTSTATIC_I extends Putstatic {
            public static final PUTSTATIC_I INSTANCE = new PUTSTATIC_I();
            private PUTSTATIC_I() { }
            public String toString() { return "PUTSTATIC_I"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		int i = getIntOpValue(getSrc(q), s);
                Reflection.putstatic_I(f, i);
            }
        }
        public static class PUTSTATIC_F extends Putstatic {
            public static final PUTSTATIC_F INSTANCE = new PUTSTATIC_F();
            private PUTSTATIC_F() { }
            public String toString() { return "PUTSTATIC_F"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		float i = getFloatOpValue(getSrc(q), s);
                Reflection.putstatic_F(f, i);
            }
        }
        public static class PUTSTATIC_L extends Putstatic {
            public static final PUTSTATIC_L INSTANCE = new PUTSTATIC_L();
            private PUTSTATIC_L() { }
            public String toString() { return "PUTSTATIC_L"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		long i = getLongOpValue(getSrc(q), s);
                Reflection.putstatic_L(f, i);
            }
        }
        public static class PUTSTATIC_D extends Putstatic {
            public static final PUTSTATIC_D INSTANCE = new PUTSTATIC_D();
            private PUTSTATIC_D() { }
            public String toString() { return "PUTSTATIC_D"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		double i = getDoubleOpValue(getSrc(q), s);
                Reflection.putstatic_D(f, i);
            }
        }
        public static class PUTSTATIC_A extends Putstatic {
            public static final PUTSTATIC_A INSTANCE = new PUTSTATIC_A();
            private PUTSTATIC_A() { }
            public String toString() { return "PUTSTATIC_A"; }
            public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		Object i = getObjectOpValue(getSrc(q), s);
                Reflection.putstatic_A(f, i);
            }
        }
        public static class PUTSTATIC_Z extends Putstatic {
            public static final PUTSTATIC_Z INSTANCE = new PUTSTATIC_Z();
            private PUTSTATIC_Z() { }
            public String toString() { return "PUTSTATIC_Z"; }
	    public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		int i = getIntOpValue(getSrc(q), s);
                Reflection.putstatic_Z(f, i!=0);
	    }
        }
        public static class PUTSTATIC_B extends Putstatic {
            public static final PUTSTATIC_B INSTANCE = new PUTSTATIC_B();
            private PUTSTATIC_B() { }
            public String toString() { return "PUTSTATIC_B"; }
	    public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		int i = getIntOpValue(getSrc(q), s);
                Reflection.putstatic_B(f, (byte)i);
	    }
        }
        public static class PUTSTATIC_S extends Putstatic {
            public static final PUTSTATIC_S INSTANCE = new PUTSTATIC_S();
            private PUTSTATIC_S() { }
            public String toString() { return "PUTSTATIC_S"; }
	    public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		int i = getIntOpValue(getSrc(q), s);
                Reflection.putstatic_S(f, (short)i);
	    }
        }
        public static class PUTSTATIC_C extends Putstatic {
            public static final PUTSTATIC_C INSTANCE = new PUTSTATIC_C();
            private PUTSTATIC_C() { }
            public String toString() { return "PUTSTATIC_C"; }
	    public void interpret(Quad q, State s) {
                jq_StaticField f = (jq_StaticField)getField(q).getField();
		int i = getIntOpValue(getSrc(q), s);
                Reflection.putstatic_C(f, (char)i);
	    }
        }
        public static class PUTSTATIC_I_DYNLINK extends PUTSTATIC_I {
            public static final PUTSTATIC_I_DYNLINK INSTANCE = new PUTSTATIC_I_DYNLINK();
            private PUTSTATIC_I_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_I%"; }
        }
        public static class PUTSTATIC_F_DYNLINK extends PUTSTATIC_F {
            public static final PUTSTATIC_F_DYNLINK INSTANCE = new PUTSTATIC_F_DYNLINK();
            private PUTSTATIC_F_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_F%"; }
        }
        public static class PUTSTATIC_L_DYNLINK extends PUTSTATIC_L {
            public static final PUTSTATIC_L_DYNLINK INSTANCE = new PUTSTATIC_L_DYNLINK();
            private PUTSTATIC_L_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_L%"; }
        }
        public static class PUTSTATIC_D_DYNLINK extends PUTSTATIC_D {
            public static final PUTSTATIC_D_DYNLINK INSTANCE = new PUTSTATIC_D_DYNLINK();
            private PUTSTATIC_D_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_D%"; }
        }
        public static class PUTSTATIC_A_DYNLINK extends PUTSTATIC_A {
            public static final PUTSTATIC_A_DYNLINK INSTANCE = new PUTSTATIC_A_DYNLINK();
            private PUTSTATIC_A_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_A%"; }
        }
        public static class PUTSTATIC_Z_DYNLINK extends PUTSTATIC_Z {
            public static final PUTSTATIC_Z_DYNLINK INSTANCE = new PUTSTATIC_Z_DYNLINK();
            private PUTSTATIC_Z_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_Z%"; }
        }
        public static class PUTSTATIC_B_DYNLINK extends PUTSTATIC_B {
            public static final PUTSTATIC_B_DYNLINK INSTANCE = new PUTSTATIC_B_DYNLINK();
            private PUTSTATIC_B_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_B%"; }
        }
        public static class PUTSTATIC_C_DYNLINK extends PUTSTATIC_C {
            public static final PUTSTATIC_C_DYNLINK INSTANCE = new PUTSTATIC_C_DYNLINK();
            private PUTSTATIC_C_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_C%"; }
        }
        public static class PUTSTATIC_S_DYNLINK extends PUTSTATIC_S {
            public static final PUTSTATIC_S_DYNLINK INSTANCE = new PUTSTATIC_S_DYNLINK();
            private PUTSTATIC_S_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTSTATIC_S%"; }
        }
    }

    public static abstract class Getfield extends Operator {
        public static Quad create(int id, Getfield operator, RegisterOperand dst, Operand base, FieldOperand field, Operand guard) {
            return new Quad(id, operator, dst, base, field, guard);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getBase(Quad q) { return q.getOp2(); }
        public static FieldOperand getField(Quad q) { return (FieldOperand)q.getOp3(); }
        public static Operand getGuard(Quad q) { return q.getOp4(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setBase(Quad q, Operand o) { q.setOp2(o); }
	public static void setField(Quad q, FieldOperand o) { q.setOp3(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp4(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg24(q); }
        public boolean hasSideEffects() { return false; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitGetfield(q);
            qv.visitInstanceField(q);
            qv.visitLoad(q);
            super.accept(q, qv);
        }
        
        public static class GETFIELD_I extends Getfield {
            public static final GETFIELD_I INSTANCE = new GETFIELD_I();
            private GETFIELD_I() { }
            public String toString() { return "GETFIELD_I"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), Reflection.getfield_I(o, f));
            }
        }
        public static class GETFIELD_F extends Getfield {
            public static final GETFIELD_F INSTANCE = new GETFIELD_F();
            private GETFIELD_F() { }
            public String toString() { return "GETFIELD_F"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_F(getDest(q).getRegister(), Reflection.getfield_F(o, f));
            }
        }
        public static class GETFIELD_L extends Getfield {
            public static final GETFIELD_L INSTANCE = new GETFIELD_L();
            private GETFIELD_L() { }
            public String toString() { return "GETFIELD_L"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_L(getDest(q).getRegister(), Reflection.getfield_L(o, f));
            }
        }
        public static class GETFIELD_D extends Getfield {
            public static final GETFIELD_D INSTANCE = new GETFIELD_D();
            private GETFIELD_D() { }
            public String toString() { return "GETFIELD_D"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_D(getDest(q).getRegister(), Reflection.getfield_D(o, f));
            }
        }
        public static class GETFIELD_A extends Getfield {
            public static final GETFIELD_A INSTANCE = new GETFIELD_A();
            private GETFIELD_A() { }
            public String toString() { return "GETFIELD_A"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_A(getDest(q).getRegister(), Reflection.getfield_A(o, f));
            }
        }
        public static class GETFIELD_B extends Getfield {
            public static final GETFIELD_B INSTANCE = new GETFIELD_B();
            private GETFIELD_B() { }
            public String toString() { return "GETFIELD_B"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getfield_B(o, f));
            }
        }
        public static class GETFIELD_C extends Getfield {
            public static final GETFIELD_C INSTANCE = new GETFIELD_C();
            private GETFIELD_C() { }
            public String toString() { return "GETFIELD_C"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getfield_C(o, f));
            }
        }
        public static class GETFIELD_S extends Getfield {
            public static final GETFIELD_S INSTANCE = new GETFIELD_S();
            private GETFIELD_S() { }
            public String toString() { return "GETFIELD_S"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), (int)Reflection.getfield_S(o, f));
            }
        }
        public static class GETFIELD_Z extends Getfield {
            public static final GETFIELD_Z INSTANCE = new GETFIELD_Z();
            private GETFIELD_Z() { }
            public String toString() { return "GETFIELD_Z"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                s.putReg_I(getDest(q).getRegister(), Reflection.getfield_Z(o, f)?1:0);
            }
        }
        public static class GETFIELD_I_DYNLINK extends GETFIELD_I {
            public static final GETFIELD_I_DYNLINK INSTANCE = new GETFIELD_I_DYNLINK();
            private GETFIELD_I_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_I%"; }
        }
        public static class GETFIELD_F_DYNLINK extends GETFIELD_F {
            public static final GETFIELD_F_DYNLINK INSTANCE = new GETFIELD_F_DYNLINK();
            private GETFIELD_F_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_F%"; }
        }
        public static class GETFIELD_L_DYNLINK extends GETFIELD_L {
            public static final GETFIELD_L_DYNLINK INSTANCE = new GETFIELD_L_DYNLINK();
            private GETFIELD_L_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_L%"; }
        }
        public static class GETFIELD_D_DYNLINK extends GETFIELD_D {
            public static final GETFIELD_D_DYNLINK INSTANCE = new GETFIELD_D_DYNLINK();
            private GETFIELD_D_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_D%"; }
        }
        public static class GETFIELD_A_DYNLINK extends GETFIELD_A {
            public static final GETFIELD_A_DYNLINK INSTANCE = new GETFIELD_A_DYNLINK();
            private GETFIELD_A_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_A%"; }
        }
        public static class GETFIELD_B_DYNLINK extends GETFIELD_B {
            public static final GETFIELD_B_DYNLINK INSTANCE = new GETFIELD_B_DYNLINK();
            private GETFIELD_B_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_B%"; }
        }
        public static class GETFIELD_C_DYNLINK extends GETFIELD_C {
            public static final GETFIELD_C_DYNLINK INSTANCE = new GETFIELD_C_DYNLINK();
            private GETFIELD_C_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_C%"; }
        }
        public static class GETFIELD_S_DYNLINK extends GETFIELD_S {
            public static final GETFIELD_S_DYNLINK INSTANCE = new GETFIELD_S_DYNLINK();
            private GETFIELD_S_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_S%"; }
        }
        public static class GETFIELD_Z_DYNLINK extends GETFIELD_Z {
            public static final GETFIELD_Z_DYNLINK INSTANCE = new GETFIELD_Z_DYNLINK();
            private GETFIELD_Z_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public boolean hasSideEffects() { return true; }
            public String toString() { return "GETFIELD_Z%"; }
        }
    }
    
    public static abstract class Putfield extends Operator {
        public static Quad create(int id, Putfield operator, Operand base, FieldOperand field, Operand src, Operand guard) {
            return new Quad(id, operator, base, field, src, guard);
        }
        public static Operand getBase(Quad q) { return q.getOp1(); }
        public static FieldOperand getField(Quad q) { return (FieldOperand)q.getOp2(); }
        public static Operand getSrc(Quad q) { return q.getOp3(); }
        public static Operand getGuard(Quad q) { return q.getOp4(); }
	public static void setBase(Quad q, Operand o) { q.setOp1(o); }
	public static void setField(Quad q, FieldOperand o) { q.setOp2(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp3(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp4(o); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1234(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitPutfield(q);
            qv.visitInstanceField(q);
            qv.visitStore(q);
            super.accept(q, qv);
        }
        
        public static class PUTFIELD_I extends Putfield {
            public static final PUTFIELD_I INSTANCE = new PUTFIELD_I();
            private PUTFIELD_I() { }
            public String toString() { return "PUTFIELD_I"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_I(o, f, getIntOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_F extends Putfield {
            public static final PUTFIELD_F INSTANCE = new PUTFIELD_F();
            private PUTFIELD_F() { }
            public String toString() { return "PUTFIELD_F"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_F(o, f, getFloatOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_L extends Putfield {
            public static final PUTFIELD_L INSTANCE = new PUTFIELD_L();
            private PUTFIELD_L() { }
            public String toString() { return "PUTFIELD_L"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_L(o, f, getLongOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_D extends Putfield {
            public static final PUTFIELD_D INSTANCE = new PUTFIELD_D();
            private PUTFIELD_D() { }
            public String toString() { return "PUTFIELD_D"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_D(o, f, getDoubleOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_A extends Putfield {
            public static final PUTFIELD_A INSTANCE = new PUTFIELD_A();
            private PUTFIELD_A() { }
            public String toString() { return "PUTFIELD_A"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_A(o, f, getObjectOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_B extends Putfield {
            public static final PUTFIELD_B INSTANCE = new PUTFIELD_B();
            private PUTFIELD_B() { }
            public String toString() { return "PUTFIELD_B"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_B(o, f, (byte)getIntOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_C extends Putfield {
            public static final PUTFIELD_C INSTANCE = new PUTFIELD_C();
            private PUTFIELD_C() { }
            public String toString() { return "PUTFIELD_C"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_C(o, f, (char)getIntOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_S extends Putfield {
            public static final PUTFIELD_S INSTANCE = new PUTFIELD_S();
            private PUTFIELD_S() { }
            public String toString() { return "PUTFIELD_S"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_S(o, f, (short)getIntOpValue(getSrc(q), s));
            }
        }
        public static class PUTFIELD_Z extends Putfield {
            public static final PUTFIELD_Z INSTANCE = new PUTFIELD_Z();
            private PUTFIELD_Z() { }
            public String toString() { return "PUTFIELD_Z"; }
            public void interpret(Quad q, State s) {
                Object o = getObjectOpValue(getBase(q), s);
                jq_InstanceField f = (jq_InstanceField)getField(q).getField();
                Reflection.putfield_Z(o, f, getIntOpValue(getSrc(q), s)!=0);
            }
        }
        public static class PUTFIELD_I_DYNLINK extends PUTFIELD_I {
            public static final PUTFIELD_I_DYNLINK INSTANCE = new PUTFIELD_I_DYNLINK();
            private PUTFIELD_I_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_I%"; }
        }
        public static class PUTFIELD_F_DYNLINK extends PUTFIELD_F {
            public static final PUTFIELD_F_DYNLINK INSTANCE = new PUTFIELD_F_DYNLINK();
            private PUTFIELD_F_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_F%"; }
        }
        public static class PUTFIELD_L_DYNLINK extends PUTFIELD_L {
            public static final PUTFIELD_L_DYNLINK INSTANCE = new PUTFIELD_L_DYNLINK();
            private PUTFIELD_L_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_L%"; }
        }
        public static class PUTFIELD_D_DYNLINK extends PUTFIELD_D {
            public static final PUTFIELD_D_DYNLINK INSTANCE = new PUTFIELD_D_DYNLINK();
            private PUTFIELD_D_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_D%"; }
        }
        public static class PUTFIELD_A_DYNLINK extends PUTFIELD_A {
            public static final PUTFIELD_A_DYNLINK INSTANCE = new PUTFIELD_A_DYNLINK();
            private PUTFIELD_A_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_A%"; }
        }
        public static class PUTFIELD_B_DYNLINK extends PUTFIELD_B {
            public static final PUTFIELD_B_DYNLINK INSTANCE = new PUTFIELD_B_DYNLINK();
            private PUTFIELD_B_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_B%"; }
        }
        public static class PUTFIELD_C_DYNLINK extends PUTFIELD_C {
            public static final PUTFIELD_C_DYNLINK INSTANCE = new PUTFIELD_C_DYNLINK();
            private PUTFIELD_C_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_C%"; }
        }
        public static class PUTFIELD_S_DYNLINK extends PUTFIELD_S {
            public static final PUTFIELD_S_DYNLINK INSTANCE = new PUTFIELD_S_DYNLINK();
            private PUTFIELD_S_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_S%"; }
        }
        public static class PUTFIELD_Z_DYNLINK extends PUTFIELD_Z {
            public static final PUTFIELD_Z_DYNLINK INSTANCE = new PUTFIELD_Z_DYNLINK();
            private PUTFIELD_Z_DYNLINK() { }
            public void accept(Quad q, QuadVisitor qv) {
                qv.visitExceptionThrower(q);
                super.accept(q, qv);
            }
            public UnmodifiableList.jq_Class getThrownExceptions() {
                return resolutionexceptions;
            }
            public String toString() { return "PUTFIELD_Z%"; }
        }
    }

    public static abstract class NullCheck extends Operator {
        public static Quad create(int id, NullCheck operator, Operand dst, Operand src) {
            return new Quad(id, operator, dst, src);
        }
        public static Operand getDest(Quad q) { return q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, Operand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1_check(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitNullCheck(q);
            qv.visitCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return nullptrexception;
        }
        
        public static class NULL_CHECK extends NullCheck {
            public static final NULL_CHECK INSTANCE = new NULL_CHECK();
            private NULL_CHECK() { }
            public String toString() { return "NULL_CHECK"; }
	    public void interpret(Quad q, State s) {
		if (getObjectOpValue(getSrc(q), s) == null) {
		    s.handleException(new NullPointerException(s.currentLocation()));
		}
	    }
        }
    }

    public static abstract class ZeroCheck extends Operator {
        public static Quad create(int id, ZeroCheck operator, Operand dst, Operand src) {
            return new Quad(id, operator, dst, src);
        }
        public static Operand getDest(Quad q) { return q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, Operand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1_check(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitZeroCheck(q);
            qv.visitCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return arithexception;
        }
        
        public static class ZERO_CHECK_I extends ZeroCheck {
            public static final ZERO_CHECK_I INSTANCE = new ZERO_CHECK_I();
            private ZERO_CHECK_I() { }
            public String toString() { return "ZERO_CHECK_I"; }
	    public void interpret(Quad q, State s) {
		if (getIntOpValue(getSrc(q), s) == 0) {
		    s.handleException(new ArithmeticException(s.currentLocation()));
		}
	    }
        }

        public static class ZERO_CHECK_L extends ZeroCheck {
            public static final ZERO_CHECK_L INSTANCE = new ZERO_CHECK_L();
            private ZERO_CHECK_L() { }
            public String toString() { return "ZERO_CHECK_L"; }
	    public void interpret(Quad q, State s) {
		if (getLongOpValue(getSrc(q), s) == 0L) {
		    s.handleException(new ArithmeticException(s.currentLocation()));
		}
	    }
        }
    }
    
    public static abstract class BoundsCheck extends Operator {
        public static Quad create(int id, BoundsCheck operator, Operand ref, Operand idx, Operand guard) {
            return new Quad(id, operator, ref, idx, guard);
        }
        public static Operand getRef(Quad q) { return q.getOp1(); }
        public static Operand getIndex(Quad q) { return q.getOp2(); }
        public static Operand getGuard(Quad q) { return q.getOp3(); }
	public static void setRef(Quad q, Operand o) { q.setOp1(o); }
	public static void setIndex(Quad q, Operand o) { q.setOp2(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg3(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg123(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitBoundsCheck(q);
            qv.visitCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return arrayboundsexception;
        }
        
        public static class BOUNDS_CHECK extends BoundsCheck {
            public static final BOUNDS_CHECK INSTANCE = new BOUNDS_CHECK();
            private BOUNDS_CHECK() { }
            public String toString() { return "BOUNDS_CHECK"; }
	    public void interpret(Quad q, State s) {
		int i = getIntOpValue(getIndex(q), s);
		Object o = getObjectOpValue(getRef(q), s);
		int length = Reflection.arraylength(o);
		if (i < 0 || i >= length) {
		    s.handleException(new ArrayIndexOutOfBoundsException(s.currentLocation()+" index: "+i+", length: "+length));
		}
	    }
        }
    }
    
    public static abstract class StoreCheck extends Operator {
        public static Quad create(int id, StoreCheck operator, Operand ref, Operand elem, Operand guard) {
            return new Quad(id, operator, ref, elem, guard);
        }
        public static Operand getRef(Quad q) { return q.getOp1(); }
        public static Operand getElement(Quad q) { return q.getOp2(); }
        public static Operand getGuard(Quad q) { return q.getOp3(); }
	public static void setRef(Quad q, Operand o) { q.setOp1(o); }
	public static void setElement(Quad q, Operand o) { q.setOp2(o); }
	public static void setGuard(Quad q, Operand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg3(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg123(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitStoreCheck(q);
            qv.visitTypeCheck(q);
            qv.visitCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return arraystoreexception;
        }
        
        public static class ASTORE_CHECK extends StoreCheck {
            public static final ASTORE_CHECK INSTANCE = new ASTORE_CHECK();
            private ASTORE_CHECK() { }
            public String toString() { return "ASTORE_CHECK"; }
	    public void interpret(Quad q, State s) {
		Object[] o = (Object[])getObjectOpValue(getRef(q), s);
		Object e = getObjectOpValue(getElement(q), s);
		if (e == null) return;
		jq_Reference t = Reflection.getTypeOf(e);
		t.load(); t.verify(); t.prepare(); t.sf_initialize(); t.cls_initialize();
		jq_Array a = (jq_Array)Reflection.getTypeOf(o);
		jq_Type t2 = a.getElementType();
		if (!TypeCheck.isAssignable(t, t2))
		    s.handleException(new ArrayStoreException(t+" into array "+a));
	    }
        }
    }
    
    public static abstract class Invoke extends Operator {
        public static Quad create(int id, Invoke operator, RegisterOperand res, MethodOperand m, int length) {
            return new Quad(id, operator, res, m, new ParamListOperand(new RegisterOperand[length]));
        }
        public static void setParam(Quad q, int i, RegisterOperand t) {
            ((ParamListOperand)q.getOp3()).set(i, t);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static MethodOperand getMethod(Quad q) { return (MethodOperand)q.getOp2(); }
        public static RegisterOperand getParam(Quad q, int i) { return ((ParamListOperand)q.getOp3()).get(i); }
        public static ParamListOperand getParamList(Quad q) { return (ParamListOperand)q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setMethod(Quad q, MethodOperand o) { q.setOp2(o); }
	public static void setParamList(Quad q, ParamListOperand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) {
            ParamListOperand plo = getParamList(q);
            RegisterOperand[] a = new RegisterOperand[plo.length()];
            for (int i=0; i<a.length; ++i) a[i] = plo.get(i);
            return new UnmodifiableList.RegisterOperand(a);
        }

        public abstract boolean isVirtual();
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitInvoke(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return anyexception;
        }
        public boolean hasSideEffects() { return true; }
        
	public void interpret_virtual(Quad q, State s) {
	    ParamListOperand plo = getParamList(q);
	    jq_Method f = getMethod(q).getMethod();
	    jq_Reference t = Reflection.getTypeOf(s.getReg_A(plo.get(0).getRegister()));
	    t.load(); t.verify(); t.prepare(); t.sf_initialize(); t.cls_initialize();
	    f = t.getVirtualMethod(f.getNameAndDesc());
	    if ((f == null) || f.isAbstract()) {
		s.handleException(new AbstractMethodError(s.currentLocation()));
		return;
	    }
	    State result = s.invokeMethod(f, plo);
	    if (result.getThrown() != null)
		s.handleException(result.getThrown());
	    else if (getDest(q) != null) {
		Object r = result.getReturnValue();
		if (f.getReturnType() == jq_Primitive.BOOLEAN && r instanceof Boolean) r = new Integer(((Boolean)r).booleanValue()?1:0);
		else if (f.getReturnType() == jq_Primitive.CHAR && r instanceof Character) r = new Integer(((Character)r).charValue());
		s.putReg(getDest(q).getRegister(), r);
	    }
	}

	public void interpret_static(Quad q, State s) {
	    ParamListOperand plo = getParamList(q);
	    jq_Method f = getMethod(q).getMethod();
	    State result = s.invokeMethod(f, plo);
	    if (result.getThrown() != null)
		s.handleException(result.getThrown());
	    else if (getDest(q) != null) {
		Object r = result.getReturnValue();
		if (f.getReturnType() == jq_Primitive.BOOLEAN && r instanceof Boolean) r = new Integer(((Boolean)r).booleanValue()?1:0);
		else if (f.getReturnType() == jq_Primitive.CHAR && r instanceof Character) r = new Integer(((Character)r).charValue());
		s.putReg(getDest(q).getRegister(), r);
	    }
	}

        public static abstract class InvokeVirtual extends Invoke {
            public boolean isVirtual() { return true; }
	    public void interpret(Quad q, State s) { interpret_virtual(q, s); }
        }
        public static abstract class InvokeStatic extends Invoke {
            public boolean isVirtual() { return false; }
	    public void interpret(Quad q, State s) { interpret_static(q, s); }
        }
        public static abstract class InvokeInterface extends Invoke {
            public boolean isVirtual() { return true; }
	    public void interpret(Quad q, State s) { interpret_virtual(q, s); }
        }
        public static class INVOKEVIRTUAL_V extends InvokeVirtual {
            public static final INVOKEVIRTUAL_V INSTANCE = new INVOKEVIRTUAL_V();
            private INVOKEVIRTUAL_V() { }
            public String toString() { return "INVOKEVIRTUAL_V"; }
        }
        public static class INVOKEVIRTUAL_I extends InvokeVirtual {
            public static final INVOKEVIRTUAL_I INSTANCE = new INVOKEVIRTUAL_I();
            private INVOKEVIRTUAL_I() { }
            public String toString() { return "INVOKEVIRTUAL_I"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEVIRTUAL_F extends InvokeVirtual {
            public static final INVOKEVIRTUAL_F INSTANCE = new INVOKEVIRTUAL_F();
            private INVOKEVIRTUAL_F() { }
            public String toString() { return "INVOKEVIRTUAL_F"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEVIRTUAL_L extends InvokeVirtual {
            public static final INVOKEVIRTUAL_L INSTANCE = new INVOKEVIRTUAL_L();
            private INVOKEVIRTUAL_L() { }
            public String toString() { return "INVOKEVIRTUAL_L"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEVIRTUAL_D extends InvokeVirtual {
            public static final INVOKEVIRTUAL_D INSTANCE = new INVOKEVIRTUAL_D();
            private INVOKEVIRTUAL_D() { }
            public String toString() { return "INVOKEVIRTUAL_D"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEVIRTUAL_A extends InvokeVirtual {
            public static final INVOKEVIRTUAL_A INSTANCE = new INVOKEVIRTUAL_A();
            private INVOKEVIRTUAL_A() { }
            public String toString() { return "INVOKEVIRTUAL_A"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKESTATIC_V extends InvokeStatic {
            public static final INVOKESTATIC_V INSTANCE = new INVOKESTATIC_V();
            private INVOKESTATIC_V() { }
            public String toString() { return "INVOKESTATIC_V"; }
        }
        public static class INVOKESTATIC_I extends InvokeStatic {
            public static final INVOKESTATIC_I INSTANCE = new INVOKESTATIC_I();
            private INVOKESTATIC_I() { }
            public String toString() { return "INVOKESTATIC_I"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKESTATIC_F extends InvokeStatic {
            public static final INVOKESTATIC_F INSTANCE = new INVOKESTATIC_F();
            private INVOKESTATIC_F() { }
            public String toString() { return "INVOKESTATIC_F"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKESTATIC_L extends InvokeStatic {
            public static final INVOKESTATIC_L INSTANCE = new INVOKESTATIC_L();
            private INVOKESTATIC_L() { }
            public String toString() { return "INVOKESTATIC_L"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKESTATIC_D extends InvokeStatic {
            public static final INVOKESTATIC_D INSTANCE = new INVOKESTATIC_D();
            private INVOKESTATIC_D() { }
            public String toString() { return "INVOKESTATIC_D"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKESTATIC_A extends InvokeStatic {
            public static final INVOKESTATIC_A INSTANCE = new INVOKESTATIC_A();
            private INVOKESTATIC_A() { }
            public String toString() { return "INVOKESTATIC_A"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEVIRTUAL_V_DYNLINK extends INVOKEVIRTUAL_V {
            public static final INVOKEVIRTUAL_V_DYNLINK INSTANCE = new INVOKEVIRTUAL_V_DYNLINK();
            private INVOKEVIRTUAL_V_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_V%"; }
        }
        public static class INVOKEVIRTUAL_I_DYNLINK extends INVOKEVIRTUAL_I {
            public static final INVOKEVIRTUAL_I_DYNLINK INSTANCE = new INVOKEVIRTUAL_I_DYNLINK();
            private INVOKEVIRTUAL_I_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_I%"; }
        }
        public static class INVOKEVIRTUAL_F_DYNLINK extends INVOKEVIRTUAL_F {
            public static final INVOKEVIRTUAL_F_DYNLINK INSTANCE = new INVOKEVIRTUAL_F_DYNLINK();
            private INVOKEVIRTUAL_F_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_F%"; }
        }
        public static class INVOKEVIRTUAL_L_DYNLINK extends INVOKEVIRTUAL_L {
            public static final INVOKEVIRTUAL_L_DYNLINK INSTANCE = new INVOKEVIRTUAL_L_DYNLINK();
            private INVOKEVIRTUAL_L_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_L%"; }
        }
        public static class INVOKEVIRTUAL_D_DYNLINK extends INVOKEVIRTUAL_D {
            public static final INVOKEVIRTUAL_D_DYNLINK INSTANCE = new INVOKEVIRTUAL_D_DYNLINK();
            private INVOKEVIRTUAL_D_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_D%"; }
        }
        public static class INVOKEVIRTUAL_A_DYNLINK extends INVOKEVIRTUAL_A {
            public static final INVOKEVIRTUAL_A_DYNLINK INSTANCE = new INVOKEVIRTUAL_A_DYNLINK();
            private INVOKEVIRTUAL_A_DYNLINK() { }
            public String toString() { return "INVOKEVIRTUAL_A%"; }
        }
        public static class INVOKESTATIC_V_DYNLINK extends INVOKESTATIC_V {
            public static final INVOKESTATIC_V_DYNLINK INSTANCE = new INVOKESTATIC_V_DYNLINK();
            private INVOKESTATIC_V_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_V%"; }
        }
        public static class INVOKESTATIC_I_DYNLINK extends INVOKESTATIC_I {
            public static final INVOKESTATIC_I_DYNLINK INSTANCE = new INVOKESTATIC_I_DYNLINK();
            private INVOKESTATIC_I_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_I%"; }
        }
        public static class INVOKESTATIC_F_DYNLINK extends INVOKESTATIC_F {
            public static final INVOKESTATIC_F_DYNLINK INSTANCE = new INVOKESTATIC_F_DYNLINK();
            private INVOKESTATIC_F_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_F%"; }
        }
        public static class INVOKESTATIC_L_DYNLINK extends INVOKESTATIC_L {
            public static final INVOKESTATIC_L_DYNLINK INSTANCE = new INVOKESTATIC_L_DYNLINK();
            private INVOKESTATIC_L_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_L%"; }
        }
        public static class INVOKESTATIC_D_DYNLINK extends INVOKESTATIC_D {
            public static final INVOKESTATIC_D_DYNLINK INSTANCE = new INVOKESTATIC_D_DYNLINK();
            private INVOKESTATIC_D_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_D%"; }
        }
        public static class INVOKESTATIC_A_DYNLINK extends INVOKESTATIC_A {
            public static final INVOKESTATIC_A_DYNLINK INSTANCE = new INVOKESTATIC_A_DYNLINK();
            private INVOKESTATIC_A_DYNLINK() { }
            public String toString() { return "INVOKESTATIC_A%"; }
        }
        public static class INVOKESPECIAL_V_DYNLINK extends INVOKESTATIC_V {
            public static final INVOKESPECIAL_V_DYNLINK INSTANCE = new INVOKESPECIAL_V_DYNLINK();
            private INVOKESPECIAL_V_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_V%"; }
        }
        public static class INVOKESPECIAL_I_DYNLINK extends INVOKESTATIC_I {
            public static final INVOKESPECIAL_I_DYNLINK INSTANCE = new INVOKESPECIAL_I_DYNLINK();
            private INVOKESPECIAL_I_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_I%"; }
        }
        public static class INVOKESPECIAL_F_DYNLINK extends INVOKESTATIC_F {
            public static final INVOKESPECIAL_F_DYNLINK INSTANCE = new INVOKESPECIAL_F_DYNLINK();
            private INVOKESPECIAL_F_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_F%"; }
        }
        public static class INVOKESPECIAL_L_DYNLINK extends INVOKESTATIC_L {
            public static final INVOKESPECIAL_L_DYNLINK INSTANCE = new INVOKESPECIAL_L_DYNLINK();
            private INVOKESPECIAL_L_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_L%"; }
        }
        public static class INVOKESPECIAL_D_DYNLINK extends INVOKESTATIC_D {
            public static final INVOKESPECIAL_D_DYNLINK INSTANCE = new INVOKESPECIAL_D_DYNLINK();
            private INVOKESPECIAL_D_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_D%"; }
        }
        public static class INVOKESPECIAL_A_DYNLINK extends INVOKESTATIC_A {
            public static final INVOKESPECIAL_A_DYNLINK INSTANCE = new INVOKESPECIAL_A_DYNLINK();
            private INVOKESPECIAL_A_DYNLINK() { }
            public String toString() { return "INVOKESPECIAL_A%"; }
        }
        public static class INVOKEINTERFACE_V extends InvokeInterface {
            public static final INVOKEINTERFACE_V INSTANCE = new INVOKEINTERFACE_V();
            private INVOKEINTERFACE_V() { }
            public String toString() { return "INVOKEINTERFACE_V"; }
        }
        public static class INVOKEINTERFACE_I extends InvokeInterface {
            public static final INVOKEINTERFACE_I INSTANCE = new INVOKEINTERFACE_I();
            private INVOKEINTERFACE_I() { }
            public String toString() { return "INVOKEINTERFACE_I"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEINTERFACE_F extends InvokeInterface {
            public static final INVOKEINTERFACE_F INSTANCE = new INVOKEINTERFACE_F();
            private INVOKEINTERFACE_F() { }
            public String toString() { return "INVOKEINTERFACE_F"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEINTERFACE_L extends InvokeInterface {
            public static final INVOKEINTERFACE_L INSTANCE = new INVOKEINTERFACE_L();
            private INVOKEINTERFACE_L() { }
            public String toString() { return "INVOKEINTERFACE_L"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEINTERFACE_D extends InvokeInterface {
            public static final INVOKEINTERFACE_D INSTANCE = new INVOKEINTERFACE_D();
            private INVOKEINTERFACE_D() { }
            public String toString() { return "INVOKEINTERFACE_D"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
        public static class INVOKEINTERFACE_A extends InvokeInterface {
            public static final INVOKEINTERFACE_A INSTANCE = new INVOKEINTERFACE_A();
            private INVOKEINTERFACE_A() { }
            public String toString() { return "INVOKEINTERFACE_A"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        }
    }
    
    public static abstract class New extends Operator {
        public static Quad create(int id, New operator, RegisterOperand res, TypeOperand type) {
            return new Quad(id, operator, res, type);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static TypeOperand getType(Quad q) { return (TypeOperand)q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setType(Quad q, TypeOperand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitNew(q);
            qv.visitAllocation(q);
            super.accept(q, qv);
        }
        
        public static class NEW extends New {
            public static final NEW INSTANCE = new NEW();
            private NEW() { }
            public String toString() { return "NEW"; }
	    public void interpret(Quad q, State s) {
		s.putReg_A(getDest(q).getRegister(), new UninitializedReference((jq_Class)getType(q).getType()));
	    }
        }
        public static class NEW_DYNLINK extends NEW {
            public static final NEW_DYNLINK INSTANCE = new NEW_DYNLINK();
            private NEW_DYNLINK() { }
            public String toString() { return "NEW%"; }
        }
    }
    
    public static abstract class NewArray extends Operator {
        public static Quad create(int id, NewArray operator, RegisterOperand res, Operand size, TypeOperand type) {
            return new Quad(id, operator, res, size, type);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSize(Quad q) { return q.getOp2(); }
        public static TypeOperand getType(Quad q) { return (TypeOperand)q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSize(Quad q, Operand o) { q.setOp2(o); }
	public static void setType(Quad q, TypeOperand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitNewArray(q);
            qv.visitAllocation(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return negativesizeexception;
        }
        
        public static class NEWARRAY extends NewArray {
            public static final NEWARRAY INSTANCE = new NEWARRAY();
            private NEWARRAY() { }
            public String toString() { return "NEWARRAY"; }
	    public void interpret(Quad q, State s) {
		jq_Type t = getType(q).getType();
		int v = getIntOpValue(getSize(q), s);
                Object o = java.lang.reflect.Array.newInstance(Reflection.getJDKType(((jq_Array)t).getElementType()), v);
		s.putReg_A(getDest(q).getRegister(), o);
	    }
        }
    }
    
    public static abstract class CheckCast extends Operator {
        public static Quad create(int id, CheckCast operator, RegisterOperand res, Operand val, TypeOperand type) {
            return new Quad(id, operator, res, val, type);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
        public static TypeOperand getType(Quad q) { return (TypeOperand)q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
	public static void setType(Quad q, TypeOperand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitCheckCast(q);
            qv.visitTypeCheck(q);
            qv.visitCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return classcastexceptions;
        }
        
        public static class CHECKCAST extends CheckCast {
            public static final CHECKCAST INSTANCE = new CHECKCAST();
            private CHECKCAST() { }
            public String toString() { return "CHECKCAST"; }
	    public void interpret(Quad q, State s) {
		jq_Type t = getType(q).getType();
		Object o = getObjectOpValue(getSrc(q), s);
		if (o != null) {
		    jq_Type t2 = Reflection.getTypeOf(o);
		    t2.load(); t2.verify(); t2.prepare(); t2.sf_initialize(); t2.cls_initialize();
		    if (!TypeCheck.isAssignable(t2, t)) {
			s.handleException(new ClassCastException(t2+" cannot be cast into "+t));
			return;
		    }
		}
		s.putReg_A(getDest(q).getRegister(), o);
	    }
        }
    }
    
    public static abstract class InstanceOf extends Operator {
        public static Quad create(int id, InstanceOf operator, RegisterOperand res, Operand val, TypeOperand type) {
            return new Quad(id, operator, res, val, type);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
        public static TypeOperand getType(Quad q) { return (TypeOperand)q.getOp3(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
	public static void setType(Quad q, TypeOperand o) { q.setOp3(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitInstanceOf(q);
            qv.visitTypeCheck(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        public UnmodifiableList.jq_Class getThrownExceptions() {
            return resolutionexceptions;
        }
        
        public static class INSTANCEOF extends InstanceOf {
            public static final INSTANCEOF INSTANCE = new INSTANCEOF();
            private INSTANCEOF() { }
            public String toString() { return "INSTANCEOF"; }
	    public void interpret(Quad q, State s) {
		jq_Type t = getType(q).getType();
		Object o = getObjectOpValue(getSrc(q), s);
                int v;
                if (o == null) v = 0;
                else v = Reflection.getJDKType(t).isAssignableFrom(o.getClass())?1:0;
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
    }
    
    public abstract static class ALength extends Operator {
        public static Quad create(int id, ALength operator, RegisterOperand res, Operand val) {
            return new Quad(id, operator, res, val);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return false; }

        public void accept(Quad q, QuadVisitor qv) {
            qv.visitALength(q);
            qv.visitArray(q);
            super.accept(q, qv);
        }
        
        public static class ARRAYLENGTH extends ALength {
            public static final ARRAYLENGTH INSTANCE = new ARRAYLENGTH();
            private ARRAYLENGTH() { }
            public String toString() { return "ARRAYLENGTH"; }
	    public void interpret(Quad q, State s) {
		Object o = getObjectOpValue(getSrc(q), s);
		int v = Reflection.arraylength(o);
		s.putReg_I(getDest(q).getRegister(), v);
	    }
        }
    }
    
    public static abstract class Monitor extends Operator {
        public static Quad create(int id, Monitor operator, Operand val) {
            return new Quad(id, operator, null, val);
        }
        public static Operand getSrc(Quad q) { return q.getOp2(); }
	public static void setSrc(Quad q, Operand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitMonitor(q);
            qv.visitExceptionThrower(q);
            super.accept(q, qv);
        }
        
        public static class MONITORENTER extends Monitor {
            public static final MONITORENTER INSTANCE = new MONITORENTER();
            private MONITORENTER() { }
            public String toString() { return "MONITORENTER"; }
	    public void interpret(Quad q, State s) {
		Object o = getObjectOpValue(getSrc(q), s);
                if (!jq.Bootstrapping)
                    Run_Time.Monitor.monitorenter(o);
	    }
	    public UnmodifiableList.jq_Class getThrownExceptions() {
		return nullptrexception;
	    }
        }
        public static class MONITOREXIT extends Monitor {
            public static final MONITOREXIT INSTANCE = new MONITOREXIT();
            private MONITOREXIT() { }
            public String toString() { return "MONITOREXIT"; }
	    public void interpret(Quad q, State s) {
		Object o = getObjectOpValue(getSrc(q), s);
                if (!jq.Bootstrapping)
                    Run_Time.Monitor.monitorexit(o);
	    }
	    public UnmodifiableList.jq_Class getThrownExceptions() {
		return illegalmonitorstateexception;
	    }
        }
    }
    
    public static abstract class MemLoad extends Operator {
        public static Quad create(int id, MemLoad operator, RegisterOperand dst, Operand addr) {
            return new Quad(id, operator, dst, addr);
        }
        public static RegisterOperand getDest(Quad q) { return (RegisterOperand)q.getOp1(); }
        public static Operand getAddress(Quad q) { return q.getOp2(); }
	public static void setDest(Quad q, RegisterOperand o) { q.setOp1(o); }
	public static void setAddress(Quad q, Operand o) { q.setOp2(o); }
        public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
        public boolean hasSideEffects() { return false; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitMemLoad(q);
            qv.visitLoad(q);
            super.accept(q, qv);
        }
        
        public static class PEEK_1 extends MemLoad {
            public static final PEEK_1 INSTANCE = new PEEK_1();
            private PEEK_1() { }
            public String toString() { return "PEEK_1"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
                if (!jq.Bootstrapping)
                    s.putReg_I(getDest(q).getRegister(), (byte)Unsafe.peek(o));
	    }
        }
        public static class PEEK_2 extends MemLoad {
            public static final PEEK_2 INSTANCE = new PEEK_2();
            private PEEK_2() { }
            public String toString() { return "PEEK_2"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
                if (!jq.Bootstrapping)
                    s.putReg_I(getDest(q).getRegister(), (short)Unsafe.peek(o));
	    }
        }
        public static class PEEK_4 extends MemLoad {
            public static final PEEK_4 INSTANCE = new PEEK_4();
            private PEEK_4() { }
            public String toString() { return "PEEK_4"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
                if (!jq.Bootstrapping)
                    s.putReg_I(getDest(q).getRegister(), (int)Unsafe.peek(o));
	    }
        }
    }

    public static abstract class MemStore extends Operator {
        public static Quad create(int id, MemStore operator, Operand addr, Operand val) {
            return new Quad(id, operator, null, addr, val);
        }
        public static Operand getAddress(Quad q) { return q.getOp1(); }
        public static Operand getValue(Quad q) { return q.getOp2(); }
	public static void setAddress(Quad q, Operand o) { q.setOp1(o); }
	public static void setValue(Quad q, Operand o) { q.setOp2(o); }
        public boolean hasSideEffects() { return true; }
        public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg12(q); }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitMemStore(q);
            qv.visitStore(q);
            super.accept(q, qv);
        }
        
        public static class POKE_1 extends MemStore {
            public static final POKE_1 INSTANCE = new POKE_1();
            private POKE_1() { }
            public String toString() { return "POKE_1"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
		byte v = (byte)getIntOpValue(getValue(q), s);
                if (!jq.Bootstrapping)
                    Unsafe.poke1(o, v);
	    }
        }
        public static class POKE_2 extends MemStore {
            public static final POKE_2 INSTANCE = new POKE_2();
            private POKE_2() { }
            public String toString() { return "POKE_2"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
		short v = (short)getIntOpValue(getValue(q), s);
                if (!jq.Bootstrapping)
                    Unsafe.poke2(o, v);
	    }
        }
        public static class POKE_4 extends MemStore {
            public static final POKE_4 INSTANCE = new POKE_4();
            private POKE_4() { }
            public String toString() { return "POKE_4"; }
	    public void interpret(Quad q, State s) {
		int o = getIntOpValue(getAddress(q), s);
		int v = (int)getIntOpValue(getValue(q), s);
                if (!jq.Bootstrapping)
                    Unsafe.poke4(o, v);
	    }
        }
    }
    
    public static abstract class Special extends Operator {
        
        public static Quad create(int id, GET_EXCEPTION operator, RegisterOperand res) {
            return new Quad(id, operator, res);
        }
        public static Quad create(int id, GET_THREAD_BLOCK operator, RegisterOperand res) {
            return new Quad(id, operator, res);
        }
        public static Quad create(int id, GET_TYPE_OF operator, RegisterOperand res, Operand src) {
            return new Quad(id, operator, res, src);
        }
        public static Quad create(int id, SET_THREAD_BLOCK operator, Operand val) {
            return new Quad(id, operator, null, val);
        }
        public static Quad create(int id, ALLOCA operator, RegisterOperand res, Operand val) {
            return new Quad(id, operator, res, val);
        }
        public static Quad create(int id, LONG_JUMP operator, Operand ip, Operand fp, Operand sp, Operand eax) {
            return new Quad(id, operator, ip, fp, sp, eax);
        }
        public static Quad create(int id, DIE operator, Operand val) {
            return new Quad(id, operator, val);
        }
        public static Operand getOp1(Quad q) { return q.getOp1(); }
        public static Operand getOp2(Quad q) { return q.getOp2(); }
        public static Operand getOp3(Quad q) { return q.getOp3(); }
        public static Operand getOp4(Quad q) { return q.getOp4(); }
	public static void setOp1(Quad q, Operand o) { q.setOp1(o); }
	public static void setOp2(Quad q, Operand o) { q.setOp2(o); }
	public static void setOp3(Quad q, Operand o) { q.setOp3(o); }
	public static void setOp4(Quad q, Operand o) { q.setOp4(o); }
        public boolean hasSideEffects() { return true; }
        
        public void accept(Quad q, QuadVisitor qv) {
            qv.visitSpecial(q);
            super.accept(q, qv);
        }
        
        public static class GET_EXCEPTION extends Special {
            public static final GET_EXCEPTION INSTANCE = new GET_EXCEPTION();
            private GET_EXCEPTION() { }
            public String toString() { return "GET_EXCEPTION"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
	    public void interpret(Quad q, State s) {
		s.putReg_A(((RegisterOperand)getOp1(q)).getRegister(), s.getCaught());
	    }
        }
        public static class GET_THREAD_BLOCK extends Special {
            public static final GET_THREAD_BLOCK INSTANCE = new GET_THREAD_BLOCK();
            private GET_THREAD_BLOCK() { }
            public String toString() { return "GET_THREAD_BLOCK"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
	    public void interpret(Quad q, State s) {
                if (!jq.Bootstrapping)
                    s.putReg_A(((RegisterOperand)getOp1(q)).getRegister(), Unsafe.getThreadBlock());
	    }
        }
        public static class SET_THREAD_BLOCK extends Special {
            public static final SET_THREAD_BLOCK INSTANCE = new SET_THREAD_BLOCK();
            private SET_THREAD_BLOCK() { }
            public String toString() { return "SET_THREAD_BLOCK"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
	    public void interpret(Quad q, State s) {
		Scheduler.jq_Thread o = (Scheduler.jq_Thread)getObjectOpValue(getOp2(q), s);
                if (!jq.Bootstrapping)
                    Unsafe.setThreadBlock(o);
	    }
        }
        public static class ALLOCA extends Special {
            public static final ALLOCA INSTANCE = new ALLOCA();
            private ALLOCA() { }
            public String toString() { return "ALLOCA"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
	    public void interpret(Quad q, State s) {
		// TODO: skip for now.
	    }
        }
        public static class LONG_JUMP extends Special {
            public static final LONG_JUMP INSTANCE = new LONG_JUMP();
            private LONG_JUMP() { }
            public String toString() { return "LONG_JUMP"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1234(q); }
	    public void interpret(Quad q, State s) {
		int a = getIntOpValue(getOp1(q), s);
		int b = getIntOpValue(getOp2(q), s);
		int c = getIntOpValue(getOp3(q), s);
		int d = getIntOpValue(getOp4(q), s);
                Unsafe.longJump(a, b, c, d);
                jq.UNREACHABLE();
	    }
        }
        public static class DIE extends Special {
            public static final DIE INSTANCE = new DIE();
            private DIE() { }
            public String toString() { return "DIE"; }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg1_check(q); }
	    public void interpret(Quad q, State s) {
		int a = getIntOpValue(getOp1(q), s);
                Run_Time.SystemInterface.die(a);
                jq.UNREACHABLE();
	    }
        }
        public static class GET_TYPE_OF extends Special {
            public static final GET_TYPE_OF INSTANCE = new GET_TYPE_OF();
            private GET_TYPE_OF() { }
            public String toString() { return "GET_TYPE_OF"; }
            public UnmodifiableList.RegisterOperand getDefinedRegisters(Quad q) { return getReg1(q); }
            public UnmodifiableList.RegisterOperand getUsedRegisters(Quad q) { return getReg2(q); }
	    public void interpret(Quad q, State s) {
		Object o = getObjectOpValue(getOp2(q), s);
		s.putReg_A(((RegisterOperand)getOp1(q)).getRegister(), Reflection.getTypeOf(o));
	    }
        }
    }
}

/*
 * DirectInterpreter.java
 *
 * Created on January 16, 2001, 1:06 AM
 * 
 */

package Interpreter;

import Allocator.*;
import Clazz.*;
import Run_Time.*;
import Compil3r.Reference.x86.x86ReferenceLinker;
import Main.jq;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class DirectInterpreter extends BytecodeInterpreter implements ObjectLayout {

    /** Creates new DirectInterpreter */
    public DirectInterpreter(State initialState) {
        super(new DirectVMInterface(), initialState);
    }

    public Object invokeMethod(jq_Method m) throws Throwable {
        jq_Class k = m.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int localsize = m.getMaxLocals()<<2;
        int stacksize = m.getMaxStack()<<2;
        int/*StackAddress*/ newframe = Unsafe.alloca(localsize+stacksize);
        DirectState callee = new DirectState(newframe, newframe+localsize, m.getMaxLocals());
        return this.invokeMethod(m, callee);
    }
    
    public Object invokeUnsafeMethod(jq_StaticMethod f) throws Throwable {
        if ((f == Unsafe._addressOf) || (f == Unsafe._floatToIntBits)) {
            return new Integer(state.pop_I());
        } else if (f == Unsafe._asObject) {
            return state.pop_A();
        } else if (f == Unsafe._intBitsToFloat) {
            return new Float(state.pop_F());
        } else if (f == Unsafe._doubleToLongBits) {
            return new Long(state.pop_L());
        } else if (f == Unsafe._longBitsToDouble) {
            return new Double(state.pop_D());
        } else if (f == Unsafe._peek) {
            return new Integer(Unsafe.peek(state.pop_I()));
        } else if (f == Unsafe._poke1) {
            byte val = (byte)state.pop_I();
            int/*Address*/ addr = state.pop_I();
            Unsafe.poke1(addr, val);
            return null;
        } else if (f == Unsafe._poke2) {
            short val = (short)state.pop_I();
            int/*Address*/ addr = state.pop_I();
            Unsafe.poke2(addr, val);
            return null;
        } else if (f == Unsafe._poke4) {
            int val = state.pop_I();
            int/*Address*/ addr = state.pop_I();
            Unsafe.poke4(addr, val);
            return null;
        } else if (f == Unsafe._getTypeOf) {
            return Unsafe.getTypeOf(state.pop_A());
        } else if (f == Unsafe._invoke) {
            return new Double(Unsafe.invoke(state.pop_I()));
        } else if (f == Unsafe._getThreadBlock) {
            return Unsafe.getThreadBlock();
        } else {
            System.err.println(f.toString());
            jq.UNREACHABLE();
            return null;
        }
    }
        
    public static class DirectState extends BytecodeInterpreter.State {
        final int/*StackAddress*/ fp;
        final int nlocals;
        int/*StackAddress*/ sp;
        int loResult, hiResult;
        
        public DirectState(int/*StackAddress*/ fp, int/*StackAddress*/ sp, int nlocals) {
            this.fp = fp; this.sp = sp;
            this.nlocals = nlocals;
        }

        public void push_I(int v) { Unsafe.poke4(sp-=4, v); }
        public void push_L(long v) { push_I((int)(v>>32)); push_I((int)v); } // hi, lo
        public void push_F(float v) { push_I(Float.floatToRawIntBits(v)); }
        public void push_D(double v) { push_L(Double.doubleToRawLongBits(v)); }
        public void push_A(Object v) { push_I(Unsafe.addressOf(v)); }
        public void push(Object v) { push_I(Unsafe.addressOf(v)); }
        public int pop_I() { int v=Unsafe.peek(sp); sp+=4; return v; }
        public long pop_L() { int lo=pop_I(); int hi=pop_I(); return jq.twoIntsToLong(lo, hi); } // lo, hi
        public float pop_F() { return Float.intBitsToFloat(pop_I()); }
        public double pop_D() { return Double.longBitsToDouble(pop_L()); }
        public Object pop_A() { return Unsafe.asObject(pop_I()); }
        public Object pop() { return Unsafe.asObject(pop_I()); }
        public void popAll() { sp=fp-(nlocals<<2); }
        public Object peek_A(int depth) { return Unsafe.asObject(sp+(depth<<2)); }
        public void setLocal_I(int i, int v) { Unsafe.poke4(fp-(i<<2), v); }
        public void setLocal_L(int i, long v) { setLocal_I(i, (int)(v>>32)); setLocal_I(i+1, (int)v); } // hi, lo
        public void setLocal_F(int i, float v) { setLocal_I(i, Float.floatToRawIntBits(v)); }
        public void setLocal_D(int i, double v) { setLocal_L(i, Double.doubleToRawLongBits(v)); }
        public void setLocal_A(int i, Object v) { setLocal_I(i, Unsafe.addressOf(v)); }
        public int getLocal_I(int i) { return Unsafe.peek(fp-(i<<2)); }
        public long getLocal_L(int i) { int lo=getLocal_I(i+1); int hi=getLocal_I(i); return jq.twoIntsToLong(lo, hi); } // lo, hi
        public float getLocal_F(int i) { return Float.intBitsToFloat(getLocal_I(i)); }
        public double getLocal_D(int i) { return Double.longBitsToDouble(getLocal_L(i)); }
        public Object getLocal_A(int i) { return Unsafe.asObject(getLocal_I(i)); }
        public void return_I(int v) { loResult = v; }
        public void return_L(long v) { loResult = (int)(v>>32); hiResult = (int)v; }
        public void return_F(float v) { loResult = Float.floatToRawIntBits(v); }
        public void return_D(double v) { return_L(Double.doubleToRawLongBits(v)); }
        public void return_A(Object v) { loResult = Unsafe.addressOf(v); }
        public void return_V() {}
        public int getReturnVal_I() { return loResult; }
        public long getReturnVal_L() { return jq.twoIntsToLong(loResult, hiResult); }
        public float getReturnVal_F() { return Float.intBitsToFloat(loResult); }
        public double getReturnVal_D() { return Double.longBitsToDouble(getReturnVal_L()); }
        public Object getReturnVal_A() { return Unsafe.asObject(loResult); }
    }
    
    public static class DirectVMInterface extends BytecodeInterpreter.VMInterface {
        public int getstatic_I(jq_StaticField f) { return Unsafe.peek(f.getAddress()); }
        public long getstatic_L(jq_StaticField f) { int lo=Unsafe.peek(f.getAddress()); int hi=Unsafe.peek(f.getAddress()+4); return jq.twoIntsToLong(lo, hi); }
        public float getstatic_F(jq_StaticField f) { return Float.intBitsToFloat(getstatic_I(f)); }
        public double getstatic_D(jq_StaticField f) { return Double.longBitsToDouble(getstatic_L(f)); }
        public Object getstatic_A(jq_StaticField f) { return Unsafe.asObject(Unsafe.peek(f.getAddress())); }
        public byte getstatic_B(jq_StaticField f) { return (byte)Unsafe.peek(f.getAddress()); }
        public char getstatic_C(jq_StaticField f) { return (char)Unsafe.peek(f.getAddress()); }
        public short getstatic_S(jq_StaticField f) { return (short)Unsafe.peek(f.getAddress()); }
        public boolean getstatic_Z(jq_StaticField f) { return Unsafe.peek(f.getAddress())!=0; }
        public void putstatic_I(jq_StaticField f, int v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_L(jq_StaticField f, long v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_F(jq_StaticField f, float v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_D(jq_StaticField f, double v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_A(jq_StaticField f, Object v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_Z(jq_StaticField f, boolean v) { f.getDeclaringClass().setStaticData(f, v?1:0); }
        public void putstatic_B(jq_StaticField f, byte v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_C(jq_StaticField f, char v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_S(jq_StaticField f, short v) { f.getDeclaringClass().setStaticData(f, v); }
        public int getfield_I(Object o, jq_InstanceField f) { return Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()); }
        public long getfield_L(Object o, jq_InstanceField f) { int lo=Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()); int hi=Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()+4); return jq.twoIntsToLong(lo, hi); }
        public float getfield_F(Object o, jq_InstanceField f) { return Float.intBitsToFloat(getfield_I(o, f)); }
        public double getfield_D(Object o, jq_InstanceField f) { return Double.longBitsToDouble(getfield_L(o, f)); }
        public Object getfield_A(Object o, jq_InstanceField f) { return Unsafe.asObject(Unsafe.peek(Unsafe.addressOf(o)+f.getOffset())); }
        public byte getfield_B(Object o, jq_InstanceField f) { return (byte)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()); }
        public char getfield_C(Object o, jq_InstanceField f) { return (char)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()); }
        public short getfield_S(Object o, jq_InstanceField f) { return (short)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()); }
        public boolean getfield_Z(Object o, jq_InstanceField f) { return (Unsafe.peek(Unsafe.addressOf(o)+f.getOffset())&0xFF)!=0; }
        public void putfield_I(Object o, jq_InstanceField f, int v) { Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), v); }
        public void putfield_L(Object o, jq_InstanceField f, long v) { Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), (int)(v>>32)); Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset()+4, (int)v); }
        public void putfield_F(Object o, jq_InstanceField f, float v) { putfield_I(o, f, Float.floatToRawIntBits(v)); }
        public void putfield_D(Object o, jq_InstanceField f, double v) { putfield_L(o, f, Double.doubleToRawLongBits(v)); }
        public void putfield_A(Object o, jq_InstanceField f, Object v) { Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), Unsafe.addressOf(v)); }
        public void putfield_B(Object o, jq_InstanceField f, byte v) { Unsafe.poke1(Unsafe.addressOf(o)+f.getOffset(), v); }
        public void putfield_C(Object o, jq_InstanceField f, char v) { Unsafe.poke2(Unsafe.addressOf(o)+f.getOffset(), (short)((v<<16)>>16)); }
        public void putfield_S(Object o, jq_InstanceField f, short v) { Unsafe.poke2(Unsafe.addressOf(o)+f.getOffset(), v); }
        public void putfield_Z(Object o, jq_InstanceField f, boolean v) { Unsafe.poke1(Unsafe.addressOf(o)+f.getOffset(), v?(byte)1:(byte)0); }
        public Object new_obj(jq_Type t) { return ((jq_Class)t).newInstance(); }
        public Object new_array(jq_Type t, int length) { return ((jq_Array)t).newInstance(length); }
        public Object checkcast(Object o, jq_Type t) { return TypeCheck.checkcast(o, t); }
        public boolean instance_of(Object o, jq_Type t) { return TypeCheck.instance_of(o, t); }
        public int arraylength(Object o) { return Unsafe.peek(Unsafe.addressOf(o)+ARRAY_LENGTH_OFFSET); }
        public void monitorenter(Object o, MethodInterpreter v) { Monitor.monitorenter(o); }
        public void monitorexit(Object o) { Monitor.monitorexit(o); }
        public Object multinewarray(int[] dims, jq_Type t) { return HeapAllocator.multinewarray_helper(dims, 0, (jq_Array)t); }
        public jq_Type getJQTypeOf(Object o) { return Unsafe.getTypeOf(o); }
    }

}

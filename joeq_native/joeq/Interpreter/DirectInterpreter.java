/*
 * DirectInterpreter.java
 *
 * Created on January 16, 2001, 1:06 AM
 * 
 */

package Interpreter;

import Allocator.HeapAllocator;
import Allocator.ObjectLayout;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Main.jq;
import Memory.Address;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.Monitor;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;

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
        StackAddress newframe = StackAddress.alloca(localsize+stacksize);
        DirectState callee = new DirectState(newframe, (StackAddress) newframe.offset(localsize), m.getMaxLocals());
        return this.invokeMethod(m, callee);
    }
    
    public Object invokeUnsafeMethod(jq_StaticMethod f) throws Throwable {
        if (f == Unsafe._intBitsToFloat) {
            return new Float(state.pop_F());
        } else if (f == Unsafe._doubleToLongBits) {
            return new Long(state.pop_L());
        } else if (f == Unsafe._longBitsToDouble) {
            return new Double(state.pop_D());
        } else if (f == Unsafe._getThreadBlock) {
            return Unsafe.getThreadBlock();
        } else {
            System.err.println(f.toString());
            jq.UNREACHABLE();
            return null;
        }
    }
        
    public static class DirectState extends BytecodeInterpreter.State {
        final StackAddress fp;
        final int nlocals;
        StackAddress sp;
        int loResult, hiResult;
        
        public DirectState(StackAddress fp, StackAddress sp, int nlocals) {
            this.fp = fp; this.sp = sp;
            this.nlocals = nlocals;
        }

        public void push_I(int v) {
            sp = (StackAddress) sp.offset(-4);
            sp.poke4(v);
        }
        public void push_L(long v) {
            push_I((int)(v>>32)); push_I((int)v); // hi, lo
        }
        public void push_F(float v) {
            push_I(Float.floatToRawIntBits(v));
        }
        public void push_D(double v) {
            push_L(Double.doubleToRawLongBits(v));
        }
        public void push_A(Object v) {
            push_R(HeapAddress.addressOf(v));
        }
        public void push_R(Address v) {
            sp = (StackAddress) sp.offset(-4);
            sp.poke(v);
        }
        public void push(Object v) {
            push_A(v);
        }
        public int pop_I() {
            int v = sp.peek4();
            sp = (StackAddress) sp.offset(4);
            return v;
        }
        public long pop_L() {
            int lo=pop_I(); int hi=pop_I();
            return jq.twoIntsToLong(lo, hi); // lo, hi
        }
        public float pop_F() {
            return Float.intBitsToFloat(pop_I());
        }
        public double pop_D() {
            return Double.longBitsToDouble(pop_L());
        }
        public Object pop_A() {
            return ((HeapAddress) pop_R()).asObject();
        }
        public Address pop_R() {
            Address v = sp.peek();
            sp = (StackAddress) sp.offset(4);
            return v;
        }
        public Object pop() {
            return pop_A();
        }
        public void popAll() {
            sp = (StackAddress) fp.offset(-(nlocals<<2));
        }
        public Object peek_A(int depth) {
            HeapAddress v = (HeapAddress) sp.offset(depth << 2).peek();
            return v.asObject();
        }
        public void setLocal_I(int i, int v) {
            fp.offset(-(i << 2)).poke4(v);
        }
        public void setLocal_L(int i, long v) {
            setLocal_I(i, (int)(v>>32)); setLocal_I(i+1, (int)v); // hi, lo
        }
        public void setLocal_F(int i, float v) {
            setLocal_I(i, Float.floatToRawIntBits(v));
        }
        public void setLocal_D(int i, double v) {
            setLocal_L(i, Double.doubleToRawLongBits(v));
        }
        public void setLocal_A(int i, Object v) {
            setLocal_R(i, HeapAddress.addressOf(v));
        }
        public void setLocal_R(int i, Address v) {
            fp.offset(-(i << 2)).poke(v);
        }
        public int getLocal_I(int i) {
            return fp.offset(-(i << 2)).peek4();
        }
        public long getLocal_L(int i) {
            int lo=getLocal_I(i+1); int hi=getLocal_I(i); // lo, hi
            return jq.twoIntsToLong(lo, hi);
        }
        public float getLocal_F(int i) {
            return Float.intBitsToFloat(getLocal_I(i));
        }
        public double getLocal_D(int i) {
            return Double.longBitsToDouble(getLocal_L(i));
        }
        public Object getLocal_A(int i) {
            return ((HeapAddress) getLocal_R(i)).asObject();
        }
        public Address getLocal_R(int i) {
            return fp.offset(-(i << 2)).peek();
        }
        public void return_I(int v) {
            loResult = v;
        }
        public void return_L(long v) {
            loResult = (int)(v>>32); hiResult = (int)v;
        }
        public void return_F(float v) {
            loResult = Float.floatToRawIntBits(v);
        }
        public void return_D(double v) {
            return_L(Double.doubleToRawLongBits(v));
        }
        public void return_A(Object v) {
            loResult = HeapAddress.addressOf(v).to32BitValue();
        }
        public void return_V() {}
        public int getReturnVal_I() {
            return loResult;
        }
        public long getReturnVal_L() {
            return jq.twoIntsToLong(loResult, hiResult);
        }
        public float getReturnVal_F() {
            return Float.intBitsToFloat(loResult);
        }
        public double getReturnVal_D() {
            return Double.longBitsToDouble(getReturnVal_L());
        }
        public Object getReturnVal_A() {
            return ((HeapAddress) getReturnVal_R()).asObject();
        }
        public Address getReturnVal_R() {
            return HeapAddress.address32(loResult);
        }
    }
    
    public static class DirectVMInterface extends BytecodeInterpreter.VMInterface {
        public int getstatic_I(jq_StaticField f) { return f.getAddress().peek4(); }
        public long getstatic_L(jq_StaticField f) { return f.getAddress().peek8(); }
        public float getstatic_F(jq_StaticField f) { return Float.intBitsToFloat(getstatic_I(f)); }
        public double getstatic_D(jq_StaticField f) { return Double.longBitsToDouble(getstatic_L(f)); }
        public Object getstatic_A(jq_StaticField f) { return ((HeapAddress) f.getAddress().peek()).asObject(); }
        public byte getstatic_B(jq_StaticField f) { return (byte)f.getAddress().peek4(); }
        public char getstatic_C(jq_StaticField f) { return (char)f.getAddress().peek4(); }
        public short getstatic_S(jq_StaticField f) { return (short)f.getAddress().peek4(); }
        public boolean getstatic_Z(jq_StaticField f) { return f.getAddress().peek4()!=0; }
        public void putstatic_I(jq_StaticField f, int v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_L(jq_StaticField f, long v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_F(jq_StaticField f, float v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_D(jq_StaticField f, double v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_A(jq_StaticField f, Object v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_Z(jq_StaticField f, boolean v) { f.getDeclaringClass().setStaticData(f, v?1:0); }
        public void putstatic_B(jq_StaticField f, byte v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_C(jq_StaticField f, char v) { f.getDeclaringClass().setStaticData(f, v); }
        public void putstatic_S(jq_StaticField f, short v) { f.getDeclaringClass().setStaticData(f, v); }
        public int getfield_I(Object o, jq_InstanceField f) { return HeapAddress.addressOf(o).offset(f.getOffset()).peek4(); }
        public long getfield_L(Object o, jq_InstanceField f) { return HeapAddress.addressOf(o).offset(f.getOffset()).peek8(); }
        public float getfield_F(Object o, jq_InstanceField f) { return Float.intBitsToFloat(getfield_I(o, f)); }
        public double getfield_D(Object o, jq_InstanceField f) { return Double.longBitsToDouble(getfield_L(o, f)); }
        public Object getfield_A(Object o, jq_InstanceField f) { return ((HeapAddress)HeapAddress.addressOf(o).offset(f.getOffset()).peek()).asObject(); }
        public byte getfield_B(Object o, jq_InstanceField f) { return HeapAddress.addressOf(o).offset(f.getOffset()).peek1(); }
        public char getfield_C(Object o, jq_InstanceField f) { return (char)HeapAddress.addressOf(o).offset(f.getOffset()).peek4(); }
        public short getfield_S(Object o, jq_InstanceField f) { return (short)HeapAddress.addressOf(o).offset(f.getOffset()).peek2(); }
        public boolean getfield_Z(Object o, jq_InstanceField f) { return HeapAddress.addressOf(o).offset(f.getOffset()).peek1() != (byte)0; }
        public void putfield_I(Object o, jq_InstanceField f, int v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke4(v); }
        public void putfield_L(Object o, jq_InstanceField f, long v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke8(v); }
        public void putfield_F(Object o, jq_InstanceField f, float v) { putfield_I(o, f, Float.floatToRawIntBits(v)); }
        public void putfield_D(Object o, jq_InstanceField f, double v) { putfield_L(o, f, Double.doubleToRawLongBits(v)); }
        public void putfield_A(Object o, jq_InstanceField f, Object v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke(HeapAddress.addressOf(v)); }
        public void putfield_B(Object o, jq_InstanceField f, byte v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke1(v); }
        public void putfield_C(Object o, jq_InstanceField f, char v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke2((short)((v<<16)>>16)); }
        public void putfield_S(Object o, jq_InstanceField f, short v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke2(v); }
        public void putfield_Z(Object o, jq_InstanceField f, boolean v) { HeapAddress.addressOf(o).offset(f.getOffset()).poke1(v?(byte)1:(byte)0); }
        public Object new_obj(jq_Type t) { return ((jq_Class)t).newInstance(); }
        public Object new_array(jq_Type t, int length) { return ((jq_Array)t).newInstance(length); }
        public Object checkcast(Object o, jq_Type t) { return TypeCheck.checkcast(o, t); }
        public boolean instance_of(Object o, jq_Type t) { return TypeCheck.instance_of(o, t); }
        public int arraylength(Object o) { return HeapAddress.addressOf(o).offset(ARRAY_LENGTH_OFFSET).peek4(); }
        public void monitorenter(Object o, MethodInterpreter v) { Monitor.monitorenter(o); }
        public void monitorexit(Object o) { Monitor.monitorexit(o); }
        public Object multinewarray(int[] dims, jq_Type t) { return HeapAllocator.multinewarray_helper(dims, 0, (jq_Array)t); }
        public jq_Reference getJQTypeOf(Object o) { return jq_Reference.getTypeOf(o); }
    }

}

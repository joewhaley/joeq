/*
 * Unsafe.java
 *
 * Created on January 2, 2001, 2:55 AM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Reference;
import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_NameAndDesc;
import Clazz.jq_CompiledCode;
import Clazz.jq_StaticMethod;
import Clazz.jq_StaticField;
import Clazz.jq_InstanceField;
import Scheduler.jq_Thread;
import UTF.Utf8;

public abstract class Unsafe {

    private static /*final*/ Remapper remapper_object = new Remapper();

    public static void installRemapper(Remapper r) {
        remapper_object = r;
    }
    
    public static final native int floatToIntBits(float i);
    public static final native float intBitsToFloat(int i);
    public static final native long doubleToLongBits(double i);
    public static final native double longBitsToDouble(long i);
    
    public static final int addressOf(Object o) { return remapper_object.addressOf(o); }
    public static final Object asObject(int a) { return remapper_object.asObject(a); }
    public static final int peek(int a) { return remapper_object.peek(a); }
    public static final void poke1(int a, byte v) { remapper_object.poke1(a, v); }
    public static final void poke2(int a, short v) { remapper_object.poke2(a, v); }
    public static final void poke4(int a, int v) { remapper_object.poke4(a, v); }
    public static final jq_Reference getTypeOf(Object o) { return remapper_object.getTypeOf(o); }
    
    public static final native void pushArg(int arg);
    public static final native float popFP32();
    public static final native double popFP64();
    public static final native void pushFP32(float v);
    public static final native void pushFP64(double v);
    public static final native long invoke(int address) throws Throwable;
    public static final native int alloca(int size);
    public static final native int EAX();
    public static final native int ESP();
    public static final native int EBP();
    public static final native jq_Thread getThreadBlock();
    public static final native void setThreadBlock(jq_Thread t);
    public static final native void longJump(int ip, int fp, int sp, int eax);
    public static final native void atomicAdd(int address, int val);
    public static final native void atomicSub(int address, int val);
    public static final native void atomicAnd(int address, int val);
    public static final int atomicCas4(int address, int before, int after) {
        int val = peek(address);
        if (val == before) { remapper_object.poke4(address, after); return after; }
        return val;
    }
    public static final native boolean isEQ();

    public static class Remapper {
        public native int addressOf(Object o);
        public native Object asObject(int a);
        public native int peek(int a);
        public native void poke1(int a, byte v);
        public native void poke2(int a, short v);
        public native void poke4(int a, int v);
        public native jq_Reference getTypeOf(Object o);
    }

    public static final jq_Class _class;
    public static final jq_StaticField _remapper_object;
    public static final jq_StaticMethod _addressOf;
    public static final jq_StaticMethod _asObject;
    public static final jq_StaticMethod _peek;
    public static final jq_StaticMethod _poke1;
    public static final jq_StaticMethod _poke2;
    public static final jq_StaticMethod _poke4;
    public static final jq_StaticMethod _getTypeOf;
    public static final jq_StaticMethod _pushArg;
    public static final jq_StaticMethod _popFP32;
    public static final jq_StaticMethod _popFP64;
    public static final jq_StaticMethod _pushFP32;
    public static final jq_StaticMethod _pushFP64;
    public static final jq_StaticMethod _invoke;
    public static final jq_StaticMethod _alloca;
    public static final jq_StaticMethod _EAX;
    public static final jq_StaticMethod _EBP;
    public static final jq_StaticMethod _ESP;
    public static final jq_StaticMethod _getThreadBlock;
    public static final jq_StaticMethod _setThreadBlock;
    public static final jq_StaticMethod _longJump;
    public static final jq_StaticMethod _atomicAdd;
    public static final jq_StaticMethod _atomicSub;
    public static final jq_StaticMethod _atomicAnd;
    public static final jq_StaticMethod _atomicCas4;
    public static final jq_StaticMethod _isEQ;
    public static final jq_StaticMethod _floatToIntBits;
    public static final jq_StaticMethod _intBitsToFloat;
    public static final jq_StaticMethod _doubleToLongBits;
    public static final jq_StaticMethod _longBitsToDouble;
    
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Unsafe;");
        _remapper_object = _class.getOrCreateStaticField("remapper_object", "LRun_Time/Unsafe$Remapper;");
        _addressOf = _class.getOrCreateStaticMethod("addressOf", "(Ljava/lang/Object;)I");
        _asObject = _class.getOrCreateStaticMethod("asObject", "(I)Ljava/lang/Object;");
        _peek = _class.getOrCreateStaticMethod("peek", "(I)I");
        _poke1 = _class.getOrCreateStaticMethod("poke1", "(IB)V");
        _poke2 = _class.getOrCreateStaticMethod("poke2", "(IS)V");
        _poke4 = _class.getOrCreateStaticMethod("poke4", "(II)V");
        _getTypeOf = _class.getOrCreateStaticMethod("getTypeOf", "(Ljava/lang/Object;)LClazz/jq_Reference;");
        _pushArg = _class.getOrCreateStaticMethod("pushArg", "(I)V");
        _popFP32 = _class.getOrCreateStaticMethod("popFP32", "()F");
        _popFP64 = _class.getOrCreateStaticMethod("popFP64", "()D");
        _pushFP32 = _class.getOrCreateStaticMethod("pushFP32", "(F)V");
        _pushFP64 = _class.getOrCreateStaticMethod("pushFP64", "(D)V");
        _invoke = _class.getOrCreateStaticMethod("invoke", "(I)J");
        _alloca = _class.getOrCreateStaticMethod("alloca", "(I)I");
        _EAX = _class.getOrCreateStaticMethod("EAX", "()I");
        _EBP = _class.getOrCreateStaticMethod("EBP", "()I");
        _ESP = _class.getOrCreateStaticMethod("ESP", "()I");
        _getThreadBlock = _class.getOrCreateStaticMethod("getThreadBlock", "()LScheduler/jq_Thread;");
        _setThreadBlock = _class.getOrCreateStaticMethod("setThreadBlock", "(LScheduler/jq_Thread;)V");
        _longJump = _class.getOrCreateStaticMethod("longJump", "(IIII)V");
        _atomicAdd = _class.getOrCreateStaticMethod("atomicAdd", "(II)V");
        _atomicSub = _class.getOrCreateStaticMethod("atomicSub", "(II)V");
        _atomicAnd = _class.getOrCreateStaticMethod("atomicAnd", "(II)V");
        _atomicCas4 = _class.getOrCreateStaticMethod("atomicCas4", "(III)I");
        _isEQ = _class.getOrCreateStaticMethod("isEQ", "()Z");
        _floatToIntBits = _class.getOrCreateStaticMethod("floatToIntBits", "(F)I");
        _intBitsToFloat = _class.getOrCreateStaticMethod("intBitsToFloat", "(I)F");
        _doubleToLongBits = _class.getOrCreateStaticMethod("doubleToLongBits", "(D)J");
        _longBitsToDouble = _class.getOrCreateStaticMethod("longBitsToDouble", "(J)D");
    }
    
}

/*
 * Unsafe.java
 */

package ClassLib.Common.sun.misc;

import ClassLib.ClassLibInterface;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Clazz.jq_Type;
import Memory.HeapAddress;
import Run_Time.SystemInterface;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Unsafe {

    public java.lang.Object getObject(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return ((HeapAddress)a.offset(x).peek()).asObject();
    }
    
    public void putObject(java.lang.Object o1, int x, java.lang.Object v) {
        HeapAddress a = HeapAddress.addressOf(o1);
        a.offset(x).poke(HeapAddress.addressOf(v));
    }
    
    public boolean getBoolean(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return a.offset(x).peek1() != (byte)0;
    }
    
    public void putBoolean(java.lang.Object o, int x, boolean v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke1(v?(byte)1:(byte)0);
    }
    
    public byte getByte(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return a.offset(x).peek1();
    }
    
    public void putByte(java.lang.Object o, int x, byte v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke1(v);
    }
    
    public short getShort(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return a.offset(x).peek2();
    }
    
    public void putShort(java.lang.Object o, int x, short v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke2(v);
    }
    
    public char getChar(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return (char) a.offset(x).peek2();
    }
    
    public void putChar(java.lang.Object o, int x, char v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke2((short) v);
    }
    
    public int getInt(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return a.offset(x).peek4();
    }
    
    public void putInt(java.lang.Object o, int x, int v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke4(v);
    }
    
    public long getLong(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return a.offset(x).peek8();
    }
    
    public void putLong(java.lang.Object o, int x, long v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke8(v);
    }
    
    public float getFloat(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return Float.intBitsToFloat(a.offset(x).peek4());
    }
    
    public void putFloat(java.lang.Object o, int x, float v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke4(Float.floatToRawIntBits(v));
    }
    
    public double getDouble(java.lang.Object o, int x) {
        HeapAddress a = HeapAddress.addressOf(o);
        return Double.longBitsToDouble(a.offset(x).peek8());
    }
    
    public void putDouble(java.lang.Object o, int x, double v) {
        HeapAddress a = HeapAddress.addressOf(o);
        a.offset(x).poke8(Double.doubleToRawLongBits(v));
    }
    
    public byte getByte(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return a.peek1();
    }
    
    public void putByte(long addr, byte v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke1(v);
    }
    
    public short getShort(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return a.peek2();
    }
    
    public void putShort(long addr, short v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke2(v);
    }
    
    public char getChar(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return (char) a.peek2();
    }
    
    public void putChar(long addr, char v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke2((short)v);
    }
    
    public int getInt(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return a.peek4();
    }
    
    public void putInt(long addr, int v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke4(v);
    }
    
    public long getLong(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return a.peek8();
    }
    
    public void putLong(long addr, long v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke8(v);
    }
    
    public float getFloat(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return Float.intBitsToFloat(a.peek4());
    }
    
    public void putFloat(long addr, float v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke4(Float.floatToRawIntBits(v));
    }
    
    public double getDouble(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return Double.longBitsToDouble(a.peek8());
    }
    
    public void putDouble(long addr, double v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        a.poke8(Double.doubleToRawLongBits(v));
    }
    
    public long getAddress(long addr) {
        HeapAddress a = HeapAddress.address32((int) addr);
        return (long) a.peek().to32BitValue();
    }
    
    public void putAddress(long addr, long v) {
        HeapAddress a = HeapAddress.address32((int) addr);
        HeapAddress b = HeapAddress.address32((int) v);
        a.poke(b);
    }
    
    public long allocateMemory(long v) {
        return SystemInterface.syscalloc((int) v).to32BitValue();
    }
    
    //public long reallocateMemory(long addr, long size) {
    
    public void setMemory(long to, long size, byte b) {
        HeapAddress a = HeapAddress.address32((int) to);
        SystemInterface.mem_set(a, (int) size, b);
    }
    
    public void copyMemory(long to, long from, long size) {
        HeapAddress a = HeapAddress.address32((int) to);
        HeapAddress b = HeapAddress.address32((int) from);
        SystemInterface.mem_cpy(a, b, (int) size);
    }
    
    public void freeMemory(long v) {
        HeapAddress a = HeapAddress.address32((int) v);
        SystemInterface.sysfree(a);
    }
    
    public int fieldOffset(java.lang.reflect.Field field) {
        jq_Field f = (jq_Field) ClassLibInterface.DEFAULT.getJQField(field);
        jq_Class c = f.getDeclaringClass();
        c.load(); c.verify(); c.prepare();
        if (f instanceof jq_InstanceField) {
            return ((jq_InstanceField)f).getOffset();
        } else {
            HeapAddress a = ((jq_StaticField)f).getAddress();
            HeapAddress b = HeapAddress.addressOf(c.getStaticData());
            return b.difference(a);
        }
    }
    
    public java.lang.Object staticFieldBase(java.lang.Class k) {
        jq_Type t = ClassLibInterface.DEFAULT.getJQType(k);
        if (t instanceof jq_Class) {
            jq_Class c = (jq_Class) t;
            return c.getStaticData();
        }
        return null;
    }
    
    public void ensureClassInitialized(java.lang.Class k) {
        jq_Type t = ClassLibInterface.DEFAULT.getJQType(k);
        t.load(); t.verify(); t.prepare(); t.sf_initialize(); t.cls_initialize();
    }
    
    public int arrayBaseOffset(java.lang.Class k) {
        return 0;
    }
    
    public int arrayIndexScale(java.lang.Class k) {
        jq_Type t = ClassLibInterface.DEFAULT.getJQType(k);
        if (t instanceof jq_Array) {
            int width = ((jq_Array)t).getElementType().getReferenceSize();
            switch (width) {
            case 4: return 2;
            case 2: return 1;
            case 1: return 0;
            case 8: return 3;
            }
        }
        return -1;
    }
    
    public int addressSize() {
        return HeapAddress.size();
    }
    
    public int pageSize() {
        return 1 << HeapAddress.pageAlign();
    }
    
    public java.lang.Class defineClass(java.lang.String name, byte[] b, int off, int len, ClassLib.Common.java.lang.ClassLoader cl, java.security.ProtectionDomain pd) {
        return cl.defineClass0(name, b, off, len, pd);
    }
    
    public java.lang.Class defineClass(java.lang.String name, byte[] b, int off, int len, ClassLib.Common.java.lang.ClassLoader cl) {
        return cl.defineClass0(name, b, off, len, null);
    }
    
    public java.lang.Object allocateInstance(java.lang.Class k)
    throws java.lang.InstantiationException {
        jq_Type t = ClassLibInterface.DEFAULT.getJQType(k);
        if (t instanceof jq_Class) {
            jq_Class c = (jq_Class) t;
            return c.newInstance();
        }
        throw new java.lang.InstantiationException();
    }
    
    public void monitorEnter(java.lang.Object o) {
        Run_Time.Monitor.monitorenter(o);
    }
    
    public void monitorExit(java.lang.Object o) {
        Run_Time.Monitor.monitorexit(o);
    }
    
    public void throwException(java.lang.Throwable o) {
        Run_Time.ExceptionDeliverer.athrow(o);
    }
}

/*
 * BootImage.java
 *
 * Created on January 14, 2001, 11:56 AM
 *
 * @author  jwhaley
 * @version 
 */

package Bootstrap;

import Clazz.jq_Class;
import Clazz.jq_Array;
import Clazz.jq_Type;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_InstanceField;
import Allocator.ObjectLayout;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Util.IdentityHashCodeWrapper;
import Compil3r.Reference.x86.x86ReferenceCompiler;
import jq;
import java.util.Iterator;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.ArrayList;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

public class BootImage extends Unsafe.Remapper implements ObjectLayout {

    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    private final Map/*<IdentityHashCodeWrapper, Entry>*/ hash;
    private final ArrayList/*<Entry>*/ entries;
    private final ObjectTraverser obj_trav;
    private int heapCurrent;
    private final int startAddress;
    private byte[] image;
    
    public BootImage(int startAddress, ObjectTraverser obj_trav, int initialCapacity, float loadFactor) {
        hash = new HashMap(initialCapacity, loadFactor);
        entries = new ArrayList(initialCapacity);
        this.obj_trav = obj_trav;
        this.heapCurrent = this.startAddress = startAddress;
    }
    public BootImage(int startAddress, ObjectTraverser obj_trav, int initialCapacity) {
        hash = new HashMap(initialCapacity);
        entries = new ArrayList(initialCapacity);
        this.obj_trav = obj_trav;
        this.heapCurrent = this.startAddress = startAddress;
    }
    public BootImage(int startAddress, ObjectTraverser obj_trav) {
        hash = new HashMap();
        entries = new ArrayList();
        this.obj_trav = obj_trav;
        this.heapCurrent = this.startAddress = startAddress;
    }

    public final int addressOf(Object o) {
        return getOrAllocateObject(o);
    }
    public final int peek(int a) {
        return jq.fourBytesToInt(image, a-startAddress);
    }
    public final void poke1(int p, byte v) {
        p -= startAddress;
        image[p  ] = (byte)(v);
    }
    public final void poke2(int p, short v) {
        p -= startAddress;
        image[p  ] = (byte)(v);
        image[p+1] = (byte)(v >>  8);
    }
    public final void poke2(int p, char v) {
        p -= startAddress;
        image[p  ] = (byte)(v);
        image[p+1] = (byte)(v >>  8);
    }
    public final void poke4(int p, int v) {
        p -= startAddress;
        image[p  ] = (byte)(v);
        image[p+1] = (byte)(v >>  8);
        image[p+2] = (byte)(v >> 16);
        image[p+3] = (byte)(v >> 24);
    }
    public void poke4(int p, float v) { poke4(p, Float.floatToRawIntBits(v)); }
    public final void poke8(int p, long v) {
        p -= startAddress;
        image[p  ] = (byte)(v);
        image[p+1] = (byte)(v >>  8);
        image[p+2] = (byte)(v >> 16);
        image[p+3] = (byte)(v >> 24);
        image[p+4] = (byte)(v >> 32);
        image[p+5] = (byte)(v >> 40);
        image[p+6] = (byte)(v >> 48);
        image[p+7] = (byte)(v >> 56);
    }
    public final void poke8(int p, double v) { poke8(p, Double.doubleToRawLongBits(v)); }
    
    public final void invokeclinit(jq_Class c) {
        // call forName on this type to trigger class initialization
        String cname = c.getName().toString();
        try {
            Class.forName(cname);
        } catch (ClassNotFoundException x) {
            // bootstrapping jvm can't find the class?
            System.err.println("ERROR: bootstrapping jvm cannot find class "+cname);
            jq.UNREACHABLE();
        }
    }

    private boolean alloc_enabled = false;
    
    public void enableAllocations() { alloc_enabled = true; }
    public void disableAllocations() { alloc_enabled = false; }
    
    public int/*Address*/ getOrAllocateObject(Object o) {
        if (o == null) return 0;
        IdentityHashCodeWrapper k = IdentityHashCodeWrapper.create(o);
        Entry e = (Entry)hash.get(k);
        if (e != null) return e.getAddress();
        // not yet allocated, allocate it.
        jq.assert(alloc_enabled);
        Class objType = o.getClass();
        jq_Reference type = (jq_Reference)Reflection.getJQType(objType);
        if (!type.isLoaded()) {
            jq.UNREACHABLE("class "+type+" is not loaded!");
            return 0;
        }
        int addr, size;
        if (type.isArrayType()) {
            addr = heapCurrent + ARRAY_HEADER_SIZE;
            size = ((jq_Array)type).getInstanceSize(Array.getLength(o));
            size = (size+3) & ~3;
            if (TRACE)
                out.println("Allocating entry "+entries.size()+": "+objType+" length "+Array.getLength(o)+" size "+size+" "+jq.hex(System.identityHashCode(o))+" at "+jq.hex(addr));
        } else {
            jq.assert(type.isClassType());
            addr = heapCurrent + OBJ_HEADER_SIZE;
            size = ((jq_Class)type).getInstanceSize();
            if (TRACE)
                out.println("Allocating entry "+entries.size()+": "+objType+" size "+size+" "+jq.hex(System.identityHashCode(o))+" at "+jq.hex(addr));
        }
        heapCurrent += size;
        e = Entry.create(o, addr);
        hash.put(k, e);
        entries.add(e);
        return addr;
    }
    
    public int/*Address*/ getAddressOf(Object o) {
        if (o == null) return 0;
        IdentityHashCodeWrapper k = IdentityHashCodeWrapper.create(o);
        Entry e = (Entry)hash.get(k);
        if (e == null) {
            jq.UNREACHABLE(o.getClass()+" "+jq.hex(System.identityHashCode(o)));
        }
        return e.getAddress();
    }

    public void initStaticField(jq_StaticField f) {
        if (TRACE) out.println("Initializing static field "+f);
        Object val = obj_trav.getStaticFieldValue(f);
        jq_Class k = f.getDeclaringClass();
        jq_Type ftype = f.getType();
        if (ftype.isPrimitiveType()) {
            if (ftype == jq_Primitive.INT)
                k.setStaticData(f, (val==null)?0:((Integer)val).intValue());
            else if (ftype == jq_Primitive.FLOAT)
                k.setStaticData(f, (val==null)?0F:((Float)val).floatValue());
            else if (ftype == jq_Primitive.LONG)
                k.setStaticData(f, (val==null)?0L:((Long)val).longValue());
            else if (ftype == jq_Primitive.DOUBLE)
                k.setStaticData(f, (val==null)?0.:((Double)val).doubleValue());
            else if (ftype == jq_Primitive.BOOLEAN)
                k.setStaticData(f, (val==null)?0:((Boolean)val).booleanValue()?1:0);
            else if (ftype == jq_Primitive.BYTE)
                k.setStaticData(f, (val==null)?0:((Byte)val).byteValue());
            else if (ftype == jq_Primitive.SHORT)
                k.setStaticData(f, (val==null)?0:((Short)val).shortValue());
            else if (ftype == jq_Primitive.CHAR)
                k.setStaticData(f, (val==null)?0:((Character)val).charValue());
            else
                jq.UNREACHABLE();
        } else {
            k.setStaticData(f, val);
        }
    }
        
    public void find_reachable() {
        for (int i=0; i<entries.size(); ++i) {
            Entry e = (Entry)entries.get(i);
            Object o = e.getObject();
            Class objType = o.getClass();
            jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
            if (TRACE)
                out.println("Entry "+i+": "+objType+" "+jq.hex(System.identityHashCode(o)));
            if (jqType.isArrayType()) {
                jq_Type elemType = ((jq_Array)jqType).getElementType();
                if (elemType.isReferenceType()) {
                    int length = Array.getLength(o);
                    Object[] v = (Object[])o;
                    for (int k=0; k<length; ++k) {
                        Object o2 = v[k];
                        getOrAllocateObject(o2);
                    }
                }
            } else {
                jq.assert(jqType.isClassType());
                jq_Class clazz = (jq_Class)jqType;
                jq_InstanceField[] fields = clazz.getInstanceFields();
                for (int k=0; k<fields.length; ++k) {
                    jq_InstanceField f = fields[k];
                    jq_Type ftype = f.getType();
                    if (ftype.isReferenceType()) {
                        Object val = obj_trav.getInstanceFieldValue(o, f);
                        getOrAllocateObject(val);
                    }
                }
            }
        }
    }

    public int size() { return heapCurrent-startAddress; }
    
    public void dump(OutputStream out) throws IOException {
        image = new byte[heapCurrent-startAddress];
        Iterator i = entries.iterator();
        int j=0;
        while (i.hasNext()) {
            Entry e = (Entry)i.next();
            Object o = e.getObject();
            int addr = e.getAddress();
            Class objType = o.getClass();
            jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
            if (TRACE)
                this.out.println("Dumping entry "+j+": "+objType+" "+jq.hex(System.identityHashCode(o)));
            int vtable = getAddressOf(jqType.getVTable());
            if (jqType.isArrayType()) {
                int length = Array.getLength(o);
                poke4(addr+ARRAY_LENGTH_OFFSET, length);
                poke4(addr+VTABLE_OFFSET, vtable);
                jq_Type elemType = ((jq_Array)jqType).getElementType();
                if (elemType.isPrimitiveType()) {
                    if (elemType == jq_Primitive.INT) {
                        int[] v = (int[])o;
                        for (int k=0; k<length; ++k)
                            poke4(addr+ARRAY_ELEMENT_OFFSET+(k<<2), v[k]);
                    } else if (elemType == jq_Primitive.FLOAT) {
                        float[] v = (float[])o;
                        for (int k=0; k<length; ++k)
                            poke4(addr+ARRAY_ELEMENT_OFFSET+(k<<2), v[k]);
                    } else if (elemType == jq_Primitive.LONG) {
                        long[] v = (long[])o;
                        for (int k=0; k<length; ++k)
                            poke8(addr+ARRAY_ELEMENT_OFFSET+(k<<3), v[k]);
                    } else if (elemType == jq_Primitive.DOUBLE) {
                        double[] v = (double[])o;
                        for (int k=0; k<length; ++k)
                            poke8(addr+ARRAY_ELEMENT_OFFSET+(k<<3), v[k]);
                    } else if (elemType == jq_Primitive.BOOLEAN) {
                        boolean[] v = (boolean[])o;
                        for (int k=0; k<length; ++k)
                            poke1(addr+ARRAY_ELEMENT_OFFSET+k, v[k]?(byte)1:(byte)0);
                    } else if (elemType == jq_Primitive.BYTE) {
                        byte[] v = (byte[])o;
                        for (int k=0; k<length; ++k)
                            poke1(addr+ARRAY_ELEMENT_OFFSET+k, v[k]);
                    } else if (elemType == jq_Primitive.SHORT) {
                        short[] v = (short[])o;
                        for (int k=0; k<length; ++k)
                            poke2(addr+ARRAY_ELEMENT_OFFSET+(k<<1), v[k]);
                    } else if (elemType == jq_Primitive.CHAR) {
                        char[] v = (char[])o;
                        for (int k=0; k<length; ++k)
                            poke2(addr+ARRAY_ELEMENT_OFFSET+(k<<1), v[k]);
                    } else jq.UNREACHABLE();
                } else {
                    Object[] v = (Object[])o;
                    for (int k=0; k<length; ++k)
                        poke4(addr+ARRAY_ELEMENT_OFFSET+(k<<2), getAddressOf(v[k]));
                }
            } else {
                jq.assert(jqType.isClassType());
                jq_Class clazz = (jq_Class)jqType;
                poke4(addr+VTABLE_OFFSET, vtable);
                jq_InstanceField[] fields = clazz.getInstanceFields();
                for (int k=0; k<fields.length; ++k) {
                    jq_InstanceField f = fields[k];
                    jq_Type ftype = f.getType();
                    Object val = obj_trav.getInstanceFieldValue(o, f);
                    int foffset = f.getOffset();
                    if (TRACE) this.out.println("Field "+f+" offset "+jq.shex(foffset)+": "+jq.hex(System.identityHashCode(val)));
                    if (ftype.isPrimitiveType()) {
                        if (ftype == jq_Primitive.INT)
                            poke4(addr+foffset, (val==null)?0:((Integer)val).intValue());
                        else if (ftype == jq_Primitive.FLOAT)
                            poke4(addr+foffset, (val==null)?0F:((Float)val).floatValue());
                        else if (ftype == jq_Primitive.LONG)
                            poke8(addr+foffset, (val==null)?0L:((Long)val).longValue());
                        else if (ftype == jq_Primitive.DOUBLE)
                            poke8(addr+foffset, (val==null)?0.:((Double)val).doubleValue());
                        else if (ftype == jq_Primitive.BOOLEAN)
                            poke1(addr+foffset, (val==null)?(byte)0:((Boolean)val).booleanValue()?(byte)1:(byte)0);
                        else if (ftype == jq_Primitive.BYTE)
                            poke1(addr+foffset, (val==null)?(byte)0:((Byte)val).byteValue());
                        else if (ftype == jq_Primitive.SHORT)
                            poke2(addr+foffset, (val==null)?(short)0:((Short)val).shortValue());
                        else if (ftype == jq_Primitive.CHAR)
                            poke2(addr+foffset, (val==null)?(char)0:((Character)val).charValue());
                        else jq.UNREACHABLE();
                    } else {
                        poke4(addr+foffset, getAddressOf(val));
                    }
                }
            }
            ++j;
        }
        x86ReferenceCompiler.patchCalls();
        out.write(image, 0, image.length);
        int e_addr = SystemInterface._entry.getAddress();
        out.write((byte)e_addr);
        out.write((byte)(e_addr>>8));
        out.write((byte)(e_addr>>16));
        out.write((byte)(e_addr>>24));
        out.flush();
    }

    public static class Entry {
        private Object o;       // object in host vm
        private int address;    // address in target vm
        private Entry(Object o, int address) { this.o = o; this.address = address; }
        public static Entry create(Object o, int address) {
            jq.assert(o != null);
            return new Entry(o, address);
        }
        public Object getObject() { return o; }
        public int getAddress() { return address; }
    }
        
}

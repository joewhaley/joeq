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
import Clazz.jq_Method;
import Clazz.jq_StaticField;
import Clazz.jq_InstanceField;
import Allocator.ObjectLayout;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Util.IdentityHashCodeWrapper;
import Assembler.x86.Heap2HeapReference;
import Assembler.x86.Heap2CodeReference;
import Assembler.x86.ExternalReference;
import Assembler.x86.DirectBindCall;
import Assembler.x86.Reloc;
import UTF.Utf8;
import jq;
import java.util.List;
import java.util.LinkedList;
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

    private BootstrapCodeAllocator bca;
    private List data_relocs;
    
    public BootImage(ObjectTraverser obj_trav, BootstrapCodeAllocator bca, int initialCapacity, float loadFactor) {
        hash = new HashMap(initialCapacity, loadFactor);
        entries = new ArrayList(initialCapacity);
        this.obj_trav = obj_trav;
        this.bca = bca;
        this.heapCurrent = this.startAddress = 0;
        this.data_relocs = new LinkedList();
    }
    public BootImage(ObjectTraverser obj_trav, BootstrapCodeAllocator bca, int initialCapacity) {
        hash = new HashMap(initialCapacity);
        entries = new ArrayList(initialCapacity);
        this.obj_trav = obj_trav;
        this.bca = bca;
        this.heapCurrent = this.startAddress = 0;
        this.data_relocs = new LinkedList();
    }
    public BootImage(ObjectTraverser obj_trav, BootstrapCodeAllocator bca) {
        hash = new HashMap();
        entries = new ArrayList();
        this.obj_trav = obj_trav;
        this.bca = bca;
        this.heapCurrent = this.startAddress = 0;
        this.data_relocs = new LinkedList();
    }

    public final int addressOf(Object o) {
        return getOrAllocateObject(o);
    }
    
    public final void addCodeReloc(int addr, int target) {
        Heap2CodeReference r = new Heap2CodeReference(addr, target);
        data_relocs.add(r);
    }
    public final void addDataReloc(int addr, int target) {
        Heap2HeapReference r = new Heap2HeapReference(addr, target);
        data_relocs.add(r);
    }
    
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
    
    public int/*HeapAddress*/ getOrAllocateObject(Object o) {
        if (o == null) return 0;
        IdentityHashCodeWrapper k = IdentityHashCodeWrapper.create(o);
        Entry e = (Entry)hash.get(k);
        if (e != null) return e.getAddress();
        // not yet allocated, allocate it.
        jq.assert(alloc_enabled);
        Class objType = o.getClass();
        jq_Reference type = (jq_Reference)Reflection.getJQType(objType);
        if (!jq.boot_types.contains(type)) {
            System.err.println("--> class "+type+" is not in the set of boot types!");
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
                out.println("Allocating entry "+entries.size()+": "+objType+" size "+size+" "+jq.hex(System.identityHashCode(o))+" at "+jq.hex(addr)+((o instanceof jq_Type)?": "+o:""));
        }
        heapCurrent += size;
        e = Entry.create(o, addr);
        hash.put(k, e);
        entries.add(e);
        return addr;
    }
    
    public int/*HeapAddress*/ getAddressOf(Object o) {
        if (o == null) return 0;
        IdentityHashCodeWrapper k = IdentityHashCodeWrapper.create(o);
        Entry e = (Entry)hash.get(k);
        if (e == null) {
            jq.UNREACHABLE(o.getClass()+" "+jq.hex(System.identityHashCode(o)));
        }
        Class objType = o.getClass();
        jq_Reference type = (jq_Reference)Reflection.getJQType(objType);
        jq.assert(type.isClsInitialized(), type.toString());
        return e.getAddress();
    }

    public Object getObject(int i) {
        Entry e = (Entry)entries.get(i);
        return e.getObject();
    }
    
    public void addStaticFieldReloc(jq_StaticField f) {
        if (TRACE) out.println("Initializing static field "+f);
        Object val = obj_trav.getStaticFieldValue(f);
        jq_Type ftype = f.getType();
        if (ftype.isPrimitiveType()) {
            if (ftype == jq_Primitive.INT) {
                // some "int" fields actually refer to addresses
                if (f.isCodeAddressType()) {
                    addCodeReloc(f.getAddress(), ((Integer)val).intValue());
                } else if (f.isHeapAddressType()) {
                    addDataReloc(f.getAddress(), ((Integer)val).intValue());
                }
            }
        } else {
            if (val != null) {
                int addr = Unsafe.addressOf(val);
                addDataReloc(f.getAddress(), addr);
            }
        }
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
            int addr = Unsafe.addressOf(val);
            k.setStaticData(f, addr);
        }
    }
    
    public int numOfEntries() { return entries.size(); }

    public void find_reachable(int i) {
        for (; i<entries.size(); ++i) {
            Entry e = (Entry)entries.get(i);
            Object o = e.getObject();
            int addr = e.getAddress();
            Class objType = o.getClass();
            jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
            if (TRACE)
                out.println("Entry "+i+": "+objType+" "+jq.hex(System.identityHashCode(o)));
            addDataReloc(addr+VTABLE_OFFSET, getOrAllocateObject(jqType));
            if (jqType.isArrayType()) {
                jq_Type elemType = ((jq_Array)jqType).getElementType();
                if (elemType.isReferenceType()) {
                    int length = Array.getLength(o);
                    Object[] v = (Object[])o;
                    for (int k=0; k<length; ++k) {
                        Object o2 = v[k];
                        if (o2 != null) {
                            getOrAllocateObject(o2);
                            addDataReloc(addr+(k<<2), Unsafe.addressOf(o2));
                        }
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
                        if (val != null) {
                            getOrAllocateObject(val);
                            addDataReloc(addr+f.getOffset(), Unsafe.addressOf(val));
                        }
                    } else if (ftype == jq_Primitive.INT) {
                        // some "int" fields actually refer to addresses
                        if (f.isCodeAddressType()) {
                            Integer val = (Integer)obj_trav.getInstanceFieldValue(o, f);
                            addCodeReloc(addr+f.getOffset(), val.intValue());
                        } else if (f.isHeapAddressType()) {
                            Integer val = (Integer)obj_trav.getInstanceFieldValue(o, f);
                            if (val.intValue() != 0)
                                addDataReloc(addr+f.getOffset(), val.intValue());
                        }
                    }
                }
            }
        }
    }

    public int size() { return heapCurrent-startAddress; }
    
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
    
    public static final char F_RELFLG = (char)0x0001;
    public static final char F_EXEC   = (char)0x0002;
    public static final char F_LNNO   = (char)0x0004;
    public static final char F_LSYMS  = (char)0x0008;
    public static final char F_AR32WR = (char)0x0100;
    
    public void dumpFILHDR(OutputStream out, int symptr, int nsyms)
    throws IOException {
        // FILHDR
        write_ushort(out, (char)0x014c);    // f_magic
        write_ushort(out, (char)2);         // f_nscns
        long ms = System.currentTimeMillis();
        int s = (int)(ms/1000);
        write_ulong(out, s);                // f_timdat
        write_ulong(out, symptr);           // f_symptr
        write_ulong(out, nsyms);            // f_nsyms
        write_ushort(out, (char)0);         // f_opthdr
        write_ushort(out, (char)(F_LNNO | F_LSYMS | F_AR32WR)); // f_flags
    }
    
    public static final int STYP_TEXT  = 0x00000020;
    public static final int STYP_DATA  = 0x00000040;
    public static final int STYP_BSS   = 0x00000080;
    public static final int STYP_RELOV = 0x01000000;
    public static final int STYP_EXEC  = 0x20000000;
    public static final int STYP_READ  = 0x40000000;
    public static final int STYP_WRITE = 0x80000000;
    
    public void dumpTEXTSCNHDR(OutputStream out, int size, int nreloc)
    throws IOException {
        // SCNHDR
        write_bytes(out, ".text", 8);       // s_name
        write_ulong(out, 0);                // s_paddr
        write_ulong(out, 0);                // s_vaddr
        write_ulong(out, size);             // s_size
        write_ulong(out, 20+40+40);         // s_scnptr
        write_ulong(out, 20+40+40+size);    // s_relptr
        write_ulong(out, 0);                // s_lnnoptr
        if (nreloc > 65535)
            write_ushort(out, (char)0xffff); // s_nreloc
        else
            write_ushort(out, (char)nreloc); // s_nreloc
        write_ushort(out, (char)0);         // s_nlnno
        if (nreloc > 65535)
            write_ulong(out, STYP_TEXT | STYP_READ | STYP_WRITE | STYP_RELOV); // s_flags
        else
            write_ulong(out, STYP_TEXT | STYP_READ | STYP_WRITE); // s_flags
    }
    
    public void dumpDATASCNHDR(OutputStream out, int scnptr, int size, int nreloc)
    throws IOException {
        // SCNHDR
        write_bytes(out, ".data", 8);       // s_name
        write_ulong(out, 0);                // s_paddr
        write_ulong(out, 0);                // s_vaddr
        write_ulong(out, size);             // s_size
        write_ulong(out, scnptr);           // s_scnptr
        write_ulong(out, scnptr+size);      // s_relptr
        write_ulong(out, 0);                // s_lnnoptr
        if (nreloc > 65535)
            write_ushort(out, (char)0xffff); // s_nreloc
        else
            write_ushort(out, (char)nreloc); // s_nreloc
        write_ushort(out, (char)0);         // s_nlnno
        if (nreloc > 65535)
            write_ulong(out, STYP_DATA | STYP_READ | STYP_WRITE | STYP_RELOV); // s_flags
        else
            write_ulong(out, STYP_DATA | STYP_READ | STYP_WRITE); // s_flags
    }
    
    public static final char RELOC_ADDR32 = (char)0x0006;
    public static final char RELOC_REL32  = (char)0x0014;
    
    public void dumpLINENO(OutputStream out, int addr, char lnno)
    throws IOException {
        write_ulong(out, addr);     // l_symndx / l_paddr
        write_ushort(out, lnno);    // l_lnno
    }
    
    public static final short N_UNDEF = 0;
    public static final short N_ABS   = -1;
    public static final short N_DEBUG = -2;
    
    public static final char T_NULL   = 0x00;
    public static final char T_VOID   = 0x01;
    public static final char T_CHAR   = 0x02;
    public static final char T_SHORT  = 0x03;
    public static final char T_INT    = 0x04;
    public static final char T_LONG   = 0x05;
    public static final char T_FLOAT  = 0x06;
    public static final char T_DOUBLE = 0x07;
    public static final char T_STRUCT = 0x08;
    public static final char T_UNION  = 0x09;
    public static final char T_ENUM   = 0x0A;
    public static final char T_MOE    = 0x0B;
    public static final char T_UCHAR  = 0x0C;
    public static final char T_USHORT = 0x0D;
    public static final char T_UINT   = 0x0E;
    public static final char T_ULONG  = 0x0F;
    public static final char T_LNGDBL = 0x10;
    
    public static final char DT_NON = 0x0000;
    public static final char DT_PTR = 0x0100;
    public static final char DT_FCN = 0x0200;
    public static final char DT_ARY = 0x0300;

    public static final byte C_NULL   = 0;
    public static final byte C_AUTO   = 1;
    public static final byte C_EXT    = 2;
    public static final byte C_STAT   = 3;
    public static final byte C_REG    = 4;
    public static final byte C_EXTDEF = 5;
    public static final byte C_LABEL  = 6;
    public static final byte C_ULABEL = 7;
    public static final byte C_MOS     = 8;
    public static final byte C_ARG     = 9;
    public static final byte C_STRTAG  = 10;
    public static final byte C_MOU     = 11;
    public static final byte C_UNTAG   = 12;
    public static final byte C_TPDEF   = 13;
    public static final byte C_USTATIC = 14;
    public static final byte C_ENTAG   = 15;
    public static final byte C_MOE     = 16;
    public static final byte C_REGPARM = 17;
    public static final byte C_FIELD   = 18;
    public static final byte C_AUTOARG = 19;
    public static final byte C_LASTENT = 20;
    public static final byte C_BLOCK   = 100;
    public static final byte C_FCN     = 101;
    public static final byte C_EOS     = 102;
    public static final byte C_FILE    = 103;
    public static final byte C_SECTION = 104;
    public static final byte C_WEAKEXT = 105;
    public static final byte C_EFCN    = -1;
    
    public void dumpSECTIONSYMENTs(OutputStream out)
    throws IOException {
        write_bytes(out, ".text", 8);
        write_ulong(out, 0);
        write_short(out, (short)1);
        write_ushort(out, (char)0);
        write_uchar(out, C_STAT);
        write_uchar(out, (byte)0);
        
        write_bytes(out, ".data", 8);
        write_ulong(out, 0);
        write_short(out, (short)2);
        write_ushort(out, (char)0);
        write_uchar(out, C_STAT);
        write_uchar(out, (byte)0);
    }
    
    public void dumpEXTSYMENTs(OutputStream out)
    throws IOException {
        write_bytes(out, "_entry@0", 8);  // s_name
        int/*CodeAddress*/ addr = SystemInterface.entry_0;
        //int/*HeapAddress*/ addr = SystemInterface._entry.getAddress();
        write_ulong(out, addr);
        write_short(out, (short)1);
        write_ushort(out, (char)DT_FCN);
        write_uchar(out, C_EXT);
        write_uchar(out, (byte)0);
        
        write_ulong(out, 0);    // e_zeroes
        int idx = alloc_string("_trap_handler@8");
        write_ulong(out, idx);  // e_offset
        addr = SystemInterface.trap_handler_8;
        //addr = SystemInterface._trap_handler.getAddress();
        write_ulong(out, addr);
        write_short(out, (short)1);
        write_ushort(out, (char)DT_FCN);
        write_uchar(out, C_EXT);
        write_uchar(out, (byte)0);
    }
    
    public void dumpEXTDEFSYMENTs(OutputStream out, List extrefs)
    throws IOException {
        Iterator i = extrefs.iterator();
        int k = 4;
        while (i.hasNext()) {
            ExternalReference extref = (ExternalReference)i.next();
            jq.assert(extref.getSymbolIndex() == k);
            String name = extref.getName();
            if (name.length() <= 8) {
                write_bytes(out, name, 8);  // s_name
            } else {
                write_ulong(out, 0);    // e_zeroes
                int idx = alloc_string(name);
                write_ulong(out, idx);  // e_offset
            }
            write_ulong(out, 0);
            write_short(out, (short)0);
            write_ushort(out, (char)DT_FCN);
            write_uchar(out, C_EXT);
            write_uchar(out, (byte)0);
            ++k;
        }
    }
    
    public void dumpSFIELDSYMENT(OutputStream out, jq_StaticField sf)
    throws IOException {
        String name = sf.getName().toString();
        if (name.length() <= 8) {
            write_bytes(out, name, 8);  // s_name
        } else {
            write_ulong(out, 0);    // e_zeroes
            int idx = alloc_string(name);
            write_ulong(out, idx);  // e_offset
        }
        int addr = sf.getAddress();
        write_ulong(out, addr);     // e_value
        write_short(out, (short)2); // e_scnum
        jq_Type t = sf.getType();
        char type = (char)0;
        if (t.isArrayType()) {
            t = ((jq_Array)t).getElementType();
            type = DT_ARY;
        } else if (t.isReferenceType()) {
            type = DT_PTR;
        }
        if (t.isPrimitiveType()) {
            if (t == jq_Primitive.INT) type |= T_LONG;
            else if (t == jq_Primitive.LONG) type |= T_LNGDBL;
            else if (t == jq_Primitive.FLOAT) type |= T_FLOAT;
            else if (t == jq_Primitive.DOUBLE) type |= T_DOUBLE;
            else if (t == jq_Primitive.BYTE) type |= T_CHAR;
            else if (t == jq_Primitive.BOOLEAN) type |= T_UCHAR;
            else if (t == jq_Primitive.SHORT) type |= T_SHORT;
            else if (t == jq_Primitive.CHAR) type |= T_USHORT;
            else jq.UNREACHABLE();
        } else {
            type |= T_STRUCT;
        }
        write_ushort(out, type);    // e_type
        write_uchar(out, C_STAT);   // e_sclass
        write_uchar(out, (byte)0);  // e_numaux
    }

    public void dumpIFIELDSYMENT(OutputStream out, jq_InstanceField f)
    throws IOException {
        String name = f.getName().toString();
        if (name.length() <= 8) {
            write_bytes(out, name, 8);  // s_name
        } else {
            write_ulong(out, 0);    // e_zeroes
            int idx = alloc_string(name);
            write_ulong(out, idx);  // e_offset
        }
        int off = f.getOffset();
        write_ulong(out, off);      // e_value
        write_short(out, (short)2); // e_scnum
        jq_Type t = f.getType();
        char type = (char)0;
        if (t.isArrayType()) {
            t = ((jq_Array)t).getElementType();
            type = DT_ARY;
        } else if (t.isReferenceType()) {
            type = DT_PTR;
        }
        if (t.isPrimitiveType()) {
            if (t == jq_Primitive.INT) type |= T_LONG;
            else if (t == jq_Primitive.LONG) type |= T_LNGDBL;
            else if (t == jq_Primitive.FLOAT) type |= T_FLOAT;
            else if (t == jq_Primitive.DOUBLE) type |= T_DOUBLE;
            else if (t == jq_Primitive.BYTE) type |= T_CHAR;
            else if (t == jq_Primitive.BOOLEAN) type |= T_UCHAR;
            else if (t == jq_Primitive.SHORT) type |= T_SHORT;
            else if (t == jq_Primitive.CHAR) type |= T_USHORT;
            else jq.UNREACHABLE();
        } else {
            type |= T_STRUCT;
        }
        write_ushort(out, type);    // e_type
        write_uchar(out, C_MOS);    // e_sclass
        write_uchar(out, (byte)0);  // e_numaux
    }
    
    public void dumpMETHODSYMENT(OutputStream out, jq_Method m)
    throws IOException {
        String name = m.getName().toString();
        if (name.length() <= 8) {
            write_bytes(out, name, 8);  // s_name
        } else {
            write_ulong(out, 0);    // e_zeroes
            int idx = alloc_string(name);
            write_ulong(out, idx);  // e_offset
        }
        int addr = m.getDefaultCompiledVersion().getEntrypoint();
        write_ulong(out, addr);     // e_value
        write_short(out, (short)1); // e_scnum
        write_ushort(out, T_NULL);  // e_type
        write_uchar(out, C_LABEL);  // e_sclass
        write_uchar(out, (byte)0);  // e_numaux
    }
    
    public void addSystemInterfaceRelocs(List extref, List heap2code) {
        jq_StaticField[] fs = SystemInterface._class.getDeclaredStaticFields();
        int total = 3;
        for (int i=0; i<fs.length; ++i) {
            jq_StaticField f = fs[i];
            if (f.isFinal()) continue;
            if (f.getType() != jq_Primitive.INT) continue;
            if (f.getName() == Utf8.get("entry_0")) {
                System.out.println("ENTRY @"+jq.hex8(f.getAddress())+" = "+jq.hex8(SystemInterface.entry_0));
                Heap2CodeReference r = new Heap2CodeReference(f.getAddress(), SystemInterface.entry_0);
                heap2code.add(r);
            } else if (f.getName() == Utf8.get("trap_handler_8")) {
                System.out.println("TRAP_HANDLER @"+jq.hex8(f.getAddress())+" = "+jq.hex8(SystemInterface.trap_handler_8));
                Heap2CodeReference r = new Heap2CodeReference(f.getAddress(), SystemInterface.trap_handler_8);
                heap2code.add(r);
            } else {
                String name = f.getName().toString();
                int ind = name.lastIndexOf('_');
                name = "_"+name.substring(0, ind)+"@"+name.substring(ind+1);
                System.out.println("External ref="+f+", symndx="+(total+1)+" address="+jq.hex8(f.getAddress()));
                ExternalReference r = new ExternalReference(f.getAddress(), name);
                r.setSymbolIndex(++total);
                extref.add(r);
                heap2code.add(r);
            }
        }
        //return total-3;
    }

    public int addVTableRelocs(List list) {
        jq_StaticField[] fs = SystemInterface._class.getDeclaredStaticFields();
        int total = 0;
        Iterator i = jq.boot_types.iterator();
        while (i.hasNext()) {
            jq_Type t = (jq_Type)i.next();
            if (t.isReferenceType()) {
                if (t == Unsafe._class) continue;
                if (TRACE) System.out.println("Adding vtable relocs for: "+t);
                int[] vtable = (int[])((jq_Reference)t).getVTable();
                int/*HeapAddress*/ addr = getAddressOf(vtable);
                jq.assert(vtable[0] != 0, t.toString());
                Heap2HeapReference r1 = new Heap2HeapReference(addr, vtable[0]);
                list.add(r1);
                for (int j=1; j<vtable.length; ++j) {
                    Heap2CodeReference r2 = new Heap2CodeReference(addr+(j*4), vtable[j]);
                    list.add(r2);
                }
                total += vtable.length;
            }
        }
        return total;
    }
    
    public void dumpCOFF(OutputStream out) throws IOException {
        // calculate sizes/offsets
        int textsize = bca.size();
        List text_relocs = bca.getAllDataRelocs();
        List exts = new LinkedList();
        int nlinenum = 0;
        int ntextreloc = text_relocs.size();
        if (ntextreloc > 65535) ++ntextreloc;
        int datastart = 20+40+40+textsize+(10*ntextreloc);
        int datasize = heapCurrent;
        int numOfVTableRelocs = addVTableRelocs(data_relocs);
        addSystemInterfaceRelocs(exts, data_relocs);
        int ndatareloc = data_relocs.size();
        if (ndatareloc > 65535) ++ndatareloc;
        int symtabstart = datastart+datasize+(10*ndatareloc)+(10*nlinenum);
        int nsyms = 2+2+exts.size();
        
        // write file header
        dumpFILHDR(out, symtabstart, nsyms);
        
        // write section headers
        dumpTEXTSCNHDR(out, textsize, ntextreloc);
        dumpDATASCNHDR(out, datastart, datasize, ndatareloc);
        
        Iterator i = bca.getAllCodeRelocs().iterator();
        while (i.hasNext()) {
            Object r = (Object)i.next();
            if (r instanceof DirectBindCall)
                ((DirectBindCall)r).patch();
        }
        
        // write text section
        bca.dump(out);
        
        // write text relocs
        if (ntextreloc > 65535) {
            write_ulong(out, ntextreloc);
            write_ulong(out, 0);
            write_ushort(out, (char)0);
        }
        Iterator it = text_relocs.iterator();
        while (it.hasNext()) {
            Reloc r = (Reloc)it.next();
            r.dump(out);
        }
        out.flush();
        
        // write data section
        dumpHeap(out);
        
        // write data relocs
        if (ndatareloc > 65535) {
            write_ulong(out, ndatareloc);
            write_ulong(out, 0);
            write_ushort(out, (char)0);
        }
        it = data_relocs.iterator();
        while (it.hasNext()) {
            Reloc r = (Reloc)it.next();
            r.dump(out);
        }
        
        // write line numbers
        
        // write symbol table
        dumpSECTIONSYMENTs(out);
        dumpEXTSYMENTs(out);
        dumpEXTDEFSYMENTs(out, exts);
        
        // write string table
        dump_strings(out);
        
        out.flush();
    }
    
    private void dumpHeap(OutputStream out)
    throws IOException {
        jq.assert(ARRAY_LENGTH_OFFSET == -12);
        jq.assert(STATUS_WORD_OFFSET == -8);
        jq.assert(VTABLE_OFFSET == -4);
        jq.assert(OBJ_HEADER_SIZE == 8);
        jq.assert(ARRAY_HEADER_SIZE == 12);
        Iterator i = entries.iterator();
        int currentAddr=0;
        int j=0;
        while (i.hasNext()) {
            Entry e = (Entry)i.next();
            Object o = e.getObject();
            int addr = e.getAddress();
            Class objType = o.getClass();
            jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
            if (TRACE)
                this.out.println("Dumping entry "+j+": "+objType+" "+jq.hex(System.identityHashCode(o))+" addr "+jq.hex(addr));
            if (!jqType.isClsInitialized()) {
                jq.UNREACHABLE(jqType.toString());
                return;
            }
            int vtable = getAddressOf(jqType.getVTable());
            if (jqType.isArrayType()) {
                while (currentAddr+ARRAY_HEADER_SIZE < addr) {
                    write_char(out, (byte)0); ++currentAddr;
                }
                int length = Array.getLength(o);
                write_ulong(out, length);
                write_ulong(out, 0);
                write_ulong(out, vtable);
                currentAddr += 12;
                jq.assert(addr == currentAddr);
                jq_Type elemType = ((jq_Array)jqType).getElementType();
                if (elemType.isPrimitiveType()) {
                    if (elemType == jq_Primitive.INT) {
                        int[] v = (int[])o;
                        for (int k=0; k<length; ++k)
                            write_ulong(out, v[k]);
                        currentAddr += length << 2;
                    } else if (elemType == jq_Primitive.FLOAT) {
                        float[] v = (float[])o;
                        for (int k=0; k<length; ++k)
                            write_ulong(out, Float.floatToRawIntBits(v[k]));
                        currentAddr += length << 2;
                    } else if (elemType == jq_Primitive.LONG) {
                        long[] v = (long[])o;
                        for (int k=0; k<length; ++k)
                            write_ulonglong(out, v[k]);
                        currentAddr += length << 3;
                    } else if (elemType == jq_Primitive.DOUBLE) {
                        double[] v = (double[])o;
                        for (int k=0; k<length; ++k)
                            write_ulonglong(out, Double.doubleToRawLongBits(v[k]));
                        currentAddr += length << 3;
                    } else if (elemType == jq_Primitive.BOOLEAN) {
                        boolean[] v = (boolean[])o;
                        for (int k=0; k<length; ++k)
                            write_uchar(out, v[k]?(byte)1:(byte)0);
                        currentAddr += length;
                    } else if (elemType == jq_Primitive.BYTE) {
                        byte[] v = (byte[])o;
                        for (int k=0; k<length; ++k)
                            write_char(out, v[k]);
                        currentAddr += length;
                    } else if (elemType == jq_Primitive.SHORT) {
                        short[] v = (short[])o;
                        for (int k=0; k<length; ++k)
                            write_short(out, v[k]);
                        currentAddr += length << 1;
                    } else if (elemType == jq_Primitive.CHAR) {
                        char[] v = (char[])o;
                        for (int k=0; k<length; ++k)
                            write_ushort(out, v[k]);
                        currentAddr += length << 1;
                    } else jq.UNREACHABLE();
                } else {
                    Object[] v = (Object[])o;
                    for (int k=0; k<length; ++k)
                        write_ulong(out, getAddressOf(v[k]));
                    currentAddr += length << 2;
                }
            } else {
                jq.assert(jqType.isClassType());
                jq_Class clazz = (jq_Class)jqType;
                while (currentAddr+OBJ_HEADER_SIZE < addr) {
                    write_char(out, (byte)0); ++currentAddr;
                }
                write_ulong(out, 0);
                write_ulong(out, vtable);
                currentAddr += 8;
                jq.assert(addr == currentAddr);
                jq_InstanceField[] fields = clazz.getInstanceFields();
                for (int k=0; k<fields.length; ++k) {
                    jq_InstanceField f = fields[k];
                    jq_Type ftype = f.getType();
                    Object val = obj_trav.getInstanceFieldValue(o, f);
                    int foffset = f.getOffset();
                    if (TRACE) this.out.println("Field "+f+" offset "+jq.shex(foffset)+": "+jq.hex(System.identityHashCode(val)));
                    while (currentAddr != addr+foffset) {
                        write_char(out, (byte)0); ++currentAddr;
                    }
                    if (ftype.isPrimitiveType()) {
                        if (ftype == jq_Primitive.INT)
                            write_ulong(out, (val==null)?0:((Integer)val).intValue());
                        else if (ftype == jq_Primitive.FLOAT)
                            write_ulong(out, (val==null)?0:Float.floatToRawIntBits(((Float)val).floatValue()));
                        else if (ftype == jq_Primitive.LONG)
                            write_ulonglong(out, (val==null)?0L:((Long)val).longValue());
                        else if (ftype == jq_Primitive.DOUBLE)
                            write_ulonglong(out, (val==null)?0L:Double.doubleToRawLongBits(((Double)val).doubleValue()));
                        else if (ftype == jq_Primitive.BOOLEAN)
                            write_uchar(out, (val==null)?(byte)0:((Boolean)val).booleanValue()?(byte)1:(byte)0);
                        else if (ftype == jq_Primitive.BYTE)
                            write_char(out, (val==null)?(byte)0:((Byte)val).byteValue());
                        else if (ftype == jq_Primitive.SHORT)
                            write_short(out, (val==null)?(short)0:((Short)val).shortValue());
                        else if (ftype == jq_Primitive.CHAR)
                            write_ushort(out, (val==null)?(char)0:((Character)val).charValue());
                        else jq.UNREACHABLE();
                    } else {
                        write_ulong(out, getAddressOf(val));
                    }
                    currentAddr += f.getSize();
                }
            }
            ++j;
        }
        while (currentAddr < heapCurrent) {
            write_char(out, (byte)0); ++currentAddr;
        }
    }
    
    public static void write_uchar(OutputStream out, byte v)
    throws IOException {
        out.write((byte)v);
    }
    public static void write_char(OutputStream out, byte v)
    throws IOException {
        out.write((byte)v);
    }
    public static void write_ushort(OutputStream out, char v)
    throws IOException {
        out.write((byte)v);
        out.write((byte)(v>>8));
    }
    public static void write_short(OutputStream out, short v)
    throws IOException {
        out.write((byte)v);
        out.write((byte)(v>>8));
    }
    public static void write_ulong(OutputStream out, int v)
    throws IOException {
        out.write((byte)v);
        out.write((byte)(v>>8));
        out.write((byte)(v>>16));
        out.write((byte)(v>>24));
    }
    public static void write_long(OutputStream out, int v)
    throws IOException {
        out.write((byte)v);
        out.write((byte)(v>>8));
        out.write((byte)(v>>16));
        out.write((byte)(v>>24));
    }
    public static void write_ulonglong(OutputStream out, long v)
    throws IOException {
        out.write((byte)v);
        out.write((byte)(v>>8));
        out.write((byte)(v>>16));
        out.write((byte)(v>>24));
        out.write((byte)(v>>32));
        out.write((byte)(v>>40));
        out.write((byte)(v>>48));
        out.write((byte)(v>>56));
    }
    public static void write_bytes(OutputStream out, String s, int len)
    throws IOException {
        jq.assert(s.length() <= len);
        int i;
        for (i=0; ; ++i) {
            if (i == s.length()) {
                for (; i<len; ++i) {
                    out.write((byte)0);
                }
                return;
            }
            out.write((byte)s.charAt(i));
        }
    }
    
    int stringTableOffset = 4;
    List stringTable = new LinkedList();
    private int alloc_string(String name) {
        int off = stringTableOffset;
        byte[] b = SystemInterface.toCString(name);
        stringTable.add(b);
        stringTableOffset += b.length;
        return off;
    }

    private void dump_strings(OutputStream out)
    throws IOException {
        Iterator i = stringTable.iterator();
        write_ulong(out, stringTableOffset);
        while (i.hasNext()) {
            byte[] b = (byte[])i.next();
            out.write(b);
        }
    }
    
}

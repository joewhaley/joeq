/*
 * jq_Array.java
 *
 * Created on December 19, 2000, 8:45 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import Main.jq;
import Run_Time.Unsafe;
import Allocator.ObjectLayout;
import Allocator.DefaultHeapAllocator;
import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;

public class jq_Array extends jq_Reference implements jq_ClassFileConstants, ObjectLayout {

    public static /*final*/ boolean TRACE = false;
    
    public final boolean isClassType() { return false; }
    public final boolean isArrayType() { return true; }
    public final String getName() {
        return element_type.getName()+"[]";
    }
    public final String shortName() {
        return element_type.shortName()+"[]";
    }
    public final String getJDKName() { return "["+element_type.getJDKDesc(); }
    public final String getJDKDesc() { return getJDKName(); }
    public final byte getLogElementSize() {
        if (element_type == jq_Primitive.LONG ||
            element_type == jq_Primitive.DOUBLE)
            return 3;
        if (element_type == jq_Primitive.CHAR ||
            element_type == jq_Primitive.SHORT)
            return 1;
        if (element_type == jq_Primitive.BYTE)
            return 0;
        return 2;
    }

    public final Object newInstance(int length) {
        load(); verify(); prepare(); sf_initialize(); cls_initialize();
        return DefaultHeapAllocator.allocateArray(length, getInstanceSize(length), vtable);
    }
    
    public final int getDimensionality() {
        if (element_type.isArrayType())
            return 1+((jq_Array)element_type).getDimensionality();
        else
            return 1;
    }
    
    public final boolean isFinal() { return element_type.isFinal(); }
    
    public static final jq_Class[] array_interfaces = new jq_Class[] {
    (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Cloneable;"),
    (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/Serializable;"),
    };
    public final jq_Class[] getInterfaces() {
        return array_interfaces;
    }
    public final jq_Class getInterface(Utf8 desc) {
        chkState(STATE_PREPARED);
        for (int i=0; i<array_interfaces.length; ++i) {
            jq_Class in = array_interfaces[i];
            if (in.getDesc() == desc)
                return in;
        }
        return null;
    }
    public final boolean implementsInterface(jq_Class k) {
        chkState(STATE_PREPARED);
        return k == array_interfaces[0] || k == array_interfaces[1];
    }
    
    public final jq_InstanceMethod getVirtualMethod(jq_NameAndDesc nd) {
        chkState(STATE_PREPARED);
        jq_Class jlo = PrimordialClassLoader.loader.getJavaLangObject();
        return jlo.getVirtualMethod(nd);
    }
    
    public final jq_Type getElementType() { return element_type; }
    
    private jq_Array(Utf8 desc, ClassLoader class_loader, jq_Type element_type) {
        super(desc, class_loader);
        jq.Assert(desc.isDescriptor(TC_ARRAY));
        jq.Assert(element_type != null);
        this.element_type = element_type;
    }
    // ONLY TO BE CALLED BY ClassLoader!!!
    public static jq_Array newArray(Utf8 descriptor, ClassLoader classLoader, jq_Type element_type) {
        return new jq_Array(descriptor, classLoader, element_type);
    }
    
    public static final jq_Array BYTE_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_BYTE), PrimordialClassLoader.loader, jq_Primitive.BYTE);
    public static final jq_Array CHAR_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_CHAR), PrimordialClassLoader.loader, jq_Primitive.CHAR);
    public static final jq_Array DOUBLE_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_DOUBLE), PrimordialClassLoader.loader, jq_Primitive.DOUBLE);
    public static final jq_Array FLOAT_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_FLOAT), PrimordialClassLoader.loader, jq_Primitive.FLOAT);
    public static final jq_Array INT_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_INT), PrimordialClassLoader.loader, jq_Primitive.INT);
    public static final jq_Array LONG_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_LONG), PrimordialClassLoader.loader, jq_Primitive.LONG);
    public static final jq_Array SHORT_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_SHORT), PrimordialClassLoader.loader, jq_Primitive.SHORT);
    public static final jq_Array BOOLEAN_ARRAY =
    new jq_Array(Utf8.get(""+(char)TC_ARRAY+(char)TC_BOOLEAN), PrimordialClassLoader.loader, jq_Primitive.BOOLEAN);
    
    public static jq_Array getPrimitiveArrayType(byte atype) {
        switch(atype) {
            case T_BOOLEAN:
                return BOOLEAN_ARRAY;
            case T_CHAR:
                return CHAR_ARRAY;
            case T_FLOAT:
                return FLOAT_ARRAY;
            case T_DOUBLE:
                return DOUBLE_ARRAY;
            case T_BYTE:
                return BYTE_ARRAY;
            case T_SHORT:
                return SHORT_ARRAY;
            case T_INT:
                return INT_ARRAY;
            case T_LONG:
                return LONG_ARRAY;
            default:
                throw new ClassFormatError();
        }
    }

    public static byte getTypecode(jq_Array array) {
	if (array == BOOLEAN_ARRAY) return T_BOOLEAN;
	if (array == CHAR_ARRAY) return T_CHAR;
	if (array == FLOAT_ARRAY) return T_FLOAT;
	if (array == DOUBLE_ARRAY) return T_DOUBLE;
	if (array == BYTE_ARRAY) return T_BYTE;
	if (array == SHORT_ARRAY) return T_SHORT;
	if (array == INT_ARRAY) return T_INT;
	if (array == LONG_ARRAY) return T_LONG;
	throw new ClassFormatError();
    }

    public final int getInstanceSize(int length) {
        int size = ARRAY_HEADER_SIZE+(length<<getLogElementSize());
        return (size+3) & ~3;
    }
    
    public final jq_Type getInnermostElementType() {
        if (element_type.isArrayType())
            return ((jq_Array)element_type).getInnermostElementType();
        else
            return element_type;
    }
    
    public final void load() {
        if (isLoaded()) return;
        if (TRACE) System.out.println("Loading "+this+"...");
        state = STATE_LOADED;
    }
    public final void verify() {
        if (isVerified()) return;
        if (TRACE) System.out.println("Verifying "+this+"...");
        state = STATE_VERIFIED;
    }
    public final void prepare() {
        if (isPrepared()) return;
        if (TRACE) System.out.println("Preparing "+this+"...");
        state = STATE_PREPARING;
        // vtable is a copy of Ljava/lang/Object;
        jq_Class jlo = PrimordialClassLoader.loader.getJavaLangObject();
        jlo.load(); jlo.verify(); jlo.prepare();
        int[] jlovtable = (int[])jlo.getVTable();
        vtable = new int[jlovtable.length];
        state = STATE_PREPARED;
    }
    public final void sf_initialize() {
        if (isSFInitialized()) return;
        if (TRACE) System.out.println("SF init "+this+"...");
        state = STATE_SFINITIALIZED;
    }
    public final void cls_initialize() {
        if (isClsInitialized()) return;
        if (TRACE) System.out.println("Class init "+this+"...");
        state = STATE_CLSINITIALIZING;
        jq_Class jlo = PrimordialClassLoader.loader.getJavaLangObject();
        jlo.sf_initialize(); jlo.cls_initialize();
        int[] jlovtable = (int[])jlo.getVTable();
        int[] vt = (int[])this.vtable;
        vt[0] = Unsafe.addressOf(this);
        System.arraycopy(jlovtable, 1, vt, 1, jlovtable.length-1);
        state = STATE_CLSINITIALIZED;
    }
    
    public void accept(jq_TypeVisitor tv) {
        tv.visitArray(this);
        super.accept(tv);
    }
    
    private final jq_Type element_type;

    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Array;");
    }
}

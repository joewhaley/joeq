/*
 * jq_Reference.java
 *
 * Created on December 19, 2000, 8:38 AM
 *
 */

package Clazz;

import Allocator.ObjectLayout;
import Bootstrap.PrimordialClassLoader;
import Compil3r.Quad.AndersenInterface.AndersenReference;
import Main.jq;
import Memory.HeapAddress;
import Run_Time.Reflection;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class jq_Reference extends jq_Type implements jq_ClassFileConstants, AndersenReference {

    public static final jq_Reference getTypeOf(Object o) {
        if (!jq.RunningNative) return Reflection.getTypeOf(o);
        return ((HeapAddress)HeapAddress.addressOf(o).offset(ObjectLayout.VTABLE_OFFSET).peek().peek()).asReferenceType();
    }

    public final int getState() { return state; }
    public final boolean isLoaded() { return state >= STATE_LOADED; }
    public final boolean isVerified() { return state >= STATE_VERIFIED; }
    public final boolean isPrepared() { return state >= STATE_PREPARED; }
    public final boolean isSFInitialized() { return state >= STATE_SFINITIALIZED; }
    public final boolean isCompiled() { return state >= STATE_COMPILED; }
    public final boolean isClsInitRunning() { return state >= STATE_CLSINITRUNNING; }
    public final boolean isClsInitialized() { return state >= STATE_CLSINITIALIZED; }
    
    public final boolean isPrimitiveType() { return false; }
    public final boolean isIntLike() { return false; }
    
    public final ClassLoader getClassLoader() { return class_loader; }
    public final int getReferenceSize() { return 4; }
    public final Object getVTable() { chkState(STATE_PREPARED); return vtable; }
    
    public abstract String getJDKName();
    public abstract jq_Class[] getInterfaces();
    public abstract jq_Class getInterface(Utf8 desc);
    public abstract boolean implementsInterface(jq_Class k);

    public abstract jq_InstanceMethod getVirtualMethod(jq_NameAndDesc nd);
    
    public static final int DISPLAY_SIZE = 8;
    
    /** 
     * The first two elements are the positive and negative cache,
     * respectively.  The remainder are the primary supertypes of this type
     * ordered by the tree relation.  This array should be inlined into the
     * jq_Reference object, hopefully.
     * 
     * See paper "Fast subtype checking in the HotSpot JVM".
     */
    protected jq_Reference[] display;
    
    /** 
     * The offset of our type in the display array if this is a primary type, or
     * 0 or 1 if this is a secondary type.
     * 
     * See paper "Fast subtype checking in the HotSpot JVM".
     */
    protected int offset;
    
    /**
     * A reference to the secondary subtype array for this type.
     * 
     * See paper "Fast subtype checking in the HotSpot JVM".
     */
    protected jq_Reference[] s_s_array;
    
    /**
     * The maximum index used in the secondary subtype array.
     * 
     * See paper "Fast subtype checking in the HotSpot JVM".
     */
    protected int s_s_array_length;
    
    public boolean isInstance(Object o) {
        if (o == null) return false;
        jq_Reference that = jq_Reference.getTypeOf(o);
        return that.isSubtypeOf(this);
    }
    
    public static final boolean TRACE = false;
    
    public final boolean isSubtypeOf(jq_Reference that) {
        this.chkState(STATE_PREPARED); that.chkState(STATE_PREPARED);
        
        int off = that.offset;
        if (that == this.display[off]) {
            // matches cache or depth
            if (TRACE) System.out.println(this+" matches "+that+" offset="+off);
            return off != 1;
        }
        if (off > 1) {
            // other class is a primary type that isn't a superclass.
            if (TRACE) System.out.println(this+" doesn't match "+that+", offset "+off+" is "+this.display[off]);
            return false;
        }
        if (this == that) {
            // classes are exactly the same.
            return true;
        }
        int n = this.s_s_array_length;
        for (int i=0; i<n; ++i) {
            if (this.s_s_array[i] == that) {
                this.display[0] = that;
                that.offset = 0;
                if (TRACE) System.out.println(this+" matches "+that+" in s_s_array");
                return true;
            }
        }
        this.display[1] = that;
        that.offset = 1;
        if (TRACE) System.out.println(this+" doesn't match "+that+" in s_s_array");
        return false;
    }
    
    public abstract jq_Reference getDirectPrimarySupertype();
    
    public final void chkState(byte s) {
        if (state >= s) return;
        jq.UNREACHABLE(this+" actual state: "+state+" expected state: "+s);
    }

    protected jq_Reference(Utf8 desc, ClassLoader class_loader) {
        super(desc, class_loader);
        jq.Assert(class_loader != null);
        this.class_loader = class_loader;
    }
    protected final ClassLoader class_loader;
    protected int/*byte*/ state; // use an 'int' so we can do cas4 on it
    protected Object vtable;

    public static class jq_NullType extends jq_Reference {
        private jq_NullType() { super(Utf8.get("L&NULL;"), PrimordialClassLoader.loader); }
        public boolean isAddressType() { return false; }
        public String getJDKName() { return desc.toString(); }
        public String getJDKDesc() { return getJDKName(); }
        public jq_Class[] getInterfaces() { jq.UNREACHABLE(); return null; }
        public jq_Class getInterface(Utf8 desc) { jq.UNREACHABLE(); return null; }
        public boolean implementsInterface(jq_Class k) { jq.UNREACHABLE(); return false; }
        public jq_InstanceMethod getVirtualMethod(jq_NameAndDesc nd) { jq.UNREACHABLE(); return null; }
        public String getName() { jq.UNREACHABLE(); return null; }
        public String shortName() { return "NULL_TYPE"; }
        public boolean isClassType() { jq.UNREACHABLE(); return false; }
        public boolean isArrayType() { jq.UNREACHABLE(); return false; }
        public boolean isFinal() { jq.UNREACHABLE(); return false; }
        public boolean isInstance(Object o) { return o == null; }
        public int getDepth() { jq.UNREACHABLE(); return 0; }
        public jq_Reference getDirectPrimarySupertype() { jq.UNREACHABLE(); return null; }
        public void load() { jq.UNREACHABLE(); }
        public void verify() { jq.UNREACHABLE(); }
        public void prepare() { jq.UNREACHABLE(); }
        public void sf_initialize() { jq.UNREACHABLE(); }
        public void compile() { jq.UNREACHABLE(); }
        public void cls_initialize() { jq.UNREACHABLE(); }
        public String toString() { return "NULL_TYPE"; }
        public static final jq_NullType NULL_TYPE = new jq_NullType();
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _vtable;
    public static /*final*/ jq_InstanceField _state; // set after PrimordialClassLoader finishes initialization
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Reference;");
        _vtable = _class.getOrCreateInstanceField("vtable", "Ljava/lang/Object;");
        // primitive types have not yet been created!
        _state = _class.getOrCreateInstanceField("state", "I");
    }
}

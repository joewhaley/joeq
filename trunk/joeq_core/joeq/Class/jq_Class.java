/*
 * jq_Class.java
 *
 * Created on December 19, 2000, 4:47 AM
 *
 */

package Clazz;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

import Main.jq;
import Allocator.ObjectLayout;
import Allocator.DefaultHeapAllocator;
import Bootstrap.BootstrapRootSet;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Compil3r.BytecodeAnalysis.Bytecodes;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import Run_Time.Reflection;
import UTF.UTFDataFormatError;
import UTF.Utf8;
import Util.LinkedHashMap;
import Synchronization.Atomic;

// friend jq_ClassLoader;

/**
 * @author  John Whaley
 * @version $Id$
 */
public final class jq_Class extends jq_Reference implements jq_ClassFileConstants, ObjectLayout {
    
    public static /*final*/ boolean TRACE = false;
    public static /*final*/ boolean WARN_STALE_CLASS_FILES = false;
    
    /**** INTERFACE ****/
    
    //// Always available
    public final boolean isClassType() { return true; }
    public final boolean isArrayType() { return false; }
    public final String getName() { // fully-qualified name, e.g. java.lang.String
        return className(desc);
    }
    public final String shortName() { // name with package removed
        String s = desc.toString();
        int index = s.lastIndexOf('/')+1;
        if (index == 0) index = 1;
        return s.substring(index, s.length()-1);
    }
    public final boolean isInSamePackage(jq_Class that) {
        if (this.getClassLoader() != that.getClassLoader()) return false;
        String s1 = this.getName();
        String s2 = that.getName();
        int ind1 = s1.lastIndexOf('.');
        int ind2 = s2.lastIndexOf('.');
        if (ind1 != ind2) return false;
        if (ind1 != -1) {
            if (!s1.substring(0, ind1).equals(s2.substring(0, ind1)))
                return false;
        }
        return true;
    }
    public final String getJDKName() { return getName(); }
    public final String getJDKDesc() { return desc.toString().replace('/','.'); }
    public final boolean needsDynamicLink(jq_Method method) {
        if (method.getDeclaringClass() == this) return false;
        if (jq.Bootstrapping && jq.isBootType(this)) return false;
        return !isClsInitialized();
    }
    public jq_Member getDeclaredMember(jq_NameAndDesc nd) {
        return (jq_Member)members.get(nd);
    }
    public jq_Member getDeclaredMember(String name, String desc) {
        return (jq_Member)members.get(new jq_NameAndDesc(Utf8.get(name), Utf8.get(desc)));
    }
    private void addDeclaredMember(jq_NameAndDesc nd, jq_Member m) {
        Object b = members.put(nd, m);
        if (TRACE) {
            SystemInterface.debugmsg("Added member to "+this+": "+m+" (old value "+b+")");
            //new InternalError().printStackTrace();
        }
    }
    public void accept(jq_TypeVisitor tv) {
        tv.visitClass(this);
        super.accept(tv);
    }
    
    //// Available only after loading
    public final int getMinorVersion() { return minor_version; }
    public final int getMajorVersion() { return major_version; }
    public final char getAccessFlags() { return access_flags; }
    public final boolean isPublic() {
        chkState(STATE_LOADING2);
        return (access_flags & ACC_PUBLIC) != 0;
    }
    public final boolean isFinal() {
        chkState(STATE_LOADING2);
        return (access_flags & ACC_FINAL) != 0;
    }
    public final boolean isSpecial() {
        chkState(STATE_LOADING2);
        return (access_flags & ACC_SUPER) != 0;
    }
    public final boolean isInterface() {
        chkState(STATE_LOADING2);
        return (access_flags & ACC_INTERFACE) != 0;
    }
    public final boolean isAbstract() {
        chkState(STATE_LOADING2);
        return (access_flags & ACC_ABSTRACT) != 0;
    }
    public final jq_Class getSuperclass() {
        chkState(STATE_LOADING3);
        return super_class;
    }
    public final jq_Class[] getDeclaredInterfaces() {
        chkState(STATE_LOADING3);
        return declared_interfaces;
    }
    public final jq_Class getDeclaredInterface(Utf8 desc) {
        chkState(STATE_LOADING3);
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class in = declared_interfaces[i];
            if (in.getDesc() == desc)
                return in;
        }
        return null;
    }
    public final jq_InstanceField[] getDeclaredInstanceFields() {
        chkState(STATE_LOADING3);
        return declared_instance_fields;
    }
    public final jq_InstanceField getDeclaredInstanceField(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        return (jq_InstanceField)findByNameAndDesc(declared_instance_fields, nd);
    }
    public final jq_StaticField[] getDeclaredStaticFields() {
        chkState(STATE_LOADING3);
        return static_fields;
    }
    public final jq_StaticField getDeclaredStaticField(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        return (jq_StaticField)findByNameAndDesc(static_fields, nd);
    }
    public final jq_StaticField getStaticField(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        jq_StaticField f = (jq_StaticField)findByNameAndDesc(static_fields, nd);
        if (f != null) return f;
        if (this.isInterface()) {
            // static fields may be in superinterfaces.
            for (int i=0; i<declared_interfaces.length; ++i) {
                jq_Class in = declared_interfaces[i];
                in.load();
                f = in.getStaticField(nd);
                if (f != null) return f;
            }
        }
        // check superclasses.
        if (super_class != null) {
            super_class.load();
            return super_class.getStaticField(nd);
        }
        return null;
    }

    public final int getNumberOfStaticFields() {
        chkState(STATE_LOADED);
        int length = static_fields.length;
        if (this.isInterface()) {
            for (int i=0; i<declared_interfaces.length; ++i) {
                jq_Class in = declared_interfaces[i];
                in.load();
                length += in.getNumberOfStaticFields();
            }
        }
        if (super_class != null) {
            super_class.load();
            length += super_class.getNumberOfStaticFields();
        }
        return length;
    }

    private int getStaticFields_helper(jq_StaticField[] sfs, int current) {
        System.arraycopy(static_fields, 0, sfs, current, static_fields.length);
        current += static_fields.length;
        if (this.isInterface()) {
            for (int i=0; i<declared_interfaces.length; ++i) {
                jq_Class in = declared_interfaces[i];
                current = in.getStaticFields_helper(sfs, current);
            }
        }
        if (super_class != null) {
            current = super_class.getStaticFields_helper(sfs, current);
        }
        return current;
    }

    // NOTE: fields in superinterfaces may appear multiple times.
    public final jq_StaticField[] getStaticFields() {
        chkState(STATE_LOADING3);
        int length = this.getNumberOfStaticFields();
        jq_StaticField[] sfs = new jq_StaticField[length];
        int current = this.getStaticFields_helper(sfs, 0);
        jq.Assert(current == sfs.length);
        return sfs;
    }
    public final jq_InstanceMethod[] getDeclaredInstanceMethods() {
        chkState(STATE_LOADING3);
        return declared_instance_methods;
    }
    public final jq_InstanceMethod getDeclaredInstanceMethod(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        return (jq_InstanceMethod)findByNameAndDesc(declared_instance_methods, nd);
    }
    public final jq_StaticMethod[] getDeclaredStaticMethods() {
        chkState(STATE_LOADING3);
        return static_methods;
    }
    public final jq_StaticMethod getDeclaredStaticMethod(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        return (jq_StaticMethod)findByNameAndDesc(static_methods, nd);
    }
    public final jq_StaticMethod getStaticMethod(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        jq_StaticMethod m = (jq_StaticMethod)findByNameAndDesc(static_methods, nd);
        if (m != null) return m;
        // check superclasses.
        if (super_class != null) {
            super_class.load();
            m = super_class.getStaticMethod(nd);
            if (m != null) return m;
        }
        // static methods may also be in superinterfaces.
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class in = declared_interfaces[i];
            in.load();
            m = in.getStaticMethod(nd);
            if (m != null) return m;
        }
        return null;
    }

    public final int getNumberOfStaticMethods() {
        chkState(STATE_LOADED);
        int length = static_methods.length;
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class in = declared_interfaces[i];
            in.load();
            length += in.getNumberOfStaticMethods();
        }
        if (super_class != null) {
            super_class.load();
            length += super_class.getNumberOfStaticMethods();
        }
        return length;
    }

    private int getStaticMethods_helper(jq_StaticMethod[] sfs, int current) {
        System.arraycopy(static_methods, 0, sfs, current, static_methods.length);
        current += static_methods.length;
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class in = declared_interfaces[i];
            current = in.getStaticMethods_helper(sfs, current);
        }
        if (super_class != null) {
            current = super_class.getStaticMethods_helper(sfs, current);
        }
        return current;
    }

    // NOTE: methods in superinterfaces may appear multiple times.
    public final jq_StaticMethod[] getStaticMethods() {
        chkState(STATE_LOADED);
        int length = this.getNumberOfStaticMethods();
        jq_StaticMethod[] sfs = new jq_StaticMethod[length];
        int current = this.getStaticMethods_helper(sfs, 0);
        jq.Assert(current == sfs.length);
        return sfs;
    }

    public final jq_InstanceMethod getInstanceMethod(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        jq_InstanceMethod m = (jq_InstanceMethod)findByNameAndDesc(declared_instance_methods, nd);
        if (m != null) return m;
        // check superclasses.
        if (super_class != null) {
            super_class.load();
            m = super_class.getInstanceMethod(nd);
            if (m != null) return m;
        }
        // check superinterfaces.
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class in = declared_interfaces[i];
            in.load();
            m = in.getInstanceMethod(nd);
            if (m != null) return m;
        }
        return null;
    }
    public final jq_Initializer getInitializer(Utf8 desc) {
        return getInitializer(new jq_NameAndDesc(Utf8.get("<init>"), desc));
    }
    public final jq_Initializer getInitializer(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        return (jq_Initializer)getDeclaredInstanceMethod(nd);
    }
    public final jq_ClassInitializer getClassInitializer() {
        chkState(STATE_LOADING3);
        return (jq_ClassInitializer)getDeclaredStaticMethod(new jq_NameAndDesc(Utf8.get("<clinit>"), Utf8.get("()V")));
    }
    public final jq_ConstantPool getCP() {
        chkState(STATE_LOADING2);
        return const_pool;
    }
    public final Object getCP(char index) {
        chkState(STATE_LOADING2);
        return const_pool.get(index);
    }
    public final int getCPCount() {
        chkState(STATE_LOADING2);
        return const_pool.getCount();
    }
    public final byte getCPtag(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getTag(index);
    }
    public final Integer getCPasInt(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsInt(index);
    }
    public final Float getCPasFloat(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsFloat(index);
    }
    public final Long getCPasLong(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsLong(index);
    }
    public final Double getCPasDouble(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsDouble(index);
    }
    public final String getCPasString(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsString(index);
    }
    public final Utf8 getCPasUtf8(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsUtf8(index);
    }
    public final jq_Type getCPasType(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsType(index);
    }
    public final jq_Member getCPasMember(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsMember(index);
    }
    public jq_StaticField getOrCreateStaticField(String name, String desc) {
        return getOrCreateStaticField(new jq_NameAndDesc(Utf8.get(name), Utf8.get(desc)));
    }
    public jq_StaticField getOrCreateStaticField(jq_NameAndDesc nd) {
        jq_StaticField sf = (jq_StaticField)getDeclaredMember(nd);
        if (sf != null) return sf;
        return createStaticField(nd);
    }
    jq_StaticField createStaticField(jq_NameAndDesc nd) {
        jq.Assert(getDeclaredMember(nd) == null);
        jq_StaticField f = jq_StaticField.newStaticField(this, nd);
        addDeclaredMember(nd, f);
        return f;
    }
    public final jq_StaticField getCPasStaticField(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsStaticField(index);
    }
    public jq_InstanceField getOrCreateInstanceField(String name, String desc) {
        return getOrCreateInstanceField(new jq_NameAndDesc(Utf8.get(name), Utf8.get(desc)));
    }
    public jq_InstanceField getOrCreateInstanceField(jq_NameAndDesc nd) {
        jq_InstanceField sf = (jq_InstanceField)getDeclaredMember(nd);
        if (sf != null) return sf;
        return createInstanceField(nd);
    }
    jq_InstanceField createInstanceField(jq_NameAndDesc nd) {
        jq.Assert(getDeclaredMember(nd) == null);
        jq_InstanceField f = jq_InstanceField.newInstanceField(this, nd);
        addDeclaredMember(nd, f);
        return f;
    }
    public final jq_InstanceField getCPasInstanceField(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsInstanceField(index);
    }
    public jq_StaticMethod getOrCreateStaticMethod(String name, String desc) {
        return getOrCreateStaticMethod(new jq_NameAndDesc(Utf8.get(name), Utf8.get(desc)));
    }
    public jq_StaticMethod getOrCreateStaticMethod(jq_NameAndDesc nd) {
        jq_StaticMethod sf = (jq_StaticMethod)getDeclaredMember(nd);
        if (sf != null) return sf;
        return createStaticMethod(nd);
    }
    jq_StaticMethod createStaticMethod(jq_NameAndDesc nd) {
        jq.Assert(getDeclaredMember(nd) == null);
        jq_StaticMethod f;
        if (nd.getName() == Utf8.get("<clinit>") &&
            nd.getDesc() == Utf8.get("()V")) {
            f = jq_ClassInitializer.newClassInitializer(this, nd);
        } else {
            f = jq_StaticMethod.newStaticMethod(this, nd);
        }
        addDeclaredMember(nd, f);
        return f;
    }
    public final jq_StaticMethod getCPasStaticMethod(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsStaticMethod(index);
    }
    public jq_InstanceMethod getOrCreateInstanceMethod(String name, String desc) {
        return getOrCreateInstanceMethod(new jq_NameAndDesc(Utf8.get(name), Utf8.get(desc)));
    }
    public jq_InstanceMethod getOrCreateInstanceMethod(jq_NameAndDesc nd) {
        jq_InstanceMethod sf = (jq_InstanceMethod)getDeclaredMember(nd);
        if (sf != null) return sf;
        return createInstanceMethod(nd);
    }
    jq_InstanceMethod createInstanceMethod(jq_NameAndDesc nd) {
        jq.Assert(getDeclaredMember(nd) == null);
        jq_InstanceMethod f;
        if (nd.getName() == Utf8.get("<init>")) {
            f = jq_Initializer.newInitializer(this, nd);
        } else {
            f = jq_InstanceMethod.newInstanceMethod(this, nd);
        }
        addDeclaredMember(nd, f);
        return f;
    }
    public final jq_InstanceMethod getCPasInstanceMethod(char index) {
        chkState(STATE_LOADING2);
        return const_pool.getAsInstanceMethod(index);
    }
    public final byte[] getAttribute(Utf8 name) {
        chkState(STATE_LOADING3);
        return (byte[])attributes.get(name);
    }
    public final byte[] getAttribute(String name) {
        chkState(STATE_LOADING3);
        return getAttribute(Utf8.get(name));
    }
    public final Iterator getAttributes() {
        return attributes.keySet().iterator();
    }
    public final Utf8 getSourceFile() {
        chkState(STATE_LOADING3);
        byte[] attrib = getAttribute("SourceFile");
        if (attrib == null) return null;
        if (attrib.length != 2)
            throw new ClassFormatError();
        char cpi = jq.twoBytesToChar(attrib, 0);
        if (getCPtag(cpi) != CONSTANT_Utf8)
            throw new ClassFormatError("cp tag "+(int)cpi+" is "+(int)getCPtag(cpi));
        return getCPasUtf8(cpi);
    }
    public final boolean isSynthetic() {
        chkState(STATE_LOADING3);
        return getAttribute("Synthetic") != null;
    }
    public final boolean isDeprecated() {
        chkState(STATE_LOADING3);
        return getAttribute("Deprecated") != null;
    }
    public final jq_Class[] getInnerClasses() {
        chkState(STATE_LOADING3);
        jq.TODO();
        return null;
    }
    public final jq_Class[] getSubClasses() {
        chkState(STATE_LOADING3);
        return subclasses;
    }
    public final jq_Class[] getSubInterfaces() {
        chkState(STATE_LOADING3);
        return subinterfaces;
    }
   //// Available after resolving
    public final jq_Class[] getInterfaces() {
        chkState(STATE_PREPARED);
        return interfaces;
    }
    public final jq_Class getInterface(Utf8 desc) {
        chkState(STATE_PREPARED);
        for (int i=0; i<interfaces.length; ++i) {
            jq_Class in = interfaces[i];
            if (in.getDesc() == desc)
                return in;
        }
        return null;
    }
    public final boolean implementsInterface(jq_Class k) {
        chkState(STATE_PREPARED);
        for (int i=0; i<interfaces.length; ++i) {
            if (interfaces[i] == k)
                return true;
        }
        for (int i=0; i<interfaces.length; ++i) {
            jq_Class k2 = interfaces[i];
            k2.load(); k2.verify(); k2.prepare();
            if (k2.implementsInterface(k))
                return true;
        }
        return false;
    }
    public final jq_InstanceField[] getInstanceFields() {
        chkState(STATE_PREPARED);
        return instance_fields;
    }
    public final jq_InstanceField getInstanceField(jq_NameAndDesc nd) {
        chkState(STATE_LOADING3);
        jq_InstanceField m = (jq_InstanceField)findByNameAndDesc(declared_instance_fields, nd);
        if (m != null) return m;
        // check superclasses.
        if (super_class != null) {
            super_class.load();
            m = super_class.getInstanceField(nd);
            if (m != null) return m;
        }
        return null;
    }
    public final jq_InstanceMethod[] getVirtualMethods() {
        chkState(STATE_PREPARED);
        return virtual_methods;
    }
    public final jq_InstanceMethod getVirtualMethod(jq_NameAndDesc nd) {
        chkState(STATE_PREPARED);
        return (jq_InstanceMethod)findByNameAndDesc(virtual_methods, nd);
    }
    public final int getInstanceSize() {
        chkState(STATE_PREPARED);
        return instance_size;
    }
    
    public final void setStaticData(jq_StaticField sf, int data) {
        chkState(STATE_SFINITIALIZED);
        jq.Assert(sf.getDeclaringClass() == this);
        jq.Assert(sf.getType().getReferenceSize() != 8);
        int index = (sf.getAddress() - Unsafe.addressOf(static_data)) >> 2;
        if (index < 0 || index >= static_data.length) {
            jq.UNREACHABLE("sf: "+sf+" index: "+index);
        }
        static_data[index] = data;
    }
    public final void setStaticData(jq_StaticField sf, long data) {
        chkState(STATE_SFINITIALIZED);
        jq.Assert(sf.getDeclaringClass() == this);
        jq.Assert(sf.getType().getReferenceSize() == 8);
        int index = (sf.getAddress() - Unsafe.addressOf(static_data)) >> 2;
        static_data[index  ] = (int)(data);
        static_data[index+1] = (int)(data >> 32);
    }
    public final void setStaticData(jq_StaticField sf, float data) {
        setStaticData(sf, Float.floatToRawIntBits(data));
    }
    public final void setStaticData(jq_StaticField sf, double data) {
        setStaticData(sf, Double.doubleToRawLongBits(data));
    }
    public final void setStaticData(jq_StaticField sf, Object data) {
        chkState(STATE_SFINITIALIZED);
        jq.Assert(sf.getDeclaringClass() == this);
        jq.Assert(sf.getType().getReferenceSize() != 8);
        int index = (sf.getAddress() - Unsafe.addressOf(static_data)) >> 2;
        static_data[index] = Unsafe.addressOf(data);
    }

    public final Object newInstance() {
        load(); verify(); prepare(); sf_initialize(); cls_initialize();
        return DefaultHeapAllocator.allocateObject(instance_size, vtable);
    }
    
    //// Implementation garbage.
    private Map/*<jq_NameAndDesc->jq_Member>*/ members;
    private char minor_version;
    private char major_version;
    private char access_flags;
    private jq_Class super_class;
    private jq_Class[] subclasses, subinterfaces;
    private jq_Class[] declared_interfaces, interfaces;
    private jq_StaticField[] static_fields;
    private jq_StaticMethod[] static_methods;
    private jq_InstanceField[] declared_instance_fields;
    private jq_InstanceMethod[] declared_instance_methods;
    private Map attributes;
    private jq_ConstantPool const_pool;
    private int static_data_size;
    private int instance_size;
    private jq_InstanceField[] instance_fields;
    private jq_InstanceMethod[] virtual_methods;
    private int[] static_data;
    private boolean dont_align;

    private static jq_Member findByNameAndDesc(jq_Member[] array, jq_NameAndDesc nd) {
        // linear search
        for (int i=0; i<array.length; ++i) {
            jq_Member m = array[i];
            if (m.getNameAndDesc().equals(nd)) return m;
        }
        return null;
    }
    
    /**
     * Private constructor.
     * Use a ClassLoader to create a jq_Class instance.
     */
    private jq_Class(ClassLoader class_loader, Utf8 desc) {
        super(desc, class_loader);
        this.subclasses = new jq_Class[0];
        this.subinterfaces = new jq_Class[0];
        this.members = new HashMap();
    }
    // ONLY TO BE CALLED BY ClassLoader!!!
    public static jq_Class newClass(ClassLoader classLoader, Utf8 desc) {
        jq.Assert(desc.isDescriptor(TC_CLASS));
        return new jq_Class(classLoader, desc);
    }

    /**
     * Loads the binary data for this class.  See Jvm spec 2.17.2.
     *
     * Throws: ClassFormatError  if the binary data is malformed in some way
     *         UnsupportedClassVersionError  if the binary data is of an unsupported version
     *         ClassCircularityError  if it would be its own superclass or superinterface
     *         NoClassDefFoundError  if the class definition cannot be found
     */
    public void load()
    throws ClassFormatError, UnsupportedClassVersionError, ClassCircularityError, NoClassDefFoundError {
        if (isLoaded()) return; // quick test
        jq.Assert(class_loader == PrimordialClassLoader.loader);
        try {
            DataInputStream in = ((PrimordialClassLoader)class_loader).getClassFileStream(desc);
            if (in == null) throw new NoClassDefFoundError(className(desc));
            load(in);
            in.close();
        } catch (IOException x) {
            x.printStackTrace(); // for debugging
            throw new ClassFormatError(x.toString());
        }
    }
    public void load(DataInput in)
    throws ClassFormatError, UnsupportedClassVersionError, ClassCircularityError {
        if (isLoaded()) return; // quick test.
        synchronized (this) {
            if (isLoaded()) return; // other thread already loaded this type.
            if ((state == STATE_LOADING1) || (state == STATE_LOADING2) || (state == STATE_LOADING3))
                throw new ClassCircularityError(this.toString()); // recursively called load (?)
            state = STATE_LOADING1;
            if (TRACE) SystemInterface.debugmsg("Beginning loading "+this+"...");
            try {
                int magicNum = in.readInt(); // 0xCAFEBABE
                if (magicNum != 0xCAFEBABE)
                    throw new ClassFormatError("bad magic number: "+Integer.toHexString(magicNum));
                minor_version = (char)in.readUnsignedShort(); // 3 or 0
                major_version = (char)in.readUnsignedShort(); // 45 or 46
                if (((major_version != 45) || (minor_version != 0)) &&
                    ((major_version != 45) || (minor_version != 3)) &&
                    ((major_version != 46) || (minor_version != 0)) &&
                    ((major_version != 48) || (minor_version != 0))) {
                    throw new UnsupportedClassVersionError("unsupported version "+(int)major_version+"."+(int)minor_version);
                }

                char constant_pool_count = (char)in.readUnsignedShort();
                const_pool = new jq_ConstantPool(constant_pool_count);
                // read in the constant pool
                const_pool.load(in);
                // resolve the non-primitive stuff
                try {
                    const_pool.resolve(class_loader);
                } catch (NoSuchMethodError x) {
                    throw new NoSuchMethodError("In class "+this+": "+x.getMessage());
                } catch (NoSuchFieldError x) {
                    throw new NoSuchFieldError("In class "+this+": "+x.getMessage());
                }
                
                access_flags = (char)in.readUnsignedShort();
                state = STATE_LOADING2;
                char selfindex = (char)in.readUnsignedShort();
                if (getCPtag(selfindex) != CONSTANT_ResolvedClass) {
                    throw new ClassFormatError("constant pool entry "+(int)selfindex+", referred to by field this_class" +
                                               ", is wrong type tag (expected="+CONSTANT_Class+", actual="+getCPtag(selfindex)+")");
                }
                if (getCP(selfindex) != this) {
                    throw new ClassFormatError("expected class "+this+" but found class "+getCP(selfindex));
                }
                char superindex = (char)in.readUnsignedShort();
                if (superindex != 0) {
                    if (getCPtag(superindex) != CONSTANT_ResolvedClass) {
                        throw new ClassFormatError("constant pool entry "+(int)superindex+", referred to by field super_class" +
                                                   ", is wrong type tag (expected="+CONSTANT_Class+", actual="+getCPtag(superindex)+")");
                    }
                    jq_Type super_type = getCPasType(superindex);
                    if (!super_type.isClassType()) {
                        throw new ClassFormatError("superclass ("+super_class.getName()+") is not a class type");
                    }
                    if (super_type == this) {
                        throw new ClassCircularityError(this.getName()+" has itself as a superclass!");
                    }
                    super_class = (jq_Class)super_type;
                    super_class.addSubclass(this);
                } else {
                    // no superclass --> java.lang.Object
                    if (PrimordialClassLoader.loader.getJavaLangObject() != this) {
                        throw new ClassFormatError("no superclass listed for class "+this);
                    }
                }
                char n_interfaces = (char)in.readUnsignedShort();
                declared_interfaces = new jq_Class[n_interfaces];
                for (int i=0; i<n_interfaces; ++i) {
                    char interface_index = (char)in.readUnsignedShort();
                    if (getCPtag(interface_index) != CONSTANT_ResolvedClass) {
                        throw new ClassFormatError("constant pool entry "+(int)interface_index+", referred to by interfaces["+i+"]"+
                                                   ", is wrong type tag (expected="+CONSTANT_Class+", actual="+getCPtag(interface_index)+")");
                    }
                    declared_interfaces[i] = (jq_Class)getCPasType(interface_index);
                    if (!declared_interfaces[i].isClassType()) {
                        throw new ClassFormatError("implemented interface ("+super_class.getName()+") is not a class type");
                    }
                    if (declared_interfaces[i].isLoaded() && !declared_interfaces[i].isInterface()) {
                        throw new ClassFormatError("implemented interface ("+super_class.getName()+") is not an interface type");
                    }
                    if (declared_interfaces[i] == jq_DontAlign._class) dont_align = true;
                    declared_interfaces[i].addSubinterface(this);
                }

                char n_declared_fields = (char)in.readUnsignedShort();
                char[] temp_declared_field_flags = new char[n_declared_fields];
                jq_Field[] temp_declared_fields = new jq_Field[n_declared_fields];
                int numStaticFields = 0, numInstanceFields = 0;
                for (int i=0; i<n_declared_fields; ++i) {
                    temp_declared_field_flags[i] = (char)in.readUnsignedShort();
                    // TODO: check flags for validity.
                    char field_name_index = (char)in.readUnsignedShort();
                    if (getCPtag(field_name_index) != CONSTANT_Utf8)
                        throw new ClassFormatError("constant pool entry "+(int)field_name_index+", referred to by field "+i+
                                                   ", is wrong type tag (expected="+CONSTANT_Utf8+", actual="+getCPtag(field_name_index)+")");
                    Utf8 field_name = getCPasUtf8(field_name_index);
                    char field_desc_index = (char)in.readUnsignedShort();
                    if (getCPtag(field_desc_index) != CONSTANT_Utf8)
                        throw new ClassFormatError("constant pool entry "+(int)field_desc_index+", referred to by field "+i+
                                                   ", is wrong type tag (expected="+CONSTANT_Utf8+", actual="+getCPtag(field_desc_index)+")");
                    Utf8 field_desc = getCPasUtf8(field_desc_index);
                    if (!field_desc.isValidTypeDescriptor())
                        throw new ClassFormatError(field_desc+" is not a valid type descriptor");
                    jq_NameAndDesc nd = new jq_NameAndDesc(field_name, field_desc);
                    jq_Field field = (jq_Field)getDeclaredMember(nd);
                    if ((temp_declared_field_flags[i] & ACC_STATIC) != 0) {
                        if (field == null) {
                            field = createStaticField(nd);
                        } else if (!field.isStatic())
                            throw new VerifyError("static field "+field+" was referred to as an instance field");
                        ++numStaticFields;
                    } else {
                        if (field == null) {
                            field = createInstanceField(nd);
                        } else if (field.isStatic())
                            throw new VerifyError("instance field "+field+" was referred to as a static field");
                        ++numInstanceFields;
                    }
                    field.load(temp_declared_field_flags[i], in);
                    temp_declared_fields[i] = field;
                }
                static_data_size = 0;
                declared_instance_fields = new jq_InstanceField[numInstanceFields];
                static_fields = new jq_StaticField[numStaticFields];
                for (int i=0, di=-1, si=-1; i<n_declared_fields; ++i) {
                    if ((temp_declared_field_flags[i] & ACC_STATIC) != 0) {
                        static_fields[++si] = (jq_StaticField)temp_declared_fields[i];
                        static_data_size += static_fields[si].getWidth();
                    } else {
                        declared_instance_fields[++di] = (jq_InstanceField)temp_declared_fields[i];
                    }
                }
                if (!dont_align) {
                    // sort instance fields in reverse by their size.
                    Arrays.sort(declared_instance_fields, new Comparator() {
                        public int compare(jq_InstanceField o1, jq_InstanceField o2) {
                            int s1 = o1.getSize(), s2 = o2.getSize();
                            if (s1 > s2) return -1;
                            else if (s1 < s2) return 1;
                            else return 0;
                        }
                        public int compare(Object o1, Object o2) {
                            return compare((jq_InstanceField)o1, (jq_InstanceField)o2);
                        }
                    });
                }

                char n_declared_methods = (char)in.readUnsignedShort();
                char[] temp_declared_method_flags = new char[n_declared_methods];
                jq_Method[] temp_declared_methods = new jq_Method[n_declared_methods];
                int numStaticMethods = 0, numInstanceMethods = 0;
                for (int i=0; i<n_declared_methods; ++i) {
                    temp_declared_method_flags[i] = (char)in.readUnsignedShort();
                    // TODO: check flags for validity.
                    char method_name_index = (char)in.readUnsignedShort();
                    if (getCPtag(method_name_index) != CONSTANT_Utf8)
                        throw new ClassFormatError("constant pool entry "+(int)method_name_index+", referred to by method "+i+
                                                   ", is wrong type tag (expected="+CONSTANT_Utf8+", actual="+getCPtag(method_name_index)+")");
                    Utf8 method_name = getCPasUtf8(method_name_index);
                    char method_desc_index = (char)in.readUnsignedShort();
                    if (getCPtag(method_desc_index) != CONSTANT_Utf8)
                        throw new ClassFormatError("constant pool entry "+(int)method_desc_index+", referred to by method "+i+
                                                   ", is wrong type tag (expected="+CONSTANT_Utf8+", actual="+getCPtag(method_desc_index)+")");
                    Utf8 method_desc = getCPasUtf8(method_desc_index);
                    if (!method_desc.isValidMethodDescriptor())
                        throw new ClassFormatError(method_desc+" is not a valid method descriptor");
                    jq_NameAndDesc nd = new jq_NameAndDesc(method_name, method_desc);
                    jq_Method method = (jq_Method)getDeclaredMember(nd);
                    if ((temp_declared_method_flags[i] & ACC_STATIC) != 0) {
                        if (method == null) {
                            method = createStaticMethod(nd);
                        } else if (!method.isStatic())
                            throw new VerifyError();
                        ++numStaticMethods;
                    } else {
                        if (method == null) {
                            method = createInstanceMethod(nd);
                        } else if (method.isStatic())
                            throw new VerifyError();
                        ++numInstanceMethods;
                    }
                    method.load(temp_declared_method_flags[i], in);
                    temp_declared_methods[i] = method;
                }
                declared_instance_methods = new jq_InstanceMethod[numInstanceMethods];
                static_methods = new jq_StaticMethod[numStaticMethods];
                for (int i=0, di=-1, si=-1; i<n_declared_methods; ++i) {
                    if ((temp_declared_method_flags[i] & ACC_STATIC) != 0) {
                        static_methods[++si] = (jq_StaticMethod)temp_declared_methods[i];
                    } else {
                        declared_instance_methods[++di] = (jq_InstanceMethod)temp_declared_methods[i];
                    }
                }
                // now read class attributes
                attributes = new HashMap();
                readAttributes(in, attributes);

                state = STATE_LOADING3;
                
                // if this is a class library, look for and load our mirror (implementation) class
                Iterator impls = ClassLibInterface.i.getImplementationClassDescs(getDesc());
                while (impls.hasNext()) {
                    Utf8 impl_utf = (Utf8)impls.next();
                    jq_Class mirrorclass = (jq_Class)ClassLibInterface.i.getOrCreateType(class_loader, impl_utf);
                    try {
                        if (TRACE) SystemInterface.debugmsg("Attempting to load mirror class "+mirrorclass);
                        mirrorclass.load();
                    } catch (NoClassDefFoundError x) {
                        // no mirror class
                        ClassLibInterface.i.unloadType(class_loader, mirrorclass);
                        mirrorclass = null;
                    }
                    if (mirrorclass != null) {
                        this.merge(mirrorclass);
                    }
                }
                
                // if this is in the class library, remap method bodies.
                if (this.getDesc().toString().startsWith("LClassLib/")) {
                    //(this.super_class != null && this.super_class.getDesc() == Utf8.get("LClassLib/ClassLibInterface;")) {
                    if (TRACE) SystemInterface.debugmsg(this+" is in the class library, rewriting method bodies.");
                    final jq_ConstantPool.ConstantPoolRebuilder cpr = this.rebuildConstantPool(false);
                    // visit instance fields
                    for (int i=0; i<this.declared_instance_fields.length; ++i) {
                        jq_InstanceField this_m = this.declared_instance_fields[i];
                        jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(this, this_m.getNameAndDesc());
                        if (this_m.getNameAndDesc() != nd) {
                            if (TRACE) SystemInterface.debugmsg("Rewriting field signature from "+this_m.getNameAndDesc()+" to "+nd);
                            jq_InstanceField this_m2 = getOrCreateInstanceField(nd);
                            this_m2.load(this_m);
                            this_m.unload(); Object b = this.members.remove(this_m.getNameAndDesc()); cpr.remove(this_m);
                            if (TRACE) SystemInterface.debugmsg("Removed member "+this_m.getNameAndDesc()+" from member set of "+this+": "+b);
                            this.addDeclaredMember(nd, this_m2);
                            this_m = declared_instance_fields[i] = this_m2;
                        }
                    }
                    // visit static fields
                    for (int i=0; i<this.static_fields.length; ++i) {
                        jq_StaticField this_m = this.static_fields[i];
                        jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(this, this_m.getNameAndDesc());
                        if (this_m.getNameAndDesc() != nd) {
                            if (TRACE) SystemInterface.debugmsg("Rewriting field signature from "+this_m.getNameAndDesc()+" to "+nd);
                            jq_StaticField this_m2 = getOrCreateStaticField(nd);
                            this_m2.load(this_m);
                            this_m.unload(); Object b = this.members.remove(this_m.getNameAndDesc()); cpr.remove(this_m);
                            if (TRACE) SystemInterface.debugmsg("Removed member "+this_m.getNameAndDesc()+" from member set of "+this+": "+b);
                            this.addDeclaredMember(nd, this_m2);
                            this_m = static_fields[i] = this_m2;
                        }
                    }
                    // visit all instance methods.
                    LinkedHashMap newInstanceMethods = new LinkedHashMap();
                    for (int i=0; i<this.declared_instance_methods.length; ++i) {
                        jq_InstanceMethod this_m = this.declared_instance_methods[i];
                        jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(this, this_m.getNameAndDesc());
                        if (this_m.getNameAndDesc() != nd) {
                            if (TRACE) SystemInterface.debugmsg("Rewriting method signature from "+this_m.getNameAndDesc()+" to "+nd);
                            jq_InstanceMethod this_m2 = getOrCreateInstanceMethod(nd);
                            this_m2.load(this_m);
                            this_m.unload(); Object b = this.members.remove(this_m.getNameAndDesc()); cpr.remove(this_m);
                            if (TRACE) SystemInterface.debugmsg("Removed member "+this_m.getNameAndDesc()+" from member set of "+this+": "+b);
                            this.addDeclaredMember(nd, this_m2);
                            this_m = this_m2;
                        }
                        byte[] bc = this_m.getBytecode();
                        Bytecodes.InstructionList il;
                        if (bc == null) {
                            il = null;
                        } else {
                            // extract instructions of method.
                            il = new Bytecodes.InstructionList(this_m);

                            // update constant pool references in the instructions, and add them to our constant pool.
                            rewriteMethod(cpr, il);
                        }
                        
                        // cache the instruction list for later.
                        newInstanceMethods.put(this_m, il);
                    }
                    // visit all static methods.
                    LinkedHashMap newStaticMethods = new LinkedHashMap();
                    for (int i=0; i<this.static_methods.length; ++i) {
                        jq_StaticMethod this_m = this.static_methods[i];
                        jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(this, this_m.getNameAndDesc());
                        if (this_m.getNameAndDesc() != nd) {
                            if (TRACE) SystemInterface.debugmsg("Rewriting method signature from "+this_m.getNameAndDesc()+" to "+nd);
                            jq_StaticMethod this_m2 = getOrCreateStaticMethod(nd);
                            this_m2.load(this_m);
                            this_m.unload(); Object b = this.members.remove(this_m.getNameAndDesc()); cpr.remove(this_m);
                            if (TRACE) SystemInterface.debugmsg("Removed member "+this_m.getNameAndDesc()+" from member set of "+this+": "+b);
                            this.addDeclaredMember(nd, this_m2);
                            this_m = this_m2;
                        }
                        byte[] bc = this_m.getBytecode();
                        Bytecodes.InstructionList il;
                        if (bc == null) {
                            il = null;
                        } else {
                            // extract instructions of method.
                            il = new Bytecodes.InstructionList(this_m);

                            // update constant pool references in the instructions, and add them to our constant pool.
                            rewriteMethod(cpr, il);
                        }
                        
                        // cache the instruction list for later.
                        newStaticMethods.put(this_m, il);
                    }
                    jq_ConstantPool new_cp = cpr.finish();
                    // rebuild method arrays
                    this.declared_instance_methods = new jq_InstanceMethod[newInstanceMethods.size()];
                    int j = -1;
                    for (Iterator i=newInstanceMethods.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        jq_InstanceMethod i_m = (jq_InstanceMethod)e.getKey();
                        Bytecodes.InstructionList i_l = (Bytecodes.InstructionList)e.getValue();
                        if (i_l != null) {
                            if (TRACE) SystemInterface.debugmsg("Rebuilding bytecodes for instance method "+i_m+", entry "+(j+1));
                            i_m.setCode(i_l, cpr);
                        } else {
                            if (TRACE) SystemInterface.debugmsg("No bytecodes for instance method "+i_m+", entry "+(j+1));
                        }
                        //if (TRACE) SystemInterface.debugmsg("Adding instance method "+i_m+" to array.");
                        this.declared_instance_methods[++j] = i_m;
                    }
                    this.static_methods = new jq_StaticMethod[newStaticMethods.size()];
                    j = -1;
                    for (Iterator i=newStaticMethods.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        jq_StaticMethod i_m = (jq_StaticMethod)e.getKey();
                        Bytecodes.InstructionList i_l = (Bytecodes.InstructionList)e.getValue();
                        if (i_l != null) {
                            if (TRACE) SystemInterface.debugmsg("Rebuilding bytecodes for static method "+i_m+", entry "+(j+1));
                            i_m.setCode(i_l, cpr);
                        } else {
                            if (TRACE) SystemInterface.debugmsg("No bytecodes for static method "+i_m+", entry "+(j+1));
                        }
                        //if (TRACE) SystemInterface.debugmsg("Adding static method "+i_m+" to array.");
                        this.static_methods[++j] = i_m;
                    }
                    this.remakeAttributes(cpr);
                    this.const_pool = new_cp;
                    getSourceFile(); // check for bug.
                    if (TRACE) SystemInterface.debugmsg("Finished rebuilding constant pool.");
                } else {
                    
                    // make sure that all member references from other classes point to actual members.
                    Iterator it = members.entrySet().iterator();
                    while (it.hasNext()) {
                        Map.Entry e = (Map.Entry)it.next();
                        jq_Member m = (jq_Member)e.getValue();
                        if (m.getState() < STATE_LOADED) {
                            // may be a reference to a member of a superclass or superinterface.
                            // this can happen when using old class files.
                            it.remove();
                            if (WARN_STALE_CLASS_FILES) {
                                Set s = PrimordialClassLoader.loader.getClassesThatReference(m);
                                System.err.println("Warning: classes "+s+" refer to member "+m+", which does not exist. This may indicate stale class files.");
                            }
                            //throw new ClassFormatError("no such member "+m+", referenced by "+s);
                        }
                    }
                    
                }

                // all done!
                if (TRACE) SystemInterface.debugmsg("Finished loading "+this);
                state = STATE_LOADED;
            }
            catch (UTFDataFormatError x) {
                //state = STATE_LOADERROR;
                throw new ClassFormatError(x.toString());
            }
            catch (IOException x) {
                //state = STATE_LOADERROR;
                throw new ClassFormatError(x.toString());
            }
            catch (ArrayIndexOutOfBoundsException x) {
                //state = STATE_LOADERROR;
                x.printStackTrace();
                throw new ClassFormatError("bad constant pool index");
            }
        } // synchronized
    }
    
    public boolean doesConstantPoolContain(Object o) {
        if (const_pool == null) return false;
        return const_pool.contains(o);
    }
    
    public jq_StaticMethod generateStaticMethodStub(jq_NameAndDesc nd, jq_StaticMethod m, char access_flags, char classfield_idx, char method_idx) {
        jq_Type[] params = m.getParamTypes();
        jq.Assert(params.length >= 1);
        int size = 3+((params.length-1)*2)+3+1;
        byte[] bc = new byte[size];
        bc[0] = (byte)0xb2; // getstatic
        bc[1] = (byte)(classfield_idx >> 8);
        bc[2] = (byte)classfield_idx;
        int k=2;
        for (int j=1, n=0; j<params.length; ++j, ++n) {
            if (params[j].isReferenceType()) {
                bc[++k] = (byte)0x19; // aload
            } else if (params[j] == jq_Primitive.LONG) {
                bc[++k] = (byte)0x16; // lload
            } else if (params[j] == jq_Primitive.FLOAT) {
                bc[++k] = (byte)0x17; // fload
            } else if (params[j] == jq_Primitive.DOUBLE) {
                bc[++k] = (byte)0x18; // dload
            } else {
                bc[++k] = (byte)0x15; // iload
            }
            bc[++k] = (byte)n;
            if ((params[j] == jq_Primitive.LONG) || (params[j] == jq_Primitive.DOUBLE))
                ++n;
        }
        bc[++k] = (byte)0xb8; // invokestatic
        bc[++k] = (byte)(method_idx>>8);
        bc[++k] = (byte)method_idx;
        jq_Type t = m.getReturnType();
        if (t.isReferenceType()) {
            bc[++k] = (byte)0xb0; // areturn
        } else if (t == jq_Primitive.LONG) {
            bc[++k] = (byte)0xad; // lreturn
        } else if (t == jq_Primitive.FLOAT) {
            bc[++k] = (byte)0xae; // freturn
        } else if (t == jq_Primitive.DOUBLE) {
            bc[++k] = (byte)0xaf; // dreturn
        } else if (t == jq_Primitive.VOID) {
            bc[++k] = (byte)0xb1; // return
        } else {
            bc[++k] = (byte)0xac; // ireturn
        }
        jq_Method stubm = (jq_Method)getDeclaredMember(nd);
        jq_StaticMethod stub;
        if (stubm == null) stub = jq_StaticMethod.newStaticMethod(this, nd);
        else {
            // method that we are overwriting must be static.
            jq.Assert(stubm.isStatic(), stubm.toString());
            stub = (jq_StaticMethod)stubm;
        }
        //char access_flags = (char)(m.getAccessFlags() & ~ACC_NATIVE);
        char max_stack = (char)Math.max(m.getParamWords(), m.getReturnType().getReferenceSize()>>2);
        char max_locals = (char)(m.getParamWords()-1);
        stub.load(access_flags, max_stack, max_locals, bc, new jq_TryCatchBC[0], new jq_LineNumberBC[0], new HashMap());
        return stub;
    }
    
    public jq_InstanceMethod generateInstanceMethodStub(jq_NameAndDesc nd, jq_StaticMethod m, char access_flags, char method_idx) {
        jq_Type[] params = m.getParamTypes();
        jq.Assert(params.length >= 1);
        int size = 1+((params.length-1)*2)+3+1;
        byte[] bc = new byte[size];
        bc[0] = (byte)0x2a; // aload_0
        int k=0;
        for (int j=1, n=1; j<params.length; ++j, ++n) {
            if (params[j].isReferenceType()) {
                bc[++k] = (byte)0x19; // aload
            } else if (params[j] == jq_Primitive.LONG) {
                bc[++k] = (byte)0x16; // lload
            } else if (params[j] == jq_Primitive.FLOAT) {
                bc[++k] = (byte)0x17; // fload
            } else if (params[j] == jq_Primitive.DOUBLE) {
                bc[++k] = (byte)0x18; // dload
            } else {
                bc[++k] = (byte)0x15; // iload
            }
            bc[++k] = (byte)n;
            if ((params[j] == jq_Primitive.LONG) || (params[j] == jq_Primitive.DOUBLE))
                ++n;
        }
        bc[++k] = (byte)0xb8; // invokestatic
        bc[++k] = (byte)(method_idx>>8);
        bc[++k] = (byte)method_idx;
        jq_Type t = m.getReturnType();
        if (t.isReferenceType()) {
            bc[++k] = (byte)0xb0; // areturn
        } else if (t == jq_Primitive.LONG) {
            bc[++k] = (byte)0xad; // lreturn
        } else if (t == jq_Primitive.FLOAT) {
            bc[++k] = (byte)0xae; // freturn
        } else if (t == jq_Primitive.DOUBLE) {
            bc[++k] = (byte)0xaf; // dreturn
        } else if (t == jq_Primitive.VOID) {
            bc[++k] = (byte)0xb1; // return
        } else {
            bc[++k] = (byte)0xac; // ireturn
        }
        jq_Method stubm = (jq_Method)getDeclaredMember(nd);
        jq_InstanceMethod stub;
        if (stubm == null) stub = jq_InstanceMethod.newInstanceMethod(this, nd);
        else {
            // method that we are overwriting must be instance.
            jq.Assert(!stubm.isStatic(), stubm.toString());
            stub = (jq_InstanceMethod)stubm;
        }
        //char access_flags = (char)(m.getAccessFlags() & ~ACC_NATIVE);
        char max_stack = (char)Math.max(m.getParamWords(), m.getReturnType().getReferenceSize()>>2);
        char max_locals = (char)m.getParamWords();
        stub.load(access_flags, max_stack, max_locals, bc, new jq_TryCatchBC[0], new jq_LineNumberBC[0], new HashMap());
        return stub;
    }
    
    public void merge(jq_Class that) {
        // initialize constant pool rebuilder
        final jq_ConstantPool.ConstantPoolRebuilder cpr = rebuildConstantPool(true);
        
        // add all instance fields.
        LinkedList newInstanceFields = new LinkedList();
        for (int i=0; i<that.declared_instance_fields.length; ++i) {
            jq_InstanceField that_f = that.declared_instance_fields[i];
            jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(that, that_f.getNameAndDesc());
            jq_InstanceField this_f = this.getDeclaredInstanceField(nd);
            if (this_f != null) {
                if (TRACE) SystemInterface.debugmsg("Instance field "+this_f+" already exists, skipping.");
                if (this_f.getAccessFlags() != that_f.getAccessFlags()) {
                    if (TRACE) 
                        SystemInterface.debugmsg("Access flags of instance field "+this_f+" from merged class do not match. ("+
                                                 (int)this_f.getAccessFlags()+"!="+(int)that_f.getAccessFlags()+")");
                }
                continue;
            }
            this_f = getOrCreateInstanceField(nd);
            jq.Assert(this_f.getState() == STATE_UNLOADED);
            this_f.load(that_f);
            that_f.unload(); Object b = that.members.remove(that_f.getNameAndDesc());
            if (TRACE) SystemInterface.debugmsg("Removed member "+that_f.getNameAndDesc()+" from member set of "+that+": "+b);
            if (TRACE) SystemInterface.debugmsg("Adding instance field: "+this_f);
            this.addDeclaredMember(nd, this_f);
            newInstanceFields.add(this_f);
            cpr.addOther(this_f.getName());
            cpr.addOther(this_f.getDesc());
            cpr.addAttributeNames(this_f);
        }
        if (newInstanceFields.size() > 0) {
            jq_InstanceField[] ifs = new jq_InstanceField[this.declared_instance_fields.length+newInstanceFields.size()];
            System.arraycopy(this.declared_instance_fields, 0, ifs, 0, this.declared_instance_fields.length);
            int j = this.declared_instance_fields.length-1;
            for (Iterator i=newInstanceFields.iterator(); i.hasNext(); )
                ifs[++j] = (jq_InstanceField)i.next();
            this.declared_instance_fields = ifs;
        }
        
        // add all static fields.
        LinkedList newStaticFields = new LinkedList();
        for (int i=0; i<that.static_fields.length; ++i) {
            jq_StaticField that_f = that.static_fields[i];
            jq_NameAndDesc nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(that, that_f.getNameAndDesc());
            jq_StaticField this_f = this.getDeclaredStaticField(nd);
            if (this_f != null) {
                if (TRACE) SystemInterface.debugmsg("Static field "+this_f+" already exists, skipping.");
                if (this_f.getAccessFlags() != that_f.getAccessFlags()) {
                    if (TRACE) 
                        SystemInterface.debugmsg("Access flags of static field "+this_f+" from merged class do not match. ("+
                                                 (int)this_f.getAccessFlags()+"!="+(int)that_f.getAccessFlags()+")");
                }
                continue;
            }
            this_f = getOrCreateStaticField(nd);
            jq.Assert(this_f.getState() == STATE_UNLOADED);
            this_f.load(that_f);
            that_f.unload(); Object b = that.members.remove(that_f.getNameAndDesc());
            if (TRACE) SystemInterface.debugmsg("Removed member "+that_f.getNameAndDesc()+" from member set of "+that+": "+b);
            if (TRACE) SystemInterface.debugmsg("Adding static field: "+this_f);
            this.addDeclaredMember(nd, this_f);
            newStaticFields.add(this_f);
            cpr.addOther(this_f.getName());
            cpr.addOther(this_f.getDesc());
            cpr.addAttributeNames(this_f);
        }
        if (newStaticFields.size() > 0) {
            jq_StaticField[] ifs = new jq_StaticField[this.static_fields.length+newStaticFields.size()];
            System.arraycopy(this.static_fields, 0, ifs, 0, this.static_fields.length);
            int j = this.static_fields.length-1;
            for (Iterator i=newStaticFields.iterator(); i.hasNext(); ) {
                ifs[++j] = (jq_StaticField)i.next();
                this.static_data_size += ifs[j].getWidth();
            }
            this.static_fields = ifs;
        }
        
        // visit all instance methods.
        LinkedHashMap newInstanceMethods = new LinkedHashMap();
        for (int i=0; i<that.declared_instance_methods.length; ++i) {
            jq_InstanceMethod that_m = that.declared_instance_methods[i];
            jq_NameAndDesc nd = that_m.getNameAndDesc();
            //jq_NameAndDesc nd = merge_convertNameAndDesc(that_m.getNameAndDesc());
            jq.Assert(ClassLib.ClassLibInterface.convertClassLibNameAndDesc(that, nd) == nd);
            jq_InstanceMethod this_m = this.getDeclaredInstanceMethod(nd);
            byte[] bc = that_m.getBytecode();
            if (bc == null) {
                if (this_m != null) {
                    if (TRACE) SystemInterface.debugmsg("Using existing body for instance method "+this_m+".");
                } else {
                    System.err.println("Body of method "+that_m+" doesn't already exist!");
                }
                continue;
            }
            if (bc.length == 5 && that_m instanceof jq_Initializer && that_m.getDesc() == Utf8.get("()V") &&
                this.getInitializer(Utf8.get("()V")) != null) {
                if (TRACE) SystemInterface.debugmsg("Skipping default initializer "+that_m+".");
                continue;
            }
            
            // extract instructions of method.
            Bytecodes.InstructionList il = new Bytecodes.InstructionList(that_m);
            
            // update constant pool references in the instructions, and add them to our constant pool.
            rewriteMethod(cpr, il);
            
            if (false) { //(this_m != null) {
                // method exists, use that one.
                if (TRACE) SystemInterface.debugmsg("Using existing instance method object "+this_m+".");
            } else {
                if (TRACE) SystemInterface.debugmsg("Creating new instance method object "+nd+".");
                this_m = this.getOrCreateInstanceMethod(nd);
                this.addDeclaredMember(nd, this_m);
                that_m.unload(); Object b = that.members.remove(that_m.getNameAndDesc());
                if (TRACE) SystemInterface.debugmsg("Removed member "+that_m.getNameAndDesc()+" from member set of "+that+": "+b);
            }
            this_m.load(that_m);
            
            // cache the instruction list for later.
            newInstanceMethods.put(this_m, il);
        }
        for (int i=0; i<this.declared_instance_methods.length; ++i) {
            jq_InstanceMethod this_m = this.declared_instance_methods[i];
            jq_Member this_m2 = this.getDeclaredMember(this_m.getNameAndDesc());
            if (newInstanceMethods.containsKey(this_m2)) {
                if (TRACE) SystemInterface.debugmsg("Skipping replaced instance method object "+this_m+".");
                continue;
            }
            jq.Assert(this_m == this_m2);
            byte[] bc = this_m.getBytecode();
            if (bc == null) {
                if (TRACE) SystemInterface.debugmsg("Skipping native/abstract instance method object "+this_m+".");
                newInstanceMethods.put(this_m, null);
                continue;
            }
            
            // extract instruction list.
            Bytecodes.InstructionList il = new Bytecodes.InstructionList(this_m);
            
            // add constant pool references from instruction list.
            cpr.addCode(il);
            
            // cache the instruction list for later.
            newInstanceMethods.put(this_m, il);
        }
        
        // visit all static methods.
        LinkedHashMap newStaticMethods = new LinkedHashMap();
        for (int i=0; i<that.static_methods.length; ++i) {
            jq_StaticMethod that_m = that.static_methods[i];
            Bytecodes.InstructionList il;
            jq_StaticMethod this_m;
            if (that_m instanceof jq_ClassInitializer) {
                if (TRACE) SystemInterface.debugmsg("Creating special static method for "+that_m+" class initializer.");
                jq.Assert(that_m.getBytecode() != null);
                Utf8 newname = Utf8.get("clinit_"+that.getJDKName());
                jq_NameAndDesc nd = new jq_NameAndDesc(newname, that_m.getDesc());
                this_m = getOrCreateStaticMethod(nd);
                this.addDeclaredMember(nd, this_m);
                this_m.load(that_m);
                
                // add a call to the special method in our class initializer.
                jq_ClassInitializer clinit = getClassInitializer();
                Bytecodes.InstructionList il2;
                if (clinit == null) {
                    jq_NameAndDesc nd2 = new jq_NameAndDesc(Utf8.get("<clinit>"), Utf8.get("()V"));
                    clinit = (jq_ClassInitializer)getOrCreateStaticMethod(nd2);
                    this.addDeclaredMember(nd2, clinit);
                    clinit.load((char)(ACC_PUBLIC | ACC_STATIC), (char)0, (char)0, new byte[0],
                           new jq_TryCatchBC[0], new jq_LineNumberBC[0], new HashMap());
                    if (TRACE) SystemInterface.debugmsg("Created class initializer "+clinit);
                    il2 = new Bytecodes.InstructionList();
                    Bytecodes.RETURN re = new Bytecodes.RETURN();
                    il2.append(re);
                    il2.setPositions();
                    newStaticMethods.put(clinit, il2);
                } else {
                    if (TRACE) SystemInterface.debugmsg("Using existing class initializer "+clinit);
                    il2 = new Bytecodes.InstructionList(clinit);
                }
                Bytecodes.INVOKESTATIC is = new Bytecodes.INVOKESTATIC(this_m);
                il2.insert(is);
                cpr.addMember(this_m);
                
                // extract instructions of method.
                il = new Bytecodes.InstructionList(that_m);
                
                that_m.unload(); Object b = that.members.remove(that_m.getNameAndDesc());
                if (TRACE) SystemInterface.debugmsg("Removed member "+that_m.getNameAndDesc()+" from member set of "+that+": "+b);
            } else {
                jq_NameAndDesc nd = that_m.getNameAndDesc();
                //jq_NameAndDesc nd = merge_convertNameAndDesc(that_m.getNameAndDesc());
                jq.Assert(ClassLib.ClassLibInterface.convertClassLibNameAndDesc(that, nd) == nd);
                this_m = this.getDeclaredStaticMethod(nd);
                byte[] bc = that_m.getBytecode();
                if (bc == null) {
                    if (this_m != null) {
                        if (TRACE) SystemInterface.debugmsg("Using existing body for static method "+this_m+".");
                    } else {
                        System.err.println("Body of method "+that_m+" doesn't already exist!");
                    }
                    continue;
                }
                // extract instructions of method.
                il = new Bytecodes.InstructionList(that_m);
                
                if (false) { //(this_m != null) {
                    // method exists, use that one.
                    if (TRACE) SystemInterface.debugmsg("Using existing static method object "+this_m+".");
                } else {
                    this_m = getOrCreateStaticMethod(nd);
                    this.addDeclaredMember(nd, this_m);
                    that_m.unload(); Object b = that.members.remove(that_m.getNameAndDesc());
                    if (TRACE) SystemInterface.debugmsg("Removed member "+that_m.getNameAndDesc()+" from member set of "+that+": "+b);
                    if (TRACE) SystemInterface.debugmsg("Created new static method object "+this_m+".");
                }
                this_m.load(that_m);
            }
            
            // update constant pool references in the instructions, and add them to our constant pool.
            rewriteMethod(cpr, il);
            
            // cache the instruction list for later.
            newStaticMethods.put(this_m, il);
        }
        for (int i=0; i<this.static_methods.length; ++i) {
            jq_StaticMethod this_m = this.static_methods[i];
            jq_Member this_m2 = this.getDeclaredMember(this_m.getNameAndDesc());
            if (newStaticMethods.containsKey(this_m2)) {
                //if (TRACE) SystemInterface.debugmsg("Skipping replaced static method object "+this_m+".");
                continue;
            }
            jq.Assert(this_m == this_m2);
            byte[] bc = this_m.getBytecode();
            if (bc == null) {
                //if (TRACE) SystemInterface.debugmsg("Skipping native/abstract static method object "+this_m+".");
                newStaticMethods.put(this_m, null);
                continue;
            }
            
            // extract instruction list.
            Bytecodes.InstructionList il = new Bytecodes.InstructionList(this_m);
            
            // add constant pool references from instruction list.
            cpr.addCode(il);
            
            // cache the instruction list for later.
            newStaticMethods.put(this_m, il);
        }
        
        // nothing more to add to constant pool, finish it.
        jq_ConstantPool new_cp = cpr.finish();
        
        // rebuild method arrays.
        this.declared_instance_methods = new jq_InstanceMethod[newInstanceMethods.size()];
        int j = -1;
        for (Iterator i=newInstanceMethods.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_InstanceMethod i_m = (jq_InstanceMethod)e.getKey();
            Bytecodes.InstructionList i_l = (Bytecodes.InstructionList)e.getValue();
            if (i_l != null) {
                if (TRACE) SystemInterface.debugmsg("Rebuilding bytecodes for instance method "+i_m+".");
                i_m.setCode(i_l, cpr);
            } else {
                if (TRACE) SystemInterface.debugmsg("No bytecodes for instance method "+i_m+".");
            }
            //if (TRACE) SystemInterface.debugmsg("Adding instance method "+i_m+" to array.");
            this.declared_instance_methods[++j] = i_m;
        }
        this.static_methods = new jq_StaticMethod[newStaticMethods.size()];
        j = -1;
        for (Iterator i=newStaticMethods.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            jq_StaticMethod i_m = (jq_StaticMethod)e.getKey();
            Bytecodes.InstructionList i_l = (Bytecodes.InstructionList)e.getValue();
            if (i_l != null) {
                if (TRACE) SystemInterface.debugmsg("Rebuilding bytecodes for static method "+i_m+".");
                i_m.setCode(i_l, cpr);
            } else {
                if (TRACE) SystemInterface.debugmsg("No bytecodes for static method "+i_m+".");
            }
            //if (TRACE) SystemInterface.debugmsg("Adding static method "+i_m+" to array.");
            this.static_methods[++j] = i_m;
        }
        this.remakeAttributes(cpr);
        this.const_pool = new_cp;
        getSourceFile(); // check for bug.
        if (TRACE) SystemInterface.debugmsg("Finished rebuilding constant pool.");
        that.super_class.removeSubclass(that);
        for (int i=0; i<that.declared_interfaces.length; ++i) {
            jq_Class di = that.declared_interfaces[i];
            di.removeSubinterface(that);
        }
        ClassLibInterface.i.unloadType(class_loader, that);
        if (TRACE) SystemInterface.debugmsg("Finished merging class "+this+".");
    }
    
    void remakeAttributes(jq_ConstantPool.ConstantPoolRebuilder cpr) {
        Utf8 sf = getSourceFile();
        if (sf != null) {
            byte[] b = new byte[2];
            jq.charToTwoBytes(cpr.get(sf), b, 0);
            attributes.put(Utf8.get("SourceFile"), b);
            if (TRACE) SystemInterface.debugmsg("Reset SourceFile attribute to cp idx "+(int)cpr.get(sf)+".");
        }
    }
    
    private void rewriteMethod(jq_ConstantPool.ConstantPoolRebuilder cp, Bytecodes.InstructionList il) {
        final jq_ConstantPool.ConstantPoolRebuilder cpr = cp;
        il.accept(new Bytecodes.EmptyVisitor() {
            public void visitCPInstruction(Bytecodes.CPInstruction x) {
                Object o = x.getObject();
                if (o instanceof String) {
                    cpr.addString((String)o);
                } else if (o instanceof jq_Type) {
                    if (o instanceof jq_Reference)
                        x.setObject(o = ClassLib.ClassLibInterface.convertClassLibCPEntry((jq_Reference)o));
                    cpr.addType((jq_Type)o);
                } else if (o instanceof jq_Member) {
                    x.setObject(o = ClassLib.ClassLibInterface.convertClassLibCPEntry((jq_Member)o));
                    cpr.addMember((jq_Member)o);
                } else {
                    cpr.addOther(o);
                }
            }
        });
    }
    
    public void merge_old(jq_Class that) {
        // initialize constant pool adder
        jq_ConstantPool.Adder cp_adder = const_pool.getAdder();
        
        // generate stubs for each of the methods in the other class.
        jq.Assert(that.declared_instance_methods.length <= 1, that.toString()); // the only instance method should be the fake <init> method.
        LinkedList toadd_instance = new LinkedList();
        LinkedList toadd_static = new LinkedList();
        char classfield_index = 0;
        for (int i=0; i<that.static_methods.length; ++i) {
            jq_StaticMethod sm = that.static_methods[i];
            if (sm.isClassInitializer()) continue;
            jq_Type[] that_param = sm.getParamTypes();
            jq.Assert(that_param.length >= 1, sm.toString());
            Utf8 name_utf = sm.getName();
            if (name_utf == Utf8.get("__init__")) name_utf = Utf8.get("<init>");
            char method_idx = cp_adder.add(sm, CONSTANT_ResolvedSMethodRef);
            if (that_param[0] == jq_Class._class) {
                // overridden static method
                char access_flags = sm.getAccessFlags();
                if (classfield_index == 0) {
                    jq_StaticField that_sf = that.getDeclaredStaticField(new jq_NameAndDesc(Utf8.get("_class"), Utf8.get("LClazz/jq_Class;")));
                    jq.Assert(that_sf != null);
                    classfield_index = cp_adder.add(that_sf, CONSTANT_ResolvedSFieldRef);
                }
uphere1:
                for (int j=0; ; ++j) {
                    if (j>=static_methods.length) {
                        StringBuffer desc = new StringBuffer("(");
                        for (int k=1; k<that_param.length; ++k) {
                            desc.append(that_param[k].getDesc().toString());
                        }
                        desc.append(")");
                        desc.append(sm.getReturnType().getDesc().toString());
                        Utf8 desc_utf = Utf8.get(desc.toString());
                        jq_NameAndDesc nd = new jq_NameAndDesc(name_utf, desc_utf);
                        jq_StaticMethod stub = generateStaticMethodStub(nd, sm, access_flags, (char)classfield_index, (char)method_idx);
                        toadd_static.add(stub);
                        break;
                    }
                    jq_StaticMethod f = this.static_methods[j];
                    if (f.getName() == name_utf) {
                        // non-public classes may have "Ljava/lang/Object;", so we need to check element-by-element.
                        jq_Type[] this_param = f.getParamTypes();
                        if (this_param.length+1 != that_param.length) continue;
                        for (int k=0; k<this_param.length; ++k) {
                            if ((this_param[k] != that_param[k+1]) &&
                                (that_param[k+1] != PrimordialClassLoader.loader.getJavaLangObject())) continue uphere1;
                        }
                        jq_NameAndDesc nd = f.getNameAndDesc();
                        access_flags = f.getAccessFlags();
                        jq_StaticMethod stub = generateStaticMethodStub(nd, sm, access_flags, (char)classfield_index, (char)method_idx);
                        if (TRACE) SystemInterface.debugmsg("Replacing static method: "+stub);
                        this.static_methods[j] = stub;
                        break;
                    }
                }
            } else {
                // overridden instance method
                char access_flags = (char)(sm.getAccessFlags() & ~ACC_STATIC);
                jq.Assert(that_param[0] == PrimordialClassLoader.loader.getJavaLangObject() || that_param[0] == this, sm.toString());
uphere2:
                for (int j=0; ; ++j) {
                    if (j>=declared_instance_methods.length) {
                        StringBuffer desc = new StringBuffer("(");
                        for (int k=1; k<that_param.length; ++k) {
                            desc.append(that_param[k].getDesc().toString());
                        }
                        desc.append(")");
                        desc.append(sm.getReturnType().getDesc().toString());
                        Utf8 desc_utf = Utf8.get(desc.toString());
                        jq_NameAndDesc nd = new jq_NameAndDesc(name_utf, desc_utf);
                        jq_InstanceMethod stub = generateInstanceMethodStub(nd, sm, access_flags, (char)method_idx);
                        toadd_instance.add(stub);
                        break;
                    }
                    jq_InstanceMethod f = this.declared_instance_methods[j];
                    if (f.getName() == name_utf) {
                        // non-public classes may have "Ljava/lang/Object;", so we need to check element-by-element.
                        jq_Type[] this_param = f.getParamTypes();
                        if (this_param.length != that_param.length) continue;
                        for (int k=0; k<this_param.length; ++k) {
                            if ((this_param[k] != that_param[k]) &&
                            (that_param[k] != PrimordialClassLoader.loader.getJavaLangObject())) continue uphere2;
                        }
                        jq_NameAndDesc nd = f.getNameAndDesc();
                        access_flags = f.getAccessFlags();
                        jq_InstanceMethod stub = generateInstanceMethodStub(nd, sm, access_flags, (char)method_idx);
                        if (TRACE) SystemInterface.debugmsg("Replacing instance method: "+stub);
                        this.declared_instance_methods[j] = stub;
                        break;
                    }
                }
            }
        }
        if (toadd_static.size() > 0) {
            jq_StaticMethod[] sms = new jq_StaticMethod[this.static_methods.length+toadd_static.size()];
            int i = this.static_methods.length-1;
            System.arraycopy(this.static_methods, 0, sms, 0, this.static_methods.length);
            Iterator it = toadd_static.iterator();
            while (it.hasNext()) {
                jq_StaticMethod stub = (jq_StaticMethod)it.next();
                if (TRACE) SystemInterface.debugmsg("Adding static method stub: "+stub);
                sms[++i] = stub;
            }
            this.static_methods = sms;
        }
        if (toadd_instance.size() > 0) {
            jq_InstanceMethod[] ims = new jq_InstanceMethod[this.declared_instance_methods.length+toadd_instance.size()];
            int i = this.declared_instance_methods.length-1;
            System.arraycopy(this.declared_instance_methods, 0, ims, 0, this.declared_instance_methods.length);
            Iterator it = toadd_instance.iterator();
            while (it.hasNext()) {
                jq_InstanceMethod stub = (jq_InstanceMethod)it.next();
                if (TRACE) SystemInterface.debugmsg("Adding instance method stub: "+stub);
                ims[++i] = stub;
            }
            this.declared_instance_methods = ims;
        }
        // add all instance fields.
        if (that.declared_instance_fields.length > 0) {
            jq_InstanceField[] ifs = new jq_InstanceField[this.declared_instance_fields.length+that.declared_instance_fields.length];
            System.arraycopy(this.declared_instance_fields, 0, ifs, 0, this.declared_instance_fields.length);
            int i = this.declared_instance_fields.length-1;
            for (int j=0; j<that.declared_instance_fields.length; ++j) {
                jq_InstanceField that_f = that.declared_instance_fields[j];
                jq_InstanceField this_f = getOrCreateInstanceField(that_f.getNameAndDesc());
                jq.Assert(this_f.getState() == STATE_UNLOADED, "conflict in field names in merged class: "+this_f);
                this_f.load(that_f.getAccessFlags(), that_f.getAttributes());
                if (TRACE) SystemInterface.debugmsg("Adding instance field: "+this_f);
                ifs[++i] = this_f;
            }
            this.declared_instance_fields = ifs;
        }
        cp_adder.finish();
    }

    public void verify() {
        if (isVerified()) return; // quick test.
        /*
        if (!Atomic.cas4(this, jq_Reference._state, STATE_LOADED, STATE_VERIFYING)) {
            // contention!  wait until verification completes.
            while (!isVerified()) Thread.yield();
            return;
        }
        */
        synchronized(this) {
            if (isVerified()) return; // other thread already loaded this type.
            if (state == STATE_VERIFYING)
                throw new ClassCircularityError(this.toString()); // recursively called verify
            state = STATE_VERIFYING;
            if (TRACE) SystemInterface.debugmsg("Beginning verifying "+this+"...");
            if (super_class != null) {
                super_class.load();
                super_class.verify();
            }
            // TODO: classfile verification
            if (TRACE) SystemInterface.debugmsg("Finished verifying "+this);
            state = STATE_VERIFIED;
        }
    }
    
    public void prepare() {
        if (isPrepared()) return; // quick test.
        /*
        if (!Atomic.cas4(this, jq_Reference._state, STATE_VERIFIED, STATE_PREPARING)) {
            // contention!  wait until verification completes.
            while (!isPrepared()) Thread.yield();
            return;
        }
         */
        synchronized(this) {
            if (isPrepared()) return; // other thread already loaded this type.
            if (state == STATE_PREPARING)
                throw new ClassCircularityError(this.toString()); // recursively called prepare (?)
            state = STATE_PREPARING;
            if (TRACE) SystemInterface.debugmsg("Beginning preparing "+this+"...");

            // TODO: check for inheritance cycles in interfaces

            // note: this method is a good candidate for specialization on super_class != null.
            if (super_class != null) {
                super_class.prepare();
            }

            int superfields;
            if (super_class != null) superfields = super_class.instance_fields.length;
            else superfields = 0;
            int numOfInstanceFields = superfields + this.declared_instance_fields.length;
            this.instance_fields = new jq_InstanceField[numOfInstanceFields];
            if (superfields > 0)
                System.arraycopy(super_class.instance_fields, 0, this.instance_fields, 0, superfields);

            // lay out instance fields
            int currentInstanceField = superfields-1;
            int size;
            if (super_class != null) size = super_class.instance_size;
            else size = OBJ_HEADER_SIZE;
            if (declared_instance_fields.length > 0) {
                if (!dont_align) {
                    // align on the largest data type
                    int largestDataType = declared_instance_fields[0].getSize();
                    int align = size & largestDataType-1;
                    if (align != 0) {
                        if (TRACE) SystemInterface.debugmsg("Gap of size "+align+" has been filled.");
                        // fill in the gap with smaller fields
                        for (int i=1; i<declared_instance_fields.length; ++i) {
                            jq_InstanceField f = declared_instance_fields[i];
                            int fsize = f.getSize();
                            if (fsize <= largestDataType-align) {
                                instance_fields[++currentInstanceField] = f;
                                if (TRACE) SystemInterface.debugmsg("Filling in field #"+currentInstanceField+" "+f+" at offset "+jq.hex(size - OBJ_HEADER_SIZE));
                                f.prepare(size - OBJ_HEADER_SIZE);
                                size += fsize;
                                align += fsize;
                            }
                            if (align == largestDataType) {
                                if (TRACE) SystemInterface.debugmsg("Gap of size "+align+" has been filled.");
                                break;
                            }
                        }
                    }
                } else {
                    if (TRACE) SystemInterface.debugmsg("Skipping field alignment for class "+this);
                }
                for (int i=0; i<declared_instance_fields.length; ++i) {
                    jq_InstanceField f = declared_instance_fields[i];
                    if (f.getState() == STATE_LOADED) {
                        instance_fields[++currentInstanceField] = f;
                        if (TRACE) SystemInterface.debugmsg("Laying out field #"+currentInstanceField+" "+f+" at offset "+jq.hex(size - OBJ_HEADER_SIZE));
                        f.prepare(size - OBJ_HEADER_SIZE);
                        size += f.getSize();
                    }
                }
            }
            this.instance_size = (size+3) & ~3;

            // lay out virtual method table
            int numOfNewVirtualMethods = 0;
            for (int i=0; i<declared_instance_methods.length; ++i) {
                jq_InstanceMethod m = declared_instance_methods[i];
                jq.Assert(m.getState() == STATE_LOADED);
                if (m.isInitializer()) {
                    // initializers cannot override or be overridden
                    continue;
                }
                if (super_class != null) {
                    jq_InstanceMethod m2 = super_class.getVirtualMethod(m.getNameAndDesc());
                    if (m2 != null) {
                        if (m.isPrivate() ||
                            m2.isPrivate() || m2.isFinal()) {// should not be overridden
                            System.out.println("error: method "+m+" overrides method "+m2);
                        }
                        m2.isOverriddenBy(m);
                        if (TRACE) SystemInterface.debugmsg("Virtual method "+m+" overrides method "+m2+" offset "+jq.hex(m2.getOffset()));
                        m.prepare(m2.getOffset());
                        continue;
                    }
                }
                if (m.isPrivate()) {
                    // private methods cannot override or be overridden
                    continue;
                }
                ++numOfNewVirtualMethods;
            }
            int super_virtual_methods;
            if (super_class != null)
                super_virtual_methods = super_class.virtual_methods.length;
            else
                super_virtual_methods = 0;
            int num_virtual_methods = super_virtual_methods + numOfNewVirtualMethods;
            virtual_methods = new jq_InstanceMethod[num_virtual_methods];
            if (super_virtual_methods > 0)
                System.arraycopy(super_class.virtual_methods, 0, this.virtual_methods, 0, super_virtual_methods);
            for (int i=0, j=super_virtual_methods-1; i<declared_instance_methods.length; ++i) {
                jq_InstanceMethod m = declared_instance_methods[i];
                if (m.isInitializer() || m.isPrivate()) {
                    // not in vtable
                    if (TRACE) SystemInterface.debugmsg("Skipping "+m+" in virtual method table.");
                    m.prepare();
                    continue;
                }
                if (m.isOverriding()) {
                    jq.Assert(m.getState() == STATE_PREPARED);
                    int entry = (m.getOffset() >> 2) - 1;
                    virtual_methods[entry] = m;
                    continue;
                }
                jq.Assert(m.getState() == STATE_LOADED);
                virtual_methods[++j] = m;
                if (TRACE) SystemInterface.debugmsg("Virtual method "+m+" is new, offset "+jq.hex((j+1)<<2));
                m.prepare((j+1)<<2);
            }
            // allocate space for vtable
            vtable = new int[num_virtual_methods+1];

            // calculate interfaces
            int n_super_interfaces;
            if (super_class != null) {
                n_super_interfaces = super_class.interfaces.length;
                if (super_class.isInterface())
                    ++n_super_interfaces; // add super_class to the list, too.
            } else
                n_super_interfaces = 0;

            interfaces = new jq_Class[n_super_interfaces + declared_interfaces.length];
            if (n_super_interfaces > 0) {
                System.arraycopy(super_class.interfaces, 0, this.interfaces, 0, super_class.interfaces.length);
                if (super_class.isInterface())
                    this.interfaces[n_super_interfaces-1] = super_class;
            }
            System.arraycopy(declared_interfaces, 0, this.interfaces, n_super_interfaces, declared_interfaces.length);

            // set prepared flags for static methods
            for (int i=0; i<static_methods.length; ++i) {
                jq_StaticMethod m = static_methods[i];
                m.prepare();
            }
            // set prepared flags for static fields
            for (int i=0; i<static_fields.length; ++i) {
                jq_StaticField m = static_fields[i];
                m.prepare();
            }
            
            if (TRACE) SystemInterface.debugmsg("Finished preparing "+this);
            state = STATE_PREPARED;
        }
    }
    
    public void sf_initialize() {
        if (isSFInitialized()) return; // quick test.
        /*
        if (!Atomic.cas4(this, jq_Reference._state, STATE_PREPARED, STATE_SFINITIALIZING)) {
            // contention!  wait until verification completes.
            while (!isSFInitialized()) Thread.yield();
            return;
        }
         */
        synchronized (this) {
            if (isSFInitialized()) return;
            if (state == STATE_SFINITIALIZING)
                throw new ClassCircularityError(this.toString()); // recursively called sf_initialize (?)
            state = STATE_SFINITIALIZING;
            if (TRACE) SystemInterface.debugmsg("Beginning SF init "+this+"...");
            if (super_class != null) {
                super_class.sf_initialize();
            }
            // lay out static fields and set their constant values
            if (static_data_size > 0) {
                static_data = new int[static_data_size>>2];
                for (int i=0, j=0; i<static_fields.length; ++i) {
                    jq_StaticField f = static_fields[i];
                    f.sf_initialize(static_data, j << 2);
                    if (f.isConstant()) {
                        Object cv = f.getConstantValue();
                        if (f.getType().isPrimitiveType()) {
                            if (f.getType() == jq_Primitive.LONG) {
                                long l = ((Long)cv).longValue();
                                static_data[j>>2  ] = (int)l;
                                static_data[j>>2+1] = (int)(l >> 32);
                            } else if (f.getType() == jq_Primitive.FLOAT) {
                                static_data[j>>2] = Float.floatToRawIntBits(((Float)cv).floatValue());
                            } else if (f.getType() == jq_Primitive.DOUBLE) {
                                long l = Double.doubleToRawLongBits(((Double)cv).doubleValue());
                                static_data[j>>2  ] = (int)l;
                                static_data[j>>2+1] = (int)(l >> 32);
                            } else {
                                static_data[j>>2] = ((Integer)cv).intValue();
                            }
                        } else {
                            // java/lang/String
                            static_data[j>>2] = Unsafe.addressOf(cv);
                        }
                    }
                    j += f.getWidth() >> 2;
                }
            }
            if (TRACE) SystemInterface.debugmsg("Finished SF init "+this);
            state = STATE_SFINITIALIZED;
        }
    }
    
    public void cls_initialize() throws ExceptionInInitializerError, NoClassDefFoundError {
        if (isClsInitialized()) return; // quick test.
        if (state == STATE_CLSINITERROR) throw new NoClassDefFoundError(this+": clinit failed");
        synchronized (this) {
            if (state >= STATE_CLSINITRUNNING) return;
            if (state == STATE_CLSINITIALIZING)
                throw new ClassCircularityError(this.toString()); // recursively called cls_initialize (?)
            state = STATE_CLSINITIALIZING;
            if (TRACE) SystemInterface.debugmsg("Beginning class init "+this+"...");
            if (super_class != null) {
                super_class.cls_initialize();
            }
            // generate compile stubs for each declared method
            for (int i=0; i<static_methods.length; ++i) {
                jq_StaticMethod m = static_methods[i];
                if (m.getState() == STATE_PREPARED) {
                    if (TRACE) SystemInterface.debugmsg("Compiling stub for: "+m);
                    jq_CompiledCode cc = m.compile_stub();
                    if (!jq.Bootstrapping) cc.patchDirectBindCalls();
                }
            }
            for (int i=0; i<declared_instance_methods.length; ++i) {
                jq_InstanceMethod m = declared_instance_methods[i];
                if (m.getState() == STATE_PREPARED) {
                    if (TRACE) SystemInterface.debugmsg("Compiling stub for: "+m);
                    jq_CompiledCode cc = m.compile_stub();
                    if (!jq.Bootstrapping) cc.patchDirectBindCalls();
                }
            }
            int[] vt = (int[])vtable;
            // 0th entry of vtable is class pointer
            vt[0] = Unsafe.addressOf(this);
            for (int i=0; i<virtual_methods.length; ++i) {
                vt[i+1] = virtual_methods[i].getDefaultCompiledVersion().getEntrypoint();
            }
            if (TRACE) SystemInterface.debugmsg(this+": "+jq.hex8(vt[0])+" vtable "+jq.hex8(Unsafe.addressOf(vt)));
            if (!jq.Bootstrapping)
                invokeclinit();
            if (TRACE) SystemInterface.debugmsg("Finished class init "+this);
            state = STATE_CLSINITIALIZED;
        }
    }

    private void invokeclinit() throws ExceptionInInitializerError {
        try {
            state = STATE_CLSINITRUNNING;
            jq_ClassInitializer clinit = this.getClassInitializer();
            if (clinit != null) 
                Reflection.invokestatic_V(clinit);
        } catch (Error x) {
            state = STATE_CLSINITERROR;
            throw x;
        } catch (Throwable x) {
            state = STATE_CLSINITERROR;
            throw new ExceptionInInitializerError(x);
        }
    }

    public static int NumOfIFieldsKept = 0;
    public static int NumOfSFieldsKept = 0;
    public static int NumOfIMethodsKept = 0;
    public static int NumOfSMethodsKept = 0;
    public static int NumOfIFieldsEliminated = 0;
    public static int NumOfSFieldsEliminated = 0;
    public static int NumOfIMethodsEliminated = 0;
    public static int NumOfSMethodsEliminated = 0;
    
    // not thread safe.
    public void trim(BootstrapRootSet trim) {
        jq.Assert(state == STATE_PREPARED);
        
        if (super_class != null)
            super_class.trim(trim);

        Set instantiatedTypes = trim.getInstantiatedTypes();
        Set necessaryFields = trim.getNecessaryFields();
        Set necessaryMethods = trim.getNecessaryMethods();
        
        Iterator it = members.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry e = (Map.Entry)it.next();
            jq_Member m = (jq_Member)e.getValue();
            if (m instanceof jq_Field) {
                if (!necessaryFields.contains(m)) {
                    if (trim.TRACE) trim.out.println("Eliminating field: "+m);
                    it.remove();
                }
            } else {
                jq.Assert(m instanceof jq_Method);
                if (!necessaryMethods.contains(m)) {
                    if (trim.TRACE) trim.out.println("Eliminating method: "+m);
                    it.remove();
                }
            }
        }

        int n;
        n=0;
        for (int i=0; i<declared_instance_fields.length; ++i) {
            jq_InstanceField f = declared_instance_fields[i];
            f.unprepare();
            if (necessaryMethods.contains(f)) ++n;
        }
        jq_InstanceField[] ifs = new jq_InstanceField[n];
        for (int i=0, j=-1; j<n-1; ++i) {
            jq_InstanceField f = declared_instance_fields[i];
            if (necessaryFields.contains(f)) {
                ifs[++j] = f;
                ++NumOfIFieldsKept;
            } else {
                if (trim.TRACE) trim.out.println("Eliminating instance field: "+f);
                ++NumOfIFieldsEliminated;
            }
        }
        declared_instance_fields = ifs;
        
        n=0; static_data_size=0;
        for (int i=0; i<static_fields.length; ++i) {
            jq_StaticField f = static_fields[i];
            f.unprepare();
            if (necessaryFields.contains(f)) ++n;
        }
        jq_StaticField[] sfs = new jq_StaticField[n];
        for (int i=0, j=-1; j<n-1; ++i) {
            jq_StaticField f = static_fields[i];
            if (necessaryFields.contains(f)) {
                sfs[++j] = f;
                static_data_size += f.getWidth();
                ++NumOfSFieldsKept;
            }
            else {
                if (trim.TRACE) trim.out.println("Eliminating static field: "+f);
                ++NumOfSFieldsEliminated;
            }
        }
        static_fields = sfs;

        n=0;
        for (int i=0; i<declared_instance_methods.length; ++i) {
            jq_InstanceMethod f = declared_instance_methods[i];
            f.unprepare();
            f.clearOverrideFlags();
            if (necessaryMethods.contains(f)) ++n;
        }
        jq_InstanceMethod[] ims = new jq_InstanceMethod[n];
        for (int i=0, j=-1; j<n-1; ++i) {
            jq_InstanceMethod f = declared_instance_methods[i];
            if (necessaryMethods.contains(f)) {
                ims[++j] = f;
                ++NumOfIMethodsKept;
            } else {
                if (trim.TRACE) trim.out.println("Eliminating instance method: "+f);
                ++NumOfIMethodsEliminated;
            }
        }
        declared_instance_methods = ims;
        
        n=0;
        for (int i=0; i<static_methods.length; ++i) {
            jq_StaticMethod f = static_methods[i];
            f.unprepare();
            if (necessaryMethods.contains(f)) ++n;
        }
        jq_StaticMethod[] sms = new jq_StaticMethod[n];
        for (int i=0, j=-1; j<n-1; ++i) {
            jq_StaticMethod f = static_methods[i];
            if (necessaryMethods.contains(f)) {
                sms[++j] = f;
                ++NumOfSMethodsKept;
            } else {
                if (trim.TRACE) trim.out.println("Eliminating static method: "+f);
                ++NumOfSMethodsEliminated;
            }
        }
        static_methods = sms;
        
        /*
        n=0;
        for (int i=0; i<declared_interfaces.length; ++i) {
            jq_Class f = declared_interfaces[i];
            if (instantiatedTypes.contains(f)) ++n;
        }
        jq_Class[] is = new jq_Class[n];
        for (int i=0, j=-1; j<n-1; ++i) {
            jq_Class f = declared_interfaces[i];
            if (instantiatedTypes.contains(f))
                is[++j] = f;
            else
                if (trim.TRACE) trim.out.println("Eliminating interface: "+f);
        }
        declared_interfaces = is;
        */
        
        const_pool.trim(necessaryFields, necessaryMethods);
        
        state = STATE_VERIFIED;
        this.prepare();
    }
    
    void readAttributes(DataInput in, Map attribMap) 
    throws IOException {
        char n_attributes = (char)in.readUnsignedShort();
        for (int i=0; i<n_attributes; ++i) {
            char attribute_name_index = (char)in.readUnsignedShort();
            if (getCPtag(attribute_name_index) != CONSTANT_Utf8)
                throw new ClassFormatError("constant pool entry "+attribute_name_index+", referred to by attribute "+i+
                                           ", is wrong type tag (expected="+CONSTANT_Utf8+", actual="+getCPtag(attribute_name_index));
            Utf8 attribute_desc = getCPasUtf8(attribute_name_index);
            int attribute_length = in.readInt();
            // todo: maybe we only want to read in attributes we care about...
            byte[] attribute_data = new byte[attribute_length];
            in.readFully(attribute_data);
            attribMap.put(attribute_desc, attribute_data);
        }
    }
    
    public static String className(Utf8 desc) {
        String temp = desc.toString();
        return temp.substring(1, temp.length()-1).replace('/','.');
    }

    private void addSubclass(jq_Class subclass) {
        jq_Class[] newsubclasses = new jq_Class[subclasses.length+1];
        System.arraycopy(subclasses, 0, newsubclasses, 0, subclasses.length);
        newsubclasses[subclasses.length] = subclass;
        subclasses = newsubclasses;
    }
    
    private void addSubinterface(jq_Class subinterface) {
        jq_Class[] newsubinterfaces = new jq_Class[subinterfaces.length+1];
        System.arraycopy(subinterfaces, 0, newsubinterfaces, 0, subinterfaces.length);
        newsubinterfaces[subinterfaces.length] = subinterface;
        subinterfaces = newsubinterfaces;
    }
    
    private void removeSubclass(jq_Class subclass) {
        jq_Class[] newsubclasses = new jq_Class[subclasses.length-1];
        for (int i=-1, j=0; i<newsubclasses.length-1; ++j) {
            if (subclass != subclasses[j]) {
                newsubclasses[++i] = subclasses[j];
            }
        }
        subclasses = newsubclasses;
    }
    
    private void removeSubinterface(jq_Class subinterface) {
        jq_Class[] newsubinterfaces = new jq_Class[subinterfaces.length-1];
        for (int i=-1, j=0; i<newsubinterfaces.length-1; ++j) {
            if (subinterface != subinterfaces[j]) {
                newsubinterfaces[++i] = subinterfaces[j];
            }
        }
        subinterfaces = newsubinterfaces;
    }
    
    public static jq_InstanceMethod getInvokespecialTarget(jq_Class clazz, jq_InstanceMethod method)
    throws AbstractMethodError {
        clazz.load();
        if (!clazz.isSpecial())
            return method;
        if (method.isInitializer())
            return method;
        if (!TypeCheck.isSuperclassOf(method.getDeclaringClass(), clazz))
            return method;
        jq_NameAndDesc nd = method.getNameAndDesc();
        for (;;) {
            clazz = clazz.getSuperclass();
            if (clazz == null)
                throw new AbstractMethodError();
            clazz.load();
            method = clazz.getDeclaredInstanceMethod(nd);
            if (method != null)
                return method;
        }
    }
    
    public jq_ConstantPool.ConstantPoolRebuilder rebuildConstantPool(boolean addCode) {
        jq_ConstantPool.ConstantPoolRebuilder cpr = new jq_ConstantPool.ConstantPoolRebuilder();
        cpr.addType(this);
        if (this.getSuperclass() != null)
            cpr.addType(this.getSuperclass());
        for (int i=0; i < declared_interfaces.length; ++i) {
            jq_Class f = declared_interfaces[i];
            cpr.addType(f);
        }
        for (int i=0; i < declared_instance_fields.length; ++i) {
            jq_InstanceField f = declared_instance_fields[i];
            cpr.addOther(f.getName());
            cpr.addOther(f.getDesc());
            cpr.addAttributeNames(f);
        }
        for (int i=0; i < static_fields.length; ++i) {
            jq_StaticField f = static_fields[i];
            cpr.addOther(f.getName());
            cpr.addOther(f.getDesc());
            cpr.addAttributeNames(f);
            if (f.isConstant())
                cpr.addOther(f.getConstantValue());
        }
        for (int i=0; i < declared_instance_methods.length; ++i) {
            jq_InstanceMethod f = declared_instance_methods[i];
            cpr.addOther(f.getName());
            cpr.addOther(f.getDesc());
            cpr.addAttributeNames(f);
            if (addCode) cpr.addCode(f);
            cpr.addExceptions(f);
        }
        for (int i=0; i < static_methods.length; ++i) {
            jq_StaticMethod f = static_methods[i];
            cpr.addOther(f.getName());
            cpr.addOther(f.getDesc());
            cpr.addAttributeNames(f);
            if (addCode) cpr.addCode(f);
            cpr.addExceptions(f);
        }
        Utf8 sourcefile = getSourceFile();
        if (sourcefile != null) {
            cpr.addOther(sourcefile);
        }
        // TODO: InnerClasses
        for (Iterator i = attributes.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            Utf8 name = (Utf8)e.getKey();
            cpr.addOther(name);
        }
        
        return cpr;
    }
    
    public void dump(DataOutput out) throws IOException {
        out.writeInt(0xcafebabe);
        out.writeChar(minor_version);
        out.writeChar(major_version);
        
        jq_ConstantPool.ConstantPoolRebuilder cpr = rebuildConstantPool(true);
        cpr.dump(out);
        
        out.writeChar(access_flags);
        out.writeChar(cpr.get(this));
        out.writeChar(cpr.get(super_class));
        
        out.writeChar(declared_interfaces.length);
        for(int i=0; i < declared_interfaces.length; i++)
            out.writeChar(cpr.get(declared_interfaces[i]));
        
        int nfields = static_fields.length + declared_instance_fields.length;
        jq.Assert(nfields <= Character.MAX_VALUE);
        out.writeChar(nfields);
        for(int i=0; i < static_fields.length; i++) {
            static_fields[i].dump(out, cpr);
        }
        for(int i=0; i < declared_instance_fields.length; i++) {
            declared_instance_fields[i].dump(out, cpr);
        }
        
        int nmethods = static_methods.length + declared_instance_methods.length;
        out.writeChar(nmethods);
        for(int i=0; i < static_methods.length; i++) {
            static_methods[i].dump(out, cpr);
        }
        for(int i=0; i < declared_instance_methods.length; i++) {
            declared_instance_methods[i].dump(out, cpr);
        }
        
        int nattributes = attributes.size();
        jq.Assert(nattributes <= Character.MAX_VALUE);
        out.writeChar(nattributes);
        for (Iterator i = attributes.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            Utf8 name = (Utf8)e.getKey();
            out.writeChar(cpr.get(name));
            byte[] value = (byte[])e.getValue();
            if (name == Utf8.get("SourceFile")) {
                char oldIndex = jq.twoBytesToChar(value, 0);
                Utf8 oldValue = (Utf8)const_pool.get(oldIndex);
                jq.charToTwoBytes(cpr.get(oldValue), value, 0);
            } else if (name == Utf8.get("InnerClasses")) {
                // TODO
            }
            out.writeInt(value.length);
            out.write(value);
        }
    }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Class;");
}

/*
 * Trimmer.java
 *
 * Created on January 1, 2001, 10:59 AM
 *
 * @author  jwhaley
 * @version 
 */

package Compil3r.Analysis;

import jq;
import Allocator.HeapAllocator;
import Allocator.DefaultHeapAllocator;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_Type;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Member;
import Clazz.jq_Field;
import Clazz.jq_StaticField;
import Clazz.jq_InstanceField;
import Clazz.jq_Method;
import Clazz.jq_StaticMethod;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Initializer;
import Clazz.jq_ClassInitializer;
import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;
import Run_Time.MathSupport;
import Run_Time.Monitor;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;
import Scheduler.jq_InterrupterThread;
import Compil3r.Reference.x86.x86ReferenceLinker;

import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.Array;

import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
import Util.ArrayIterator;
import Util.Relation;
import Util.LightRelation;
import Util.IdentityHashCodeWrapper;

public class Trimmer {

    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    final Set/*jq_Type*/ instantiatedTypes;
    final Set/*jq_Type*/ necessaryTypes;
    final Set/*jq_Member*/ necessaryMembers;
    final List/*jq_Method*/ worklist;
    final boolean AddAllClassMethods = false;
    final boolean AddAllClassFields;
    
    final Set/*IdentityHashCodeWrapper*/ visitedObjects;
    final ObjectTraverser obj_trav;
    
    public Trimmer(jq_Method method, ObjectTraverser obj_trav, boolean addAllClassFields) {
        this(obj_trav, addAllClassFields);
        addToWorklist(method);
    }
    public Trimmer(ObjectTraverser obj_trav, boolean addAllClassFields) {
        this.necessaryMembers = new HashSet();
        this.necessaryTypes = new HashSet();
        this.instantiatedTypes = new HashSet();
        this.worklist = new LinkedList();
        this.visitedObjects = new HashSet();
        this.obj_trav = obj_trav;
        this.AddAllClassFields = addAllClassFields;
        
        // some internal vm data structures are necessary for correct execution
        // under just about any circumstances.
        jq_Class c = jq_Class._class; c.load(); c.verify(); c.prepare();
        c = jq_Primitive._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_Array._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_InstanceField._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_StaticField._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_InstanceMethod._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_StaticMethod._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_Initializer._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        c = jq_ClassInitializer._class; c.load(); c.verify(); c.prepare();
        addToNecessaryTypes(c);
        addToNecessarySet(jq_Reference._vtable);
        
        // the bootstrap loader uses the static fields in the SystemInterface class.
        c = SystemInterface._class; c.load(); c.verify(); c.prepare();
        jq_StaticField[] sfs = SystemInterface._class.getDeclaredStaticFields();
        for (int i=0; i<sfs.length; ++i) {
            addToNecessarySet(sfs[i]);
        }
        // even if there are no calls to these Unsafe methods, we need their definitions
        // to stick around so that we can check against them.
        c = Unsafe._class; c.load(); c.verify(); c.prepare();
        jq_StaticMethod[] sms = Unsafe._class.getDeclaredStaticMethods();
        for (int i=0; i<sms.length; ++i) {
            if (sms[i] instanceof jq_ClassInitializer) continue;
            this.necessaryMembers.add(sms[i]);
        }
        
        addToNecessarySet(Unsafe._remapper_object);

        // setIn0, setOut0, and setErr0 use these fields, but the trimmer doesn't detect the uses.
        c = PrimordialClassLoader.loader.getJavaLangSystem();
        c.load(); c.verify(); c.prepare();
        jq_StaticField _sf = c.getOrCreateStaticField("in", "Ljava/io/InputStream;");
        addToNecessarySet(_sf);
        _sf = c.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
        addToNecessarySet(_sf);
        _sf = c.getOrCreateStaticField("err", "Ljava/io/PrintStream;");
        addToNecessarySet(_sf);
        
        // an instance of this class is created via reflection during VM initialization.
        c = (jq_Class)Reflection.getJQType(sun.io.CharToByteConverter.getDefault().getClass());
        c.load(); c.verify(); c.prepare();
        addToInstantiatedTypes(c);
        jq_InstanceMethod i = c.getOrCreateInstanceMethod("<init>", "()V");
        addToWorklist(i);
        
        // an instance of this class is created via reflection during VM initialization.
        c = (jq_Class)Reflection.getJQType(sun.io.ByteToCharConverter.getDefault().getClass());
        c.load(); c.verify(); c.prepare();
        addToInstantiatedTypes(c);
        i = c.getOrCreateInstanceMethod("<init>", "()V");
        addToWorklist(i);
        
        // created via reflection when loading from a zip file
        c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile$ZipFileInputStream;");
        c.load(); c.verify(); c.prepare();
        addToInstantiatedTypes(c);
        
        // the trap handler can be implicitly called from any bytecode than can trigger a hardware exception.
        jq_StaticMethod sm = ExceptionDeliverer._trap_handler;
        c = sm.getDeclaringClass(); c.load(); c.verify(); c.prepare();
        addToWorklist(sm);
        
        addToWorklist(jq_Method._compile);
        
        // entrypoint for new threads
        c = jq_NativeThread._class; c.load(); c.verify(); c.prepare();
        addToWorklist(jq_NativeThread._nativeThreadEntry);
        // thread switch interrupt
        addToWorklist(jq_NativeThread._threadSwitch);
        
        // entrypoint for interrupter thread
        c = jq_InterrupterThread._class; c.load(); c.verify(); c.prepare();
        addToWorklist(jq_InterrupterThread._run);
        
        // tracing in the compiler uses these
        //c = jq._class; c.load(); c.verify(); c.prepare();
        //addToWorklist(jq._hex8);
        //addToWorklist(jq._hex16);
    }

    public void addToNecessaryTypes(jq_Type t) {
        if (t.isClassType()) {
            if (!necessaryTypes.contains(t)) {
                necessaryTypes.add(t);
                jq_Class c2 = (jq_Class)t;
                // add all supertypes as necessary, as well
                for (jq_Class c3 = c2.getSuperclass(); c3 != null; c3 = c3.getSuperclass())
                    addToNecessaryTypes(c3);
                if (AddAllClassMethods) {
                    if (TRACE) out.println("Adding methods of new class "+t);
                    for(Iterator it = new ArrayIterator(c2.getDeclaredStaticMethods());
                        it.hasNext(); ) {
                        jq_StaticMethod m = (jq_StaticMethod)it.next();
                        addToWorklist(m);
                    }
                    for(Iterator it = new ArrayIterator(c2.getDeclaredInstanceMethods());
                        it.hasNext(); ) {
                        jq_InstanceMethod m = (jq_InstanceMethod)it.next();
                        addToWorklist(m);
                    }
                }
                if (AddAllClassFields) {
                    if (TRACE) out.println("Adding fields of new class "+t);
                    for(Iterator it = new ArrayIterator(c2.getDeclaredStaticFields());
                        it.hasNext(); ) {
                        jq_StaticField f = (jq_StaticField)it.next();
                        addToNecessarySet(f);
                        if (f.getType().isReferenceType()) addStaticFieldValue(f);
                    }
                    for(Iterator it = new ArrayIterator(c2.getDeclaredInstanceFields());
                        it.hasNext(); ) {
                        jq_InstanceField f = (jq_InstanceField)it.next();
                        addToNecessarySet(f);
                    }
                }
            }
            return;
        }
        necessaryTypes.add(t);
    }
    
    public void addToInstantiatedTypes(jq_Type t) {
        if (TRACE) out.println("Adding instantiated type "+t);
        jq.assert(t.isPrepared());
        instantiatedTypes.add(t);
        addToNecessaryTypes(t);
    }
    public void addToWorklist(jq_Method m) {
        if (TRACE) out.println("Adding method "+m+" to worklist");
        jq.assert(m.getDeclaringClass().isPrepared());
        necessaryMembers.add(m);
        worklist.add(m);
        addToNecessaryTypes(m.getDeclaringClass());
    }
    public void addToNecessarySet(jq_Field m) {
        if (TRACE) out.println("Adding field "+m+" to necessary set");
        necessaryMembers.add(m);
        addToNecessaryTypes(m.getDeclaringClass());
    }
    
    public Set getNecessaryMembers() {
        return necessaryMembers;
    }
    public Set getNecessaryTypes() {
        return necessaryTypes;
    }
    public Set getInstantiatedTypes() {
        return instantiatedTypes;
    }

    private void addClassInterfaceImplementations(jq_Class k) {
        jq_Class[] in = k.getInterfaces();
        for (int i=0; i<in.length; ++i) {
            jq_Class f = in[i];
            f.load(); f.verify(); f.prepare();
            jq_InstanceMethod[] ims = f.getVirtualMethods();
            for (int j=0; j<ims.length; ++j) {
                jq_InstanceMethod im = ims[j];
                if (!necessaryMembers.contains(im)) {
                    continue;
                }
                jq_InstanceMethod m2 = k.getVirtualMethod(im.getNameAndDesc());
                if (m2 == null) {
                    // error:
                    if (TRACE) out.println("Error! Class "+k+" doesn't implement interface method "+im);
                    continue;
                }
                if (!necessaryMembers.contains(m2)) {
                    addToWorklist(m2);
                } else {
                    if (TRACE) out.println(m2+" already added as necessary");
                }
            }
        }
    }
    
    public static final boolean ADD_STATIC_FIELD_VALUES = true;
    
    private static Field getField(Class c, String fieldName) {
        Class c2 = c;
        while (c != null) {
            Field[] fields = c.getDeclaredFields();
            for (int i=0; i<fields.length; ++i) {
                Field f = fields[i];
                if (f.getName().equals(fieldName)) {
                    f.setAccessible(true);
                    return f;
                }
            }
            c = c.getSuperclass();
        }
        jq.UNREACHABLE("host jdk does not contain field "+c2.getName()+"."+fieldName);
        return null;
    }
    private void addObject(Object o) {
        if (o == null) return;
        IdentityHashCodeWrapper a = IdentityHashCodeWrapper.create(o);
        if (visitedObjects.contains(a))
            return;
        visitedObjects.add(a);
        Class objType = o.getClass();
        jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
        if (TRACE) out.println("Visiting object of type "+jqType);
        if (!instantiatedTypes.contains(jqType)) {
            jqType.load(); jqType.verify(); jqType.prepare();
            addToInstantiatedTypes(jqType);
            if (jqType.isClassType()) {
                addClassInitializer((jq_Class)jqType);
                addSuperclassVirtualMethods((jq_Class)jqType);
                addClassInterfaceImplementations((jq_Class)jqType);
            }
        }
        if (jqType.isArrayType()) {
            jq_Type elemType = ((jq_Array)jqType).getElementType();
            if (elemType.isReferenceType()) {
                int length = Array.getLength(o);
                Object[] v = (Object[])o;
                if (TRACE) out.println("Visiting array of "+length+" elements");
                for (int k=0; k<length; ++k) {
                    Object o2 = v[k];
                    addObject(o2);
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
                    if (TRACE) out.println("Visiting field "+f);
                    Object o2 = obj_trav.getInstanceFieldValue(o, f);
                    addObject(o2);
                }
            }
        }
    }
    
    public void addStaticFieldValue(jq_StaticField f) {
        if (ADD_STATIC_FIELD_VALUES) {
            jq_Type ftype = f.getType();
            if (TRACE) out.println("Visiting static field "+f+" "+ftype);
            jq.assert(ftype.isReferenceType());
            Object o = obj_trav.getStaticFieldValue(f);
            addObject(o);
        }
    }
    
    private void addInterfaceImplementations(jq_InstanceMethod m) {
        jq_Class interf = m.getDeclaringClass();
        jq.assert(interf.isInterface());
        Iterator i = instantiatedTypes.iterator();
        while (i.hasNext()) {
            jq_Type t = (jq_Type)i.next();
            if (t.isReferenceType()) {
                jq_Reference r = (jq_Reference)t;
                if (r.implementsInterface(interf)) {
                    if (!r.isClassType()) {
                        // error:
                        if (TRACE) out.println("Error: interface call to "+m+" on array "+r);
                        continue;
                    }
                    jq_InstanceMethod m2 = ((jq_Class)r).getVirtualMethod(m.getNameAndDesc());
                    if (m2 == null) {
                        // error:
                        if (TRACE) out.println("Error: class "+r+" does not implement interface method "+m);
                        continue;
                    }
                    if (!necessaryMembers.contains(m2)) {
                        addToWorklist(m2);
                    }
                }
            }
        }
    }

    private boolean isSuperclassMethodNecessary(jq_Class c, jq_InstanceMethod m) {
        for ( ; c != null; c = c.getSuperclass()) {
            jq_InstanceMethod m2 = c.getVirtualMethod(m.getNameAndDesc());
            if (m2 != null) {
                if (necessaryMembers.contains(m2)) {
                    if (TRACE) out.println("Overridden method "+m2+" is necessary!");
                    return true;
                }
            }
        }
        return false;
    }
    
    private void addSuperclassVirtualMethods(jq_Class c) {
        jq_InstanceMethod[] ms = c.getVirtualMethods();
        for (int i=0; i<ms.length; ++i) {
            jq_InstanceMethod m = ms[i];
            if (m.isOverriding()) {
                if (TRACE) out.println("Checking virtual method "+m);
                if (isSuperclassMethodNecessary(c.getSuperclass(), m)) {
                    addToWorklist(m);
                }
            }
        }
    }
    
    private void addSubclassVirtualMethods(jq_Class c, jq_InstanceMethod m) {
        if (!m.isOverridden())
            return;
        jq_Class[] subclasses = c.getSubClasses();
        for (int i=0; i<subclasses.length; ++i) {
            jq_Class subclass = subclasses[i];
            jq_Method m2 = (jq_Method)subclass.getDeclaredMember(m.getNameAndDesc());
            if (m2 != null && !m2.isStatic()) {
                if (!necessaryMembers.contains(m2)) {
                    addToWorklist(m2);
                }
            }
            addSubclassVirtualMethods(subclass, m);
        }
    }

    static final boolean ADD_CLASS_INITIALIZERS = false;
    private void addClassInitializer(jq_Class c) {
        c.load(); c.verify(); c.prepare();
        if (ADD_CLASS_INITIALIZERS) {
            jq_Method m = c.getClassInitializer();
            if (m != null) {
                if (!necessaryMembers.contains(m)) {
                    addToWorklist(m);
                    jq_Class superclass = c.getSuperclass();
                    if (superclass != null) addClassInitializer(superclass);
                }
            }
        }
    }
    
    public void go() {
        while (!worklist.isEmpty()) {
            jq_Method m = (jq_Method)worklist.remove(0);
            if (TRACE) out.println("Pulling method "+m+" from worklist");
            jq_Class c = m.getDeclaringClass();
            addClassInitializer(c);
            if (!m.isStatic()) {
                if (c.isInterface()) {
                    jq.assert(m.isAbstract());
                    addInterfaceImplementations((jq_InstanceMethod)m);
                    continue;
                } else {
                    addSubclassVirtualMethods(c, (jq_InstanceMethod)m);
                }
            }
            if (m.isNative() || m.isAbstract()) {
                // native/abstract method
                continue;
            }
            TrimmerVisitor v = new TrimmerVisitor(m);
            v.forwardTraversal();
        }
    }
    
    class TrimmerVisitor extends BytecodeVisitor {
        
        TrimmerVisitor(jq_Method method) {
            super(method);
            //this.TRACE=true;
        }

        jq_Method getstatic_method = null;
        boolean was_getstatic_method;
        
        public String toString() {
            return "Trim/"+jq.left(method.getName().toString(), 10);
        }

        public void forwardTraversal() throws VerifyError {
            if (this.TRACE) this.out.println(this+": Starting traversal.");
            super.forwardTraversal();
            if (this.TRACE) this.out.println(this+": Finished traversal.");
        }

        public void visitBytecode() throws VerifyError {
            was_getstatic_method = false;
            super.visitBytecode();
            //if (!was_getstatic_method) getstatic_method = null;
        }
        
        public void visitAASTORE() {
            super.visitAASTORE();
            INVOKEhelper(INVOKE_STATIC, TypeCheck._arrayStoreCheck);
        }
        public void visitLBINOP(byte op) {
            super.visitLBINOP(op);
            switch(op) {
                case BINOP_DIV:
                    INVOKEhelper(INVOKE_STATIC, MathSupport._ldiv);
                    break;
                case BINOP_REM:
                    INVOKEhelper(INVOKE_STATIC, MathSupport._lrem);
                    break;
                default:
                    break;
            }
        }
        public void visitF2I() {
            super.visitF2I();
            jq_Class c = MathSupport._class;
            c.load(); c.verify(); c.prepare();
            addToNecessarySet(MathSupport._maxint);
            addToNecessarySet(MathSupport._minint);
        }
        public void visitD2I() {
            super.visitD2I();
            jq_Class c = MathSupport._class;
            addToNecessarySet(MathSupport._maxint);
            addToNecessarySet(MathSupport._minint);
        }
        public void visitF2L() {
            super.visitF2L();
            jq_Class c = MathSupport._class;
            addToNecessarySet(MathSupport._maxlong);
            addToNecessarySet(MathSupport._minlong);
        }
        public void visitD2L() {
            super.visitD2L();
            jq_Class c = MathSupport._class;
            addToNecessarySet(MathSupport._maxlong);
            addToNecessarySet(MathSupport._minlong);
        }
        private void GETSTATIChelper(jq_StaticField f) {
            addClassInitializer(f.getDeclaringClass());
            addToNecessarySet(f);
            if (false) {
                if (f.getWidth() == 8)
                    INVOKEhelper(INVOKE_STATIC, x86ReferenceLinker._getstatic8);
                else
                    INVOKEhelper(INVOKE_STATIC, x86ReferenceLinker._getstatic4);
            }
        }
        public void visitIGETSTATIC(jq_StaticField f) {
            super.visitIGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitLGETSTATIC(jq_StaticField f) {
            super.visitLGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitFGETSTATIC(jq_StaticField f) {
            super.visitFGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitDGETSTATIC(jq_StaticField f) {
            super.visitDGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitAGETSTATIC(jq_StaticField f) {
            super.visitAGETSTATIC(f);
            GETSTATIChelper(f);
            if (f.getType() == jq_InstanceMethod._class ||
                f.getType() == jq_StaticMethod._class ||
                f.getType() == jq_Method._class ||
                f.getType() == jq_Initializer._class ||
                f.getType() == jq_ClassInitializer._class) {
                // reading from a static field of type jq_Method
                getstatic_method = (jq_Method)obj_trav.getStaticFieldValue(f);
                was_getstatic_method = true;
                if (this.TRACE) this.out.println("getstatic field "+f+" value: "+getstatic_method);
            }
            addStaticFieldValue(f);
        }
        private void PUTSTATIChelper(jq_StaticField f) {
            addClassInitializer(f.getDeclaringClass());
            addToNecessarySet(f);
            if (false) {
                if (f.getWidth() == 8)
                    INVOKEhelper(INVOKE_STATIC, x86ReferenceLinker._putstatic8);
                else
                    INVOKEhelper(INVOKE_STATIC, x86ReferenceLinker._putstatic4);
            }
        }
        public void visitIPUTSTATIC(jq_StaticField f) {
            super.visitIPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitLPUTSTATIC(jq_StaticField f) {
            super.visitLPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitFPUTSTATIC(jq_StaticField f) {
            super.visitFPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitDPUTSTATIC(jq_StaticField f) {
            super.visitDPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitAPUTSTATIC(jq_StaticField f) {
            super.visitAPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        private void GETFIELDhelper(jq_InstanceField f) {
            jq_Class k = f.getDeclaringClass();
            k.load(); k.verify(); k.prepare();
            addToNecessarySet(f);
        }
        public void visitIGETFIELD(jq_InstanceField f) {
            super.visitIGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitLGETFIELD(jq_InstanceField f) {
            super.visitLGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitFGETFIELD(jq_InstanceField f) {
            super.visitFGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitDGETFIELD(jq_InstanceField f) {
            super.visitDGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitAGETFIELD(jq_InstanceField f) {
            super.visitAGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitBGETFIELD(jq_InstanceField f) {
            super.visitBGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitCGETFIELD(jq_InstanceField f) {
            super.visitCGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitSGETFIELD(jq_InstanceField f) {
            super.visitSGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitZGETFIELD(jq_InstanceField f) {
            super.visitZGETFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitIPUTFIELD(jq_InstanceField f) {
            super.visitIPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitLPUTFIELD(jq_InstanceField f) {
            super.visitLPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitFPUTFIELD(jq_InstanceField f) {
            super.visitFPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitDPUTFIELD(jq_InstanceField f) {
            super.visitDPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitAPUTFIELD(jq_InstanceField f) {
            super.visitAPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitBPUTFIELD(jq_InstanceField f) {
            super.visitBPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitCPUTFIELD(jq_InstanceField f) {
            super.visitCPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitSPUTFIELD(jq_InstanceField f) {
            super.visitSPUTFIELD(f);
            GETFIELDhelper(f);
        }
        public void visitZPUTFIELD(jq_InstanceField f) {
            super.visitZPUTFIELD(f);
            GETFIELDhelper(f);
        }
        private void INVOKEhelper(byte op, jq_Method f) {
            switch (op) {
                case INVOKE_STATIC:
                    if (f.getDeclaringClass() == Unsafe._class)
                        return;
                    if (!necessaryMembers.contains(x86ReferenceLinker._invokestatic)) {
                        x86ReferenceLinker._class.load(); x86ReferenceLinker._class.verify(); x86ReferenceLinker._class.prepare();
                        addToWorklist(x86ReferenceLinker._invokestatic);
                    }
                    break;
                case INVOKE_SPECIAL:
                    f = jq_Class.getInvokespecialTarget(method.getDeclaringClass(), (jq_InstanceMethod)f);
                    if (!necessaryMembers.contains(x86ReferenceLinker._invokespecial)) {
                        x86ReferenceLinker._class.load(); x86ReferenceLinker._class.verify(); x86ReferenceLinker._class.prepare();
                        addToWorklist(x86ReferenceLinker._invokespecial);
                    }
                    break;
                case INVOKE_INTERFACE:
                    if (!necessaryMembers.contains(x86ReferenceLinker._invokeinterface)) {
                        x86ReferenceLinker._class.load(); x86ReferenceLinker._class.verify(); x86ReferenceLinker._class.prepare();
                        addToWorklist(x86ReferenceLinker._invokeinterface);
                    }
                    break;
            }
            // class initializer added when method is visited.
            if (!necessaryMembers.contains(f)) {
                jq_Class c = f.getDeclaringClass();
                c.load(); c.verify(); c.prepare();
                addToWorklist(f);
            }
            jq.assert(f.getDeclaringClass() != Unsafe._class);
        }
        private void reflective_invoke(byte op, jq_Method f) {
            if (f.getDeclaringClass() == Reflection._class) {
                if (f.getName().toString().startsWith("invokestatic")) {
                    System.out.println(method+": Reflective static invocation: "+getstatic_method);
                    // reflective invocation.  where does it go?
                    if (getstatic_method != null)
                        INVOKEhelper(INVOKE_STATIC, getstatic_method);
                } else if (f.getName().toString().startsWith("invokeinstance")) {
                    System.out.println(method+": Reflective instance invocation: "+getstatic_method);
                    // reflective invocation.  where does it go?
                    if (getstatic_method != null)
                        INVOKEhelper(INVOKE_SPECIAL, getstatic_method);
                }
            }
        }
        public void visitIINVOKE(byte op, jq_Method f) {
            super.visitIINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitLINVOKE(byte op, jq_Method f) {
            super.visitLINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitFINVOKE(byte op, jq_Method f) {
            super.visitFINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitDINVOKE(byte op, jq_Method f) {
            super.visitDINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitAINVOKE(byte op, jq_Method f) {
            super.visitAINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitVINVOKE(byte op, jq_Method f) {
            super.visitVINVOKE(op, f);
            reflective_invoke(op, f);
            INVOKEhelper(op, f);
        }
        public void visitNEW(jq_Type f) {
            super.visitNEW(f);
            if (true) {
                INVOKEhelper(INVOKE_STATIC, DefaultHeapAllocator._allocateObject);
            } else {
                INVOKEhelper(INVOKE_STATIC, HeapAllocator._clsinitAndAllocateObject);
            }
            if (!instantiatedTypes.contains(f)) {
                f.load(); f.verify(); f.prepare();
                addToInstantiatedTypes(f);
                addClassInitializer((jq_Class)f);
                addSuperclassVirtualMethods((jq_Class)f);
                addClassInterfaceImplementations((jq_Class)f);
            }
        }
        public void visitNEWARRAY(jq_Array f) {
            super.visitNEWARRAY(f);
            INVOKEhelper(INVOKE_STATIC, DefaultHeapAllocator._allocateArray);
            f.load(); f.verify(); f.prepare();
            if (!instantiatedTypes.contains(f)) {
                f.load(); f.verify(); f.prepare();
                addToInstantiatedTypes(f);
            }
        }
        public void visitATHROW() {
            super.visitATHROW();
            INVOKEhelper(INVOKE_STATIC, ExceptionDeliverer._athrow);
        }
        public void visitCHECKCAST(jq_Type f) {
            super.visitCHECKCAST(f);
            INVOKEhelper(INVOKE_STATIC, TypeCheck._checkcast);
        }
        public void visitINSTANCEOF(jq_Type f) {
            super.visitINSTANCEOF(f);
            INVOKEhelper(INVOKE_STATIC, TypeCheck._instance_of);
        }
        public void visitMONITOR(byte op) {
            super.visitMONITOR(op);
            if (op == MONITOR_ENTER)
                INVOKEhelper(INVOKE_STATIC, Monitor._monitorenter);
            else
                INVOKEhelper(INVOKE_STATIC, Monitor._monitorexit);
        }
        public void visitMULTINEWARRAY(jq_Type f, char dim) {
            super.visitMULTINEWARRAY(f, dim);
            INVOKEhelper(INVOKE_STATIC, HeapAllocator._multinewarray);
            if (!instantiatedTypes.contains(f)) {
                f.load(); f.verify(); f.prepare();
                addToInstantiatedTypes(f);
                for (int i=0; i<dim; ++i) {
                    if (!f.isArrayType()) {
                        // TODO: throws VerifyError here!
                        break;
                    }
                    f = ((jq_Array)f).getElementType();
                    if (!instantiatedTypes.contains(f)) {
                        f.load(); f.verify(); f.prepare();
                        addToInstantiatedTypes(f);
                    }
                }
            }
        }
    }

}

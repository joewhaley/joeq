/*
 * BootstrapRootSet.java
 *
 * Created on June 10, 2002, 5:01 PM
 */

package Bootstrap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_Field;
import Clazz.jq_FieldVisitor;
import Clazz.jq_Initializer;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Clazz.jq_TypeVisitor;
import Main.jq;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.ExceptionDeliverer;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Util.IdentityHashCodeWrapper;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class BootstrapRootSet {

    public static /*final*/ boolean TRACE = false;
    public static final java.io.PrintStream out = System.out;
    
    protected final Set/*jq_Type*/ instantiatedTypes;
    protected final Set/*jq_Type*/ necessaryTypes;
    protected final Set/*jq_Field*/ necessaryFields;
    protected final Set/*jq_Method*/ necessaryMethods;
    
    protected final LinkedHashSet/*Object*/ visitedObjects;
    
    protected List/*jq_TypeVisitor*/ instantiatedTypesListeners;
    protected List/*jq_TypeVisitor*/ necessaryTypesListeners;
    protected List/*jq_FieldVisitor*/ necessaryFieldsListeners;
    protected List/*jq_MethodVisitor*/ necessaryMethodsListeners;
    
    public boolean AddAllFields;
    
    /** Creates new BootstrapRootSet */
    public BootstrapRootSet(boolean addall) {
        this.instantiatedTypes = new HashSet();
        this.necessaryTypes = new HashSet();
        this.necessaryFields = new HashSet();
        this.necessaryMethods = new HashSet();
        this.visitedObjects = new LinkedHashSet();
        this.AddAllFields = addall;
    }
    
    public Set/*jq_Type*/ getInstantiatedTypes() { return instantiatedTypes; }
    public Set/*jq_Type*/ getNecessaryTypes() { return necessaryTypes; }
    public Set/*jq_Field*/ getNecessaryFields() { return necessaryFields; }
    public Set/*jq_Method*/ getNecessaryMethods() { return necessaryMethods; }
    
    public void registerInstantiatedTypeListener(jq_TypeVisitor tv) {
        if (instantiatedTypesListeners == null)
            instantiatedTypesListeners = new LinkedList();
        instantiatedTypesListeners.add(tv);
    }
    public void unregisterInstantiatedTypeListener(jq_TypeVisitor tv) {
        instantiatedTypesListeners.remove(tv);
    }
    public void registerNecessaryTypeListener(jq_TypeVisitor tv) {
        if (necessaryTypesListeners == null)
            necessaryTypesListeners = new LinkedList();
        necessaryTypesListeners.add(tv);
    }
    public void unregisterNecessaryTypeListener(jq_TypeVisitor tv) {
        necessaryTypesListeners.remove(tv);
    }
    public void registerNecessaryFieldListener(jq_FieldVisitor tv) {
        if (necessaryFieldsListeners == null)
            necessaryFieldsListeners = new LinkedList();
        necessaryFieldsListeners.add(tv);
    }
    public void unregisterNecessaryFieldListener(jq_FieldVisitor tv) {
        necessaryFieldsListeners.remove(tv);
    }
    public void registerNecessaryMethodListener(jq_MethodVisitor tv) {
        if (necessaryMethodsListeners == null)
            necessaryMethodsListeners = new LinkedList();
        necessaryMethodsListeners.add(tv);
    }
    public void unregisterNecessaryMethodListener(jq_MethodVisitor tv) {
        necessaryMethodsListeners.remove(tv);
    }
    
    public boolean addInstantiatedType(jq_Type t) {
        jq.Assert(t != null);
        addNecessaryType(t);
        boolean b = instantiatedTypes.add(t);
        if (b) {
            if (TRACE) out.println("New instantiated type: "+t);
            if (instantiatedTypesListeners != null) {
                for (Iterator i=instantiatedTypesListeners.iterator(); i.hasNext(); ) {
                    jq_TypeVisitor tv = (jq_TypeVisitor)i.next();
                    t.accept(tv);
                }
            }
        }
        return b;
    }
    
    public boolean addNecessaryType(jq_Type t) {
        if (t == null) return false;
        t.load(); t.verify(); t.prepare();
        boolean b = necessaryTypes.add(t);
        if (b) {
            if (TRACE) out.println("New necessary type: "+t);
            if (necessaryTypesListeners != null) {
                for (Iterator i=necessaryTypesListeners.iterator(); i.hasNext(); ) {
                    jq_TypeVisitor tv = (jq_TypeVisitor)i.next();
                    t.accept(tv);
                }
            }
            if (t instanceof jq_Class) {
                jq_Class klass = (jq_Class)t;
                if (AddAllFields) {
                    jq_StaticField[] sfs = klass.getDeclaredStaticFields();
                    for (int i=0; i<sfs.length; ++i) {
                        addNecessaryField(sfs[i]);
                    }
                }
                // add superclass as necessary, as well.
                addNecessaryType(klass.getSuperclass());
            }
        }
        return b;
    }
    
    public boolean addNecessaryField(jq_Field t) {
        addNecessaryType(t.getDeclaringClass());
        boolean b = necessaryFields.add(t);
        if (b) {
            if (TRACE) out.println("New necessary field: "+t);
            if (necessaryFieldsListeners != null) {
                for (Iterator i=necessaryFieldsListeners.iterator(); i.hasNext(); ) {
                    jq_FieldVisitor tv = (jq_FieldVisitor)i.next();
                    t.accept(tv);
                }
            }
        }
        return b;
    }
    
    public boolean addNecessaryMethod(jq_Method t) {
        addNecessaryType(t.getDeclaringClass());
        boolean b = necessaryMethods.add(t);
        if (b) {
            if (TRACE) out.println("New necessary method: "+t);
            if (necessaryMethodsListeners != null) {
                for (Iterator i=necessaryMethodsListeners.iterator(); i.hasNext(); ) {
                    jq_MethodVisitor tv = (jq_MethodVisitor)i.next();
                    t.accept(tv);
                }
            }
        }
        return b;
    }
    
    public void addDefaultRoots() {
        jq_Class c;
        jq_StaticField s_f; jq_InstanceField i_f;
        jq_StaticMethod s_m; jq_InstanceMethod i_m;
        
        // some internal vm data structures are necessary for correct execution
        // under just about any circumstances.
        addNecessaryType(jq_Class._class);
        addNecessaryType(jq_Primitive._class);
        addNecessaryType(jq_Array._class);
        addNecessaryType(jq_InstanceField._class);
        addNecessaryType(jq_StaticField._class);
        addNecessaryType(jq_InstanceMethod._class);
        addNecessaryType(jq_StaticMethod._class);
        addNecessaryType(jq_Initializer._class);
        addNecessaryType(jq_ClassInitializer._class);
        addNecessaryType(CodeAddress._class);
        addNecessaryType(HeapAddress._class);
        addNecessaryType(StackAddress._class);
        addNecessaryField(jq_Reference._vtable);
        
        // the bootstrap loader uses the static fields in the SystemInterface class.
        SystemInterface._class.load();
        jq_StaticField[] sfs = SystemInterface._class.getDeclaredStaticFields();
        for (int i=0; i<sfs.length; ++i) {
            addNecessaryField(sfs[i]);
        }
        // even if there are no calls to these Unsafe methods, we need their definitions
        // to stick around so that we can check against them.
        Unsafe._class.load();
        jq_StaticMethod[] sms = Unsafe._class.getDeclaredStaticMethods();
        for (int i=0; i<sms.length; ++i) {
            if (sms[i] instanceof jq_ClassInitializer) continue;
            addNecessaryMethod(sms[i]);
        }
        //addNecessaryField(Unsafe._remapper_object);

        // We need to be able to allocate objects and code.
        addNecessaryType(Allocator.SimpleAllocator._class);
        addNecessaryType(PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/RuntimeCodeAllocator;"));
        
        // setIn0, setOut0, and setErr0 use these fields, but the trimmer doesn't detect the uses.
        c = PrimordialClassLoader.loader.getJavaLangSystem();
        s_f = c.getOrCreateStaticField("in", "Ljava/io/InputStream;");
        addNecessaryField(s_f);
        s_f = c.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
        addNecessaryField(s_f);
        s_f = c.getOrCreateStaticField("err", "Ljava/io/PrintStream;");
        addNecessaryField(s_f);
        
        // private method initializeSystemClass is called reflectively
        s_m = c.getOrCreateStaticMethod("initializeSystemClass", "()V");
        addNecessaryMethod(s_m);
        
        // an instance of this class is created via reflection during VM initialization.
        c = (jq_Class)Reflection.getJQType(sun.io.CharToByteConverter.getDefault().getClass());
        i_m = c.getOrCreateInstanceMethod("<init>", "()V");
        addNecessaryMethod(i_m);
        
        // an instance of this class is created via reflection during VM initialization.
        c = (jq_Class)Reflection.getJQType(sun.io.ByteToCharConverter.getDefault().getClass());
        i_m = c.getOrCreateInstanceMethod("<init>", "()V");
        addNecessaryMethod(i_m);
        
        // an instance of this class is created via reflection during VM initialization.
        try {
            c = (jq_Class)Reflection.getJQType(sun.io.ByteToCharConverter.getConverter("ISO-8859-1").getClass());
            i_m = c.getOrCreateInstanceMethod("<init>", "()V");
            addNecessaryMethod(i_m);
        } catch (java.io.UnsupportedEncodingException x) { }

        // created via reflection when loading from a zip file
        //c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile$ZipFileInputStream;");
        //i_m = c.getOrCreateInstanceMethod("<init>", "(JJ)V");
        //addNecessaryMethod(i_m);
        
        // the trap handler can be implicitly called from any bytecode than can trigger a hardware exception.
        s_m = ExceptionDeliverer._trap_handler;
        addNecessaryMethod(s_m);
        
        // debugger is large, so compile it on demand.
        if (false) {
            s_m = ExceptionDeliverer._debug_trap_handler;
            addNecessaryMethod(s_m);
        }
        
        // we want the compiler to be able to run at run time, too.
        i_m = jq_Method._compile;
        addNecessaryMethod(i_m);
        
        // entrypoint for new threads
        addNecessaryMethod(Scheduler.jq_NativeThread._nativeThreadEntry);
        // thread switch interrupt
        addNecessaryMethod(Scheduler.jq_NativeThread._threadSwitch);
        // ctrl-break handler
        addNecessaryMethod(Scheduler.jq_NativeThread._ctrl_break_handler);
        // entrypoint for interrupter thread
        addNecessaryMethod(Scheduler.jq_InterrupterThread._run);
        
        // dunno why this doesn't show up
        addNecessaryType(Assembler.x86.Heap2HeapReference._class);
        
        // JDK1.4: an instance of this class is created via reflection during VM initialization.
        try {
            Class.forName("sun.nio.cs.ISO8859_1");
            c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/nio/cs/ISO_8859_1;");
            c.load();
            i_m = c.getOrCreateInstanceMethod("<init>", "()V");
            addNecessaryMethod(i_m);
        }
        catch (java.lang.NoClassDefFoundError x) { }
        catch (java.lang.ClassNotFoundException x) { }
        
        // tracing in the compiler uses these
        //c = jq._class; c.load(); c.verify(); c.prepare();
        //addToWorklist(jq._hex8);
        //addToWorklist(jq._hex16);
    }
    
    public boolean addObjectAndSubfields(Object o) {
        return addObjectAndSubfields(o, visitedObjects);
    }
    private boolean addObjectAndSubfields(Object o, LinkedHashSet objs) {
        if (o == null) return false;
        IdentityHashCodeWrapper a = IdentityHashCodeWrapper.create(o);
        if (visitedObjects.contains(a) || objs.contains(a))
            return false;
        objs.add(a);
        Class objType = o.getClass();
        jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
        if (TRACE) out.println("Adding object of type "+jqType+": "+o);
        addInstantiatedType(jqType);
        /*
                addClassInitializer((jq_Class)jqType);
                addSuperclassVirtualMethods((jq_Class)jqType);
                addClassInterfaceImplementations((jq_Class)jqType);
         */
        if (jqType.isArrayType()) {
            jq_Type elemType = ((jq_Array)jqType).getElementType();
            if (elemType.isAddressType()) {
                // no need to visit.
            } else if (elemType.isReferenceType()) {
                int length = java.lang.reflect.Array.getLength(o);
                Object[] v = (Object[])o;
                if (TRACE) out.println("Visiting "+jqType+" of "+length+" elements");
                for (int k=0; k<length; ++k) {
                    Object o2 = Reflection.arrayload_A(v, k);
                    addObjectAndSubfields(o2, objs);
                }
            }
        } else {
            jq.Assert(jqType.isClassType());
            jq_Class clazz = (jq_Class)jqType;
            jq_InstanceField[] fields = clazz.getInstanceFields();
            for (int k=0; k<fields.length; ++k) {
                jq_InstanceField f = fields[k];
                if (!AddAllFields && !necessaryFields.contains(f))
                    continue;
                jq_Type ftype = f.getType();
                if (ftype.isAddressType()) {
                    // no need to visit.
                } else if (ftype.isReferenceType()) {
                    if (TRACE) out.println("Visiting field "+f);
                    Object o2 = Reflection.getfield_A(o, f);
                    addObjectAndSubfields(o2, objs);
                }
            }
        }
        return true;
    }
    
    public void addNecessarySubfieldsOfVisitedObjects() {
        if (AddAllFields) return;
        LinkedHashSet objs = visitedObjects;
        for (;;) {
            LinkedHashSet objs2 = new LinkedHashSet();
            boolean change = false;
            for (Iterator i = objs.iterator(); i.hasNext(); ) {
                Object o = ((IdentityHashCodeWrapper)i.next()).getObject();
                Class objType = o.getClass();
                jq_Reference jqType = (jq_Reference)Reflection.getJQType(objType);
                if (jqType.isArrayType()) continue;
                jq.Assert(jqType.isClassType());
                jq_Class clazz = (jq_Class)jqType;
                jq_InstanceField[] fields = clazz.getInstanceFields();
                for (int k=0; k<fields.length; ++k) {
                    jq_InstanceField f = fields[k];
                    if (!necessaryFields.contains(f))
                        continue;
                    jq_Type ftype = f.getType();
                    if (ftype.isAddressType()) {
                        // no need to visit.
                    } else if (ftype.isReferenceType()) {
                        if (TRACE) out.println("Visiting field "+f+" of object of type "+clazz);
                        Object o2 = Reflection.getfield_A(o, f);
                        if (addObjectAndSubfields(o2, objs2))
                            change = true;
                    }
                }
            }
            if (!change) break;
            if (TRACE) out.println("Objects added: "+objs2.size()+", iterating over those objects.");
            visitedObjects.addAll(objs2);
            objs = objs2;
        }
    }
    
    public void addAllInterfaceMethodImplementations(jq_InstanceMethod i_m) {
        addNecessaryMethod(i_m);
        jq_Class interf = i_m.getDeclaringClass();
        jq.Assert(interf.isInterface());
        Iterator i = necessaryTypes.iterator();
        while (i.hasNext()) {
            jq_Type t = (jq_Type)i.next();
            if (!t.isReferenceType()) continue;
            if (t.isAddressType()) continue;
            jq_Reference r = (jq_Reference)t;
            if (!r.implementsInterface(interf)) continue;
            jq_InstanceMethod m2 = r.getVirtualMethod(i_m.getNameAndDesc());
            if (m2 == null) {
                // error:
                if (TRACE) out.println("Error: class "+r+" does not implement interface method "+i_m);
                continue;
            }
            addNecessaryMethod(m2);
        }
    }
    
    public void addAllVirtualMethodImplementations(jq_InstanceMethod i_m) {
        addNecessaryMethod(i_m);
        addAllVirtualMethodImplementations(i_m.getDeclaringClass(), i_m);
    }
    
    public void addAllVirtualMethodImplementations(jq_Class c, jq_InstanceMethod i_m) {
        if (!i_m.isOverridden())
            return;
        jq_Class[] subclasses = c.getSubClasses();
        for (int i=0; i<subclasses.length; ++i) {
            jq_Class subclass = subclasses[i];
            subclass.verify(); subclass.prepare();
            jq_Method m2 = (jq_Method)subclass.getDeclaredMember(i_m.getNameAndDesc());
            if (m2 != null && !m2.isStatic()) {
                addNecessaryMethod(m2);
            }
            addAllVirtualMethodImplementations(subclass, i_m);
        }
    }
    
}

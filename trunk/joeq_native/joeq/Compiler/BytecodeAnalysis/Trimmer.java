/*
 * Trimmer.java
 *
 * Created on January 1, 2001, 10:59 AM
 *
 * @author  jwhaley
 * @version 
 */

package Compil3r.BytecodeAnalysis;

import Main.jq;
import Allocator.HeapAllocator;
import Allocator.DefaultHeapAllocator;
import Clazz.*;
import Bootstrap.*;
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
    
    private final BootstrapRootSet rs;
    private final List/*jq_Method*/ methodWorklist;
    
    private final Set/*jq_Method*/ invokedVirtualMethods;
    private final Set/*jq_Method*/ invokedInterfaceMethods;
    
    public Trimmer(jq_Method method, Set initialClassSet, boolean addall) {
        this(initialClassSet, addall);
        rs.addNecessaryMethod(method);
    }
    public Trimmer(Set initialClassSet, boolean addall) {
        methodWorklist = new LinkedList();
        invokedVirtualMethods = new HashSet();
        invokedInterfaceMethods = new HashSet();
        rs = new BootstrapRootSet(addall);
        rs.registerNecessaryMethodListener(new AddMethodToWorklist());
        rs.registerNecessaryFieldListener(new AddStaticFieldContents());
        rs.registerNecessaryTypeListener(new UpkeepForNewlyDiscoveredClasses());
        
        rs.addDefaultRoots();
        for (Iterator i = initialClassSet.iterator(); i.hasNext(); ) {
            rs.addNecessaryType((jq_Type)i.next());
        }
    }

    public BootstrapRootSet getRootSet() { return rs; }
    
    public void addToWorklist(jq_Method m) {
        methodWorklist.add(m);
    }
    
    public void addInvokedInterfaceMethod(jq_InstanceMethod m) {
        invokedInterfaceMethods.add(m);
    }
    
    public void addInvokedVirtualMethod(jq_InstanceMethod m) {
        invokedVirtualMethods.add(m);
    }
    
    /*
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
     */
    
    public class AddMethodToWorklist extends jq_MethodVisitor.EmptyVisitor {
        public void visitMethod(jq_Method m) {
            addToWorklist(m);
        }
    }
    
    public class AddStaticFieldContents extends jq_FieldVisitor.EmptyVisitor {
        public void visitStaticField(jq_StaticField f) {
            if (f.getType().isPrimitiveType()) return;
            Object o2 = Reflection.getstatic_A(f);
            rs.addObjectAndSubfields(o2);
        }
    }
    /*
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
     */
    
    /*
    private void addClassInitializer(jq_Class c) {
        c.load(); c.verify(); c.prepare();
        if (ADD_CLASS_INITIALIZERS) {
            jq_Method m = c.getClassInitializer();
            if (m != null) {
                if (TRACE) out.println("Adding class initializer "+m);
                if (!necessaryMembers.contains(m)) {
                    addToWorklist(m);
                    jq_Class superclass = c.getSuperclass();
                    if (superclass != null) addClassInitializer(superclass);
                }
            }
        }
    }
     */
    
    public void go() {
        while (!methodWorklist.isEmpty()) {
            while (!methodWorklist.isEmpty()) {
                jq_Method m = (jq_Method)methodWorklist.remove(0);
                if (TRACE) out.println("Pulling method "+m+" from worklist");

                //jq_Class c = m.getDeclaringClass();
                //addClassInitializer(c);
                //if (!m.isStatic()) {
                //    if (c.isInterface()) {
                //        jq.Assert(m.isAbstract());
                //        addAllInterfaceMethodImplementations((jq_InstanceMethod)m);
                //        continue;
                //    } else {
                //        addSubclassVirtualMethods(c, (jq_InstanceMethod)m);
                //    }
                //}
                if (m.getBytecode() == null) {
                    // native/abstract method
                    continue;
                }
                TrimmerVisitor v = new TrimmerVisitor(m);
                v.forwardTraversal();
            }
            rs.addNecessarySubfieldsOfVisitedObjects();
        }
    }
    
    class TrimmerVisitor extends BytecodeVisitor {
        
        jq_Method getstatic_method;
        
        TrimmerVisitor(jq_Method method) {
            super(method);
            //this.TRACE = true;
        }

        public String toString() {
            return "Trim/"+jq.left(method.getName().toString(), 10);
        }

        public void forwardTraversal() throws VerifyError {
            if (this.TRACE) this.out.println(this+": Starting traversal.");
            super.forwardTraversal();
            if (this.TRACE) this.out.println(this+": Finished traversal.");
        }

        public void visitBytecode() throws VerifyError {
            super.visitBytecode();
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
            rs.addNecessaryField(MathSupport._maxint);
            rs.addNecessaryField(MathSupport._minint);
        }
        public void visitD2I() {
            super.visitD2I();
            rs.addNecessaryField(MathSupport._maxint);
            rs.addNecessaryField(MathSupport._minint);
        }
        public void visitF2L() {
            super.visitF2L();
            rs.addNecessaryField(MathSupport._maxlong);
            rs.addNecessaryField(MathSupport._minlong);
        }
        public void visitD2L() {
            super.visitD2L();
            rs.addNecessaryField(MathSupport._maxlong);
            rs.addNecessaryField(MathSupport._minlong);
        }
        private void GETSTATIChelper(jq_StaticField f) {
            f = resolve(f);
            //addClassInitializer(f.getDeclaringClass());
            rs.addNecessaryField(f);
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
            if (true) {
                if (f.getType() == jq_InstanceMethod._class ||
                    f.getType() == jq_StaticMethod._class ||
                    f.getType() == jq_Method._class ||
                    f.getType() == jq_Initializer._class ||
                    f.getType() == jq_ClassInitializer._class) {
                    // reading from a static field of type jq_Method
                    getstatic_method = (jq_Method)Reflection.getstatic_A(f);
                    if (this.TRACE) this.out.println("getstatic field "+f+" value: "+getstatic_method);
                }
            }
        }
        public void visitZGETSTATIC(jq_StaticField f) {
            super.visitZGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitBGETSTATIC(jq_StaticField f) {
            super.visitBGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitCGETSTATIC(jq_StaticField f) {
            super.visitCGETSTATIC(f);
            GETSTATIChelper(f);
        }
        public void visitSGETSTATIC(jq_StaticField f) {
            super.visitSGETSTATIC(f);
            GETSTATIChelper(f);
        }
        private void PUTSTATIChelper(jq_StaticField f) {
            f = resolve(f);
            //addClassInitializer(f.getDeclaringClass());
            rs.addNecessaryField(f);
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
        public void visitZPUTSTATIC(jq_StaticField f) {
            super.visitZPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitBPUTSTATIC(jq_StaticField f) {
            super.visitBPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitCPUTSTATIC(jq_StaticField f) {
            super.visitCPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        public void visitSPUTSTATIC(jq_StaticField f) {
            super.visitSPUTSTATIC(f);
            PUTSTATIChelper(f);
        }
        private void GETFIELDhelper(jq_InstanceField f) {
            f = resolve(f);
            rs.addNecessaryField(f);
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
            f = (jq_Method)resolve(f);
            switch (op) {
            case INVOKE_STATIC:
                if (f.getDeclaringClass() == Unsafe._class)
                    return;
                rs.addNecessaryMethod(x86ReferenceLinker._invokestatic);
                rs.addNecessaryMethod(f);
                break;
            case INVOKE_SPECIAL:
                rs.addNecessaryMethod(x86ReferenceLinker._invokespecial);
                f = jq_Class.getInvokespecialTarget(method.getDeclaringClass(), (jq_InstanceMethod)f);
                rs.addNecessaryMethod(f);
                break;
            case INVOKE_INTERFACE:
                rs.addNecessaryMethod(x86ReferenceLinker._invokeinterface);
                rs.addAllInterfaceMethodImplementations((jq_InstanceMethod)f);
                addInvokedInterfaceMethod((jq_InstanceMethod)f);
                break;
            case INVOKE_VIRTUAL:
                rs.addAllVirtualMethodImplementations((jq_InstanceMethod)f);
                addInvokedVirtualMethod((jq_InstanceMethod)f);
                break;
            }
            // class initializer added when method is visited.
            //jq.Assert(f.getDeclaringClass() != Unsafe._class);
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
            //INVOKEhelper(INVOKE_STATIC, DefaultHeapAllocator._allocateObject);
            INVOKEhelper(INVOKE_STATIC, HeapAllocator._clsinitAndAllocateObject);
            rs.addNecessaryType(f);
        }
        public void visitNEWARRAY(jq_Array f) {
            super.visitNEWARRAY(f);
            INVOKEhelper(INVOKE_STATIC, DefaultHeapAllocator._allocateArray);
            rs.addNecessaryType(f);
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
            rs.addNecessaryType(f);
            for (int i=0; i<dim; ++i) {
                if (!f.isArrayType()) {
                    // TODO: throws VerifyError here!
                    break;
                }
                f = ((jq_Array)f).getElementType();
                rs.addNecessaryType(f);
            }
        }
    }

    public void addNecessaryInterfaceMethodImplementations(jq_Class c, jq_Class inter) {
        inter.load(); inter.verify(); inter.prepare();
        jq.Assert(inter.isInterface());
        jq_InstanceMethod[] ms = inter.getVirtualMethods();
        for (int i=0; i<ms.length; ++i) {
            jq_InstanceMethod m = ms[i];
            if (!invokedInterfaceMethods.contains(m)) continue;
            jq_InstanceMethod m2 = c.getVirtualMethod(m.getNameAndDesc());
            if (m2 == null) continue;
            rs.addNecessaryMethod(m2);
        }
        jq_Class[] interfaces = inter.getInterfaces();
        for (int i=0; i<interfaces.length; ++i) {
            jq_Class k2 = interfaces[i];
            addNecessaryInterfaceMethodImplementations(c, k2);
        }
    }
    
    public class UpkeepForNewlyDiscoveredClasses extends jq_TypeVisitor.EmptyVisitor {
        public void visitClass(jq_Class c) {
            jq_InstanceMethod[] ms = c.getDeclaredInstanceMethods();
            for (int i=0; i<ms.length; ++i) {
                jq_InstanceMethod m = ms[i];
                if (m.isOverriding()) {
                    //if (TRACE) out.println("Checking virtual method "+m);
                    jq_InstanceMethod m2 = c.getSuperclass().getVirtualMethod(m.getNameAndDesc());
                    if (m2 != null) {
                        if (invokedVirtualMethods.contains(m2)) {
                            if (TRACE) out.println("Method "+m+" is necessary because it overrides "+m2);
                            rs.addNecessaryMethod(m);
                            continue;
                        } else {
                            //if (TRACE) out.println("Overridden method "+m2+" is not necessary!");
                        }
                    }
                }
            }
            jq_Class[] interfaces = c.getInterfaces();
            for (int i=0; i<interfaces.length; ++i) {
                jq_Class k2 = interfaces[i];
                addNecessaryInterfaceMethodImplementations(c, k2);
            }
        }
    }

}

/*
 * ClassDump.java
 *
 * Created on December 20, 2000, 1:18 AM
 *
 * @author  jwhaley
 * @version 
 */

package Main;

import java.io.PrintStream;
import java.util.Iterator;

import Clazz.*;
import Bootstrap.*;
import Run_Time.*;
import UTF.Utf8;
import jq;

import Util.ArrayIterator;

public abstract class ClassDump {
    
    public static void main(String[] args) {
        String classname;
        if (args.length > 0) classname = args[0];
        else classname = "Ljq;";
        
        jq.Bootstrapping = true;
        jq.DontCompile = true;

        Unsafe.installRemapper(new Unsafe.Remapper() {
            public int addressOf(Object o) { return 0; }
            public jq_Type getType(Object o) { return Reflection.getJQType(o.getClass()); }
        });
        
        jq.assert(true); // initialize jq
        String classpath = System.getProperty("java.class.path")+
                           System.getProperty("path.separator")+
                           System.getProperty("sun.boot.class.path");
        PrimordialClassLoader.loader.addToClasspath(classpath);
        jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(classname);
        System.out.println("Loading "+c+"...");
        c.load();
        System.out.println("Verifying "+c+"...");
        c.verify();
        System.out.println("Preparing "+c+"...");
        c.prepare();
        dumpClass(System.out, c);
        //jq_Class c2 = (jq_Class)PrimordialClassLoader.loader.getOrCreateType("Ljava/lang/Exception;");
        //System.out.println(Run_Time.TypeCheck.isAssignable(c, c2));
        //System.out.println(Run_Time.TypeCheck.isAssignable(c2, c));
        //compileClass(System.out, c);
    }

    public static void compileClass(PrintStream out, jq_Class t) {
        Iterator it;
        for(it = new ArrayIterator(t.getDeclaredStaticMethods());
            it.hasNext(); ) {
            jq_StaticMethod c = (jq_StaticMethod)it.next();
            Compil3r.Reference.x86.x86ReferenceCompiler comp = new Compil3r.Reference.x86.x86ReferenceCompiler(c);
            comp.compile();
        }
    }
    
    public static void dumpType(PrintStream out, jq_Type t) {
        if (t.isClassType()) out.print("class ");
        if (t.isArrayType()) out.print("array ");
        if (t.isPrimitiveType()) out.print("primitive ");
        out.print(t.getName());
    }

    public static void dumpClass(PrintStream out, jq_Class t) {
        dumpType(out, t);
        out.println();
        out.println("state: "+t.getState());
        
        if (t.isLoaded()) {
            out.println("java class file version "+(int)t.getMajorVersion()+"."+(int)t.getMinorVersion());
            out.println("source file name: "+t.getSourceFile());
            out.print("access flags: ");
            if (t.isPublic()) out.print("public ");
            if (t.isFinal()) out.print("final ");
            if (t.isSpecial()) out.print("special ");
            if (t.isInterface()) out.print("interface ");
            if (t.isAbstract()) out.print("abstract ");
            if (t.isSynthetic()) out.print("synthetic ");
            if (t.isDeprecated()) out.print("deprecated ");
            out.println();
            out.println("superclass: "+t.getSuperclass().getName());
            Iterator it;
            out.print("known subclasses: ");
            for(it = new ArrayIterator(t.getSubClasses());
                it.hasNext(); ) {
                jq_Class c = (jq_Class)it.next();
                out.print(c.getName()+" ");
            }
            out.println();
            out.print("declared interfaces: ");
            for(it = new ArrayIterator(t.getDeclaredInterfaces());
                it.hasNext(); ) {
                jq_Class c = (jq_Class)it.next();
                out.print(c.getName()+" ");
            }
            out.println();
            out.print("declared instance fields: ");
            for(it = new ArrayIterator(t.getDeclaredInstanceFields());
                it.hasNext(); ) {
                jq_InstanceField c = (jq_InstanceField)it.next();
                out.print(c.getName()+" ");
            }
            out.println();
            out.print("declared static fields: ");
            for(it = new ArrayIterator(t.getDeclaredStaticFields());
                it.hasNext(); ) {
                jq_StaticField c = (jq_StaticField)it.next();
                out.print(c.getName()+" ");
            }
            out.println();
            out.print("declared instance methods: ");
            for(it = new ArrayIterator(t.getDeclaredInstanceMethods());
                it.hasNext(); ) {
                jq_InstanceMethod c = (jq_InstanceMethod)it.next();
                out.println(c.getName()+" ");
                out.println("method attributes:");
                for(Iterator it2 = c.getAttributes().keySet().iterator();
                    it2.hasNext(); ) {
                    Utf8 key = (Utf8)it2.next();
                    out.print("\t"+key);
                    byte[] val = t.getAttribute(key);
                    out.println(": "+((val!=null)?"(length "+val.length+")\t":"\t")+val);
                }
            }
            out.println();
            out.print("declared static methods: ");
            for(it = new ArrayIterator(t.getDeclaredStaticMethods());
                it.hasNext(); ) {
                jq_StaticMethod c = (jq_StaticMethod)it.next();
                out.println(c.getName()+" ");
                out.println("method attributes:");
                for(Iterator it2 = c.getAttributes().keySet().iterator();
                    it2.hasNext(); ) {
                    Utf8 key = (Utf8)it2.next();
                    out.print("\t"+key);
                    byte[] val = t.getAttribute(key);
                    out.println(": "+((val!=null)?"(length "+val.length+")\t":"\t")+val);
                }
            }
            out.println();
            out.print("class initializer: ");
            if (t.getClassInitializer() != null) out.println("present");
            else out.println("absent");
            out.println("constant pool size: "+t.getCPCount());
            out.println("attributes:");
            for(it = t.getAttributes();
                it.hasNext(); ) {
                Utf8 key = (Utf8)it.next();
                byte[] val = t.getAttribute(key);
                out.println("\t"+key+": (length "+val.length+")\t"+val);
            }
        }
        if (t.isPrepared()) {
            Iterator it;
            out.print("interfaces: ");
            for(it = new ArrayIterator(t.getInterfaces());
                it.hasNext(); ) {
                jq_Class c = (jq_Class)it.next();
                out.print(c+" ");
            }
            out.println();
            out.print("virtual methods: ");
            for(it = new ArrayIterator(t.getVirtualMethods());
                it.hasNext(); ) {
                jq_InstanceMethod c = (jq_InstanceMethod)it.next();
                out.print(c+" ");
            }
            out.println();
        }
        if (t.isSFInitialized()) {
        }
        if (t.isClsInitialized()) {
        }
        
    }
}

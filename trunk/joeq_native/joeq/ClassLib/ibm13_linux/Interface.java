/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Allocator.DefaultCodeAllocator;
import jq;

public final class Interface extends ClassLib.ClassLibInterface {

    /** Creates new Interface */
    public Interface() {}

    public static boolean USE_JOEQ_CLASSLIB = false;
    
    public void useJoeqClasslib(boolean b) { USE_JOEQ_CLASSLIB = b; }
    
    public java.lang.String getImplementationClassDesc(UTF.Utf8 desc) {
        if (USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/") ||
	                         desc.toString().startsWith("Lcom/ibm/jvm/")) {
            return "LClassLib/ibm13_linux/"+desc.toString().substring(1);
        }
        return null;
    }
    
    public java.util.Set bootstrapNullStaticFields() {
        java.util.Set nullStaticFields = new java.util.HashSet();
        nullStaticFields.add(Unsafe._remapper_object);
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("in", "Ljava/io/InputStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("out", "Ljava/io/PrintStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("err", "Ljava/io/PrintStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("props", "Ljava/util/Properties;"));
        nullStaticFields.add(Reflection._obj_trav);
        nullStaticFields.add(DefaultCodeAllocator._default_allocator);
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("loadedLibraryNames", "Ljava/util/Vector;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("systemNativeLibraries", "Ljava/util/Vector;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("nativeLibraryContext", "Ljava/util/Stack;"));
        jq_Class jq_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljq;");
        nullStaticFields.add(jq_class.getOrCreateStaticField("on_vm_startup", "Ljava/util/List;"));
        jq_Class launcher_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Launcher;");
        nullStaticFields.add(launcher_class.getOrCreateStaticField("launcher", "Lsun/misc/Launcher;"));
        //jq_Class urlclassloader_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader;");
        //nullStaticFields.add(urlclassloader_class.getOrCreateStaticField("extLoader", "Ljava/net/URLClassLoader;"));
        jq_Class zipfile_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
        nullStaticFields.add(zipfile_class.getOrCreateStaticField("inflaters", "Ljava/util/Vector;"));

	// we need to reinitialize the inflaters array on startup.
	Object[] args = { } ;
	jq_Method init_inflaters = zipfile_class.getOrCreateStaticMethod("init_inflaters", "()V");
	Bootstrap.MethodInvocation mi = new Bootstrap.MethodInvocation(init_inflaters, args);
	jq.on_vm_startup.add(mi);
	System.out.println("Added call to reinitialize java.util.zip.ZipFile.inflaters field on joeq startup: "+mi);

        return nullStaticFields;
    }
    
    public java.util.Set bootstrapNullInstanceFields() {
        java.util.Set nullInstanceFields = new java.util.HashSet();
	jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader$ClassFinder;");
	nullInstanceFields.add(k.getOrCreateInstanceField("name", "Ljava/lang/String;"));
        return nullInstanceFields;
    }
    
    public java.lang.Class createNewClass(Clazz.jq_Type f) {
        return ClassLib.ibm13_linux.java.lang.Class.createNewClass(ClassLib.ibm13_linux.java.lang.Class._class, f);
    }
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) {
        return ClassLib.ibm13_linux.java.lang.reflect.Constructor.createNewConstructor(ClassLib.ibm13_linux.java.lang.reflect.Constructor._class, f);
    }
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f) {
        return ClassLib.ibm13_linux.java.lang.reflect.Field.createNewField(ClassLib.ibm13_linux.java.lang.reflect.Field._class, f);
    }
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) {
        return ClassLib.ibm13_linux.java.lang.reflect.Method.createNewMethod(ClassLib.ibm13_linux.java.lang.reflect.Method._class, f);
    }
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f) {
        return (Clazz.jq_Field)Reflection.getfield_A(f, ClassLib.ibm13_linux.java.lang.reflect.Field._jq_field);
    }
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) {
        return (Clazz.jq_Initializer)Reflection.getfield_A(f, ClassLib.ibm13_linux.java.lang.reflect.Constructor._jq_init);
    }
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) {
        return (Clazz.jq_Method)Reflection.getfield_A(f, ClassLib.ibm13_linux.java.lang.reflect.Method._jq_method);
    }
    
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
        return (Scheduler.jq_Thread)Reflection.getfield_A(t, ClassLib.ibm13_linux.java.lang.Thread._jq_thread);
    }
    
    public Clazz.jq_Type getJQType(java.lang.Class k) {
        return (Clazz.jq_Type)Reflection.getfield_A(k, ClassLib.ibm13_linux.java.lang.Class._jq_type);
    }
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) {
        return ClassLib.ibm13_linux.java.lang.ClassLoader.getOrCreateType(cl, desc);
    }
    
    public void init_zipfile(java.util.zip.ZipFile o, java.lang.String name) throws java.io.IOException {
        ClassLib.ibm13_linux.java.util.zip.ZipFile.__init__(o, name);
    }
    
    public void init_inflater(java.util.zip.Inflater o, boolean nowrap) {
        ClassLib.ibm13_linux.java.util.zip.Inflater.__init__(o, nowrap);
    }
    
    public void initializeSystemClass() throws java.lang.Throwable {
        Reflection.invokestatic_V(ClassLib.ibm13_linux.java.lang.System._initializeSystemClass);
    }
    
    public void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable) throws java.io.FileNotFoundException {
        ClassLib.ibm13_linux.java.io.RandomAccessFile.open(dis, name, writeable);
    }
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) {
        ClassLib.ibm13_linux.java.lang.ClassLoader.unloadType(cl, t);
    }
    
}

/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_win32;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Allocator.DefaultCodeAllocator;

public final class Interface extends ClassLib.ClassLibInterface {

    /** Creates new Interface */
    public Interface() {}

    public java.lang.String getImplementationClassDesc(UTF.Utf8 desc) {
        if (desc.toString().startsWith("Ljava/")) {
            return "LClassLib/sun13_win32/"+desc.toString().substring(1);
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
        return nullStaticFields;
    }
    
    public java.lang.Class createNewClass(Clazz.jq_Type f) {
        return ClassLib.sun13_win32.java.lang.Class.createNewClass(ClassLib.sun13_win32.java.lang.Class._class, f);
    }
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) {
        return ClassLib.sun13_win32.java.lang.reflect.Constructor.createNewConstructor(ClassLib.sun13_win32.java.lang.reflect.Constructor._class, f);
    }
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f) {
        return ClassLib.sun13_win32.java.lang.reflect.Field.createNewField(ClassLib.sun13_win32.java.lang.reflect.Field._class, f);
    }
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) {
        return ClassLib.sun13_win32.java.lang.reflect.Method.createNewMethod(ClassLib.sun13_win32.java.lang.reflect.Method._class, f);
    }
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f) {
        return (Clazz.jq_Field)Reflection.getfield_A(f, ClassLib.sun13_win32.java.lang.reflect.Field._jq_field);
    }
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) {
        return (Clazz.jq_Initializer)Reflection.getfield_A(f, ClassLib.sun13_win32.java.lang.reflect.Constructor._jq_init);
    }
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) {
        return (Clazz.jq_Method)Reflection.getfield_A(f, ClassLib.sun13_win32.java.lang.reflect.Method._jq_method);
    }
    
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
        return (Scheduler.jq_Thread)Reflection.getfield_A(t, ClassLib.sun13_win32.java.lang.Thread._jq_thread);
    }
    
    public Clazz.jq_Type getJQType(java.lang.Class k) {
        return (Clazz.jq_Type)Reflection.getfield_A(k, ClassLib.sun13_win32.java.lang.Class._jq_type);
    }
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) {
        return ClassLib.sun13_win32.java.lang.ClassLoader.getOrCreateType(cl, desc);
    }
    
    public void init_zipfile(java.util.zip.ZipFile o, java.lang.String name) throws java.io.IOException {
        ClassLib.sun13_win32.java.util.zip.ZipFile.__init__(o, name);
    }
    
    public void initializeSystemClass() throws java.lang.Throwable {
        Reflection.invokestatic_V(ClassLib.sun13_win32.java.lang.System._initializeSystemClass);
    }
    
    public void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable) throws java.io.FileNotFoundException {
        ClassLib.sun13_win32.java.io.RandomAccessFile.open(dis, name, writeable);
    }
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) {
        ClassLib.sun13_win32.java.lang.ClassLoader.unloadType(cl, t);
    }
    
}

/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.ibm13_linux;

import Allocator.DefaultCodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Method;
import Main.jq;
import Run_Time.Reflection;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Interface extends ClassLib.ClassLibInterface {

    /** Creates new Interface */
    public Interface() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (USE_JOEQ_CLASSLIB && (desc.toString().startsWith("Ljava/") ||
                                  desc.toString().startsWith("Lcom/ibm/jvm/"))) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/Common/"+desc.toString().substring(1));
            java.util.LinkedList ll = new java.util.LinkedList();
            ll.add(u);
            u = UTF.Utf8.get("LClassLib/ibm13_linux/"+desc.toString().substring(1));
            ll.add(u);
            return ll.iterator();
        }
        return Util.NullIterator.INSTANCE;
    }
    
    public java.util.Set bootstrapNullStaticFields() {
        java.util.Set nullStaticFields = new java.util.HashSet();
        nullStaticFields.add(Unsafe._remapper_object);
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("in", "Ljava/io/InputStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("out", "Ljava/io/PrintStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("err", "Ljava/io/PrintStream;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("props", "Ljava/util/Properties;"));
        nullStaticFields.add(Reflection._obj_trav);
        nullStaticFields.add(Reflection._declaredFieldsCache);
        nullStaticFields.add(DefaultCodeAllocator._default_allocator);
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("loadedLibraryNames", "Ljava/util/Vector;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("systemNativeLibraries", "Ljava/util/Vector;"));
        nullStaticFields.add(PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("nativeLibraryContext", "Ljava/util/Stack;"));
        jq_Class jq_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LMain/jq;");
        nullStaticFields.add(jq_class.getOrCreateStaticField("on_vm_startup", "Ljava/util/List;"));
        jq_Class launcher_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Launcher;");
        nullStaticFields.add(launcher_class.getOrCreateStaticField("launcher", "Lsun/misc/Launcher;"));
        //jq_Class urlclassloader_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader;");
        //nullStaticFields.add(urlclassloader_class.getOrCreateStaticField("extLoader", "Ljava/net/URLClassLoader;"));
        jq_Class zipfile_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
        nullStaticFields.add(zipfile_class.getOrCreateStaticField("inflaters", "Ljava/util/Vector;"));
        jq_Class string_class = PrimordialClassLoader.getJavaLangString();
        nullStaticFields.add(string_class.getOrCreateStaticField("btcConverter", "Ljava/lang/ThreadLocal;"));
        nullStaticFields.add(string_class.getOrCreateStaticField("ctbConverter", "Ljava/lang/ThreadLocal;"));
        return nullStaticFields;
    }
    
    public java.util.Set bootstrapNullInstanceFields() {
        java.util.Set nullInstanceFields = new java.util.HashSet();
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader$ClassFinder;");
        nullInstanceFields.add(k.getOrCreateInstanceField("name", "Ljava/lang/String;"));
        return nullInstanceFields;
    }
    
    public void initializeDefaults() {
        // access the ISO-8859-1 character encoding, as it is used during bootstrapping
        try {
            String s = new String(new byte[0], 0, 0, "ISO-8859-1");
        } catch (java.io.UnsupportedEncodingException x) {}
        Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Lsun/io/CharToByteISO8859_1;");

        // we need to reinitialize the inflaters array on startup.
        Object[] args = { } ;
        jq_Class zipfile_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
        jq_Method init_inflaters = zipfile_class.getOrCreateStaticMethod("init_inflaters", "()V");
        Bootstrap.MethodInvocation mi = new Bootstrap.MethodInvocation(init_inflaters, args);
        jq.on_vm_startup.add(mi);
        System.out.println("Added call to reinitialize java.util.zip.ZipFile.inflaters field on joeq startup: "+mi);
    }
    
    public java.lang.Class createNewClass(Clazz.jq_Type f) {
        return ClassLib.Common.java.lang.Class.createNewClass(f);
    }
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) {
        return ClassLib.Common.java.lang.reflect.Constructor.createNewConstructor(f);
    }
    
    public void initNewConstructor(java.lang.reflect.Constructor dis, Clazz.jq_Initializer f) {
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Constructor.initNewConstructor((ClassLib.Common.java.lang.reflect.Constructor)o, f);
    }
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f) {
        return ClassLib.Common.java.lang.reflect.Field.createNewField(f);
    }
    
    public void initNewField(java.lang.reflect.Field dis, Clazz.jq_Field f) {
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Field.initNewField((ClassLib.Common.java.lang.reflect.Field)o, f);
    }
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) {
        return ClassLib.Common.java.lang.reflect.Method.createNewMethod(f);
    }
    
    public void initNewMethod(java.lang.reflect.Method dis, Clazz.jq_Method f) {
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Method.initNewMethod((ClassLib.Common.java.lang.reflect.Method)o, f);
    }
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f) {
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Field)o).jq_field;
    }
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) {
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Constructor)o).jq_init;
    }
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) {
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Method)o).jq_method;
    }
    
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
        if (jq.Bootstrapping) 
            return (Scheduler.jq_Thread)Reflection.obj_trav.getMappedInstanceFieldValue(t, java.lang.Thread.class, "jq_thread");
        java.lang.Object o = t;
        return ((ClassLib.Common.java.lang.Thread)o).jq_thread;
    }
    
    public Clazz.jq_Type getJQType(java.lang.Class k) {
        if (jq.Bootstrapping) 
            return Reflection.getJQType(k);
        java.lang.Object o = k;
        return ((ClassLib.Common.java.lang.Class)o).jq_type;
    }
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) {
        if (jq.Bootstrapping) {
            jq.Assert(cl == PrimordialClassLoader.loader);
            return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        }
        java.lang.Object o = cl;
        return ((ClassLib.Common.java.lang.ClassLoader)o).getOrCreateType(desc);
    }
    
    public void init_zipfile(java.util.zip.ZipFile dis, java.lang.String name) throws java.io.IOException {
        /*
        if (jq.Bootstrapping) {
            ClassLib.Common.java.util.zip.ZipFile.bootstrap_init(dis, name);
            return;
        }
         */
        java.lang.Object o = dis;
        ((ClassLib.Common.java.util.zip.ZipFile)o).__init__(name);
    }
    
    public void init_inflater(java.util.zip.Inflater o, boolean nowrap) {
        //ClassLib.Common.java.util.zip.Inflater.__init__(o, nowrap);
    }
    
    public void initializeSystemClass() throws java.lang.Throwable {
        ClassLib.Common.java.lang.System.initializeSystemClass();
    }
    
    public void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable) throws java.io.FileNotFoundException {
        java.lang.Object o = dis;
        ((ClassLib.Common.java.io.RandomAccessFile)o).open(name, writeable);
    }
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) {
        if (jq.Bootstrapping) {
            jq.Assert(cl == PrimordialClassLoader.loader);
            PrimordialClassLoader.loader.unloadBSType(t);
            return;
        }
        java.lang.Object o = cl;
        ((ClassLib.Common.java.lang.ClassLoader)o).unloadType(t);
    }
    
}

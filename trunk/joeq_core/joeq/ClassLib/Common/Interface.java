/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.Common;

import ClassLib.ClassLibInterface;
import Main.jq;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class Interface extends Bootstrap.ObjectTraverser {

    /** Creates new Interface */
    public Interface() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/Common/"+desc.toString().substring(1));
            return java.util.Collections.singleton(u).iterator();
        }
        return java.util.Collections.EMPTY_SET.iterator();
    }
    
    public static java.util.Set nullStaticFields = new java.util.HashSet();
    public static java.util.Set nullInstanceFields = new java.util.HashSet();
    
    public void initialize() {
        nullStaticFields.add(Run_Time.Unsafe._remapper_object);
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("in", "Ljava/io/InputStream;"));
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("out", "Ljava/io/PrintStream;"));
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("err", "Ljava/io/PrintStream;"));
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangSystem().getOrCreateStaticField("props", "Ljava/util/Properties;"));
        nullStaticFields.add(Run_Time.Reflection._obj_trav);
        nullStaticFields.add(Run_Time.Reflection._declaredFieldsCache);
        nullStaticFields.add(Allocator.DefaultCodeAllocator._default_allocator);
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("loadedLibraryNames", "Ljava/util/Vector;"));
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("systemNativeLibraries", "Ljava/util/Vector;"));
        nullStaticFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClassLoader().getOrCreateStaticField("nativeLibraryContext", "Ljava/util/Stack;"));
        Clazz.jq_Class jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("LMain/jq;");
        nullStaticFields.add(jq_class.getOrCreateStaticField("on_vm_startup", "Ljava/util/List;"));
    }
    
    public java.lang.Object mapStaticField(Clazz.jq_StaticField f) {
    	if (nullStaticFields.contains(f)) {
    		return null;
    	}
    	return super.mapStaticField(f);
    }
    public java.lang.Object mapInstanceField(java.lang.Object o, Clazz.jq_InstanceField f) {
    	if (nullInstanceFields.contains(f)) {
    		return null;
    	}
    	return super.mapInstanceField(o, f);
    }
    
    public java.lang.Class createNewClass(Clazz.jq_Type f) {
    	jq.Assert(!jq.Bootstrapping);
        return ClassLib.Common.java.lang.Class.createNewClass(f);
    }
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) {
    	jq.Assert(!jq.Bootstrapping);
        return ClassLib.Common.java.lang.reflect.Constructor.createNewConstructor(f);
    }
    
    public void initNewConstructor(java.lang.reflect.Constructor dis, Clazz.jq_Initializer f) {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Constructor.initNewConstructor((ClassLib.Common.java.lang.reflect.Constructor)o, f);
    }
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f) {
    	jq.Assert(!jq.Bootstrapping);
        return ClassLib.Common.java.lang.reflect.Field.createNewField(f);
    }
    
    public void initNewField(java.lang.reflect.Field dis, Clazz.jq_Field f) {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Field.initNewField((ClassLib.Common.java.lang.reflect.Field)o, f);
    }
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) {
    	jq.Assert(!jq.Bootstrapping);
        return ClassLib.Common.java.lang.reflect.Method.createNewMethod(f);
    }
    
    public void initNewMethod(java.lang.reflect.Method dis, Clazz.jq_Method f) {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = dis;
        ClassLib.Common.java.lang.reflect.Method.initNewMethod((ClassLib.Common.java.lang.reflect.Method)o, f);
    }
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f) {
        jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Field)o).jq_field;
    }
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) {
        jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Constructor)o).jq_init;
    }
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) {
        jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = f;
        return ((ClassLib.Common.java.lang.reflect.Method)o).jq_method;
    }
    
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
    	if (jq.Bootstrapping) {
	        Clazz.jq_InstanceField f = Bootstrap.PrimordialClassLoader.getJavaLangThread().getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
	        return (Scheduler.jq_Thread)this.mapInstanceField(t, f);
    	}
        jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = t;
        return ((ClassLib.Common.java.lang.Thread)o).jq_thread;
    }
    
    public Clazz.jq_Type getJQType(java.lang.Class k) {
        jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = k;
        return ((ClassLib.Common.java.lang.Class)o).jq_type;
    }
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = cl;
        return ((ClassLib.Common.java.lang.ClassLoader)o).getOrCreateType(desc);
    }
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = cl;
        ((ClassLib.Common.java.lang.ClassLoader)o).unloadType(t);
    }
    
    public void init_zipfile(java.util.zip.ZipFile dis, java.lang.String name) throws java.io.IOException {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = dis;
        ((ClassLib.Common.java.util.zip.ZipFile)o).__init__(name);
    }
    
    public void initializeSystemClass() throws java.lang.Throwable {
    	jq.Assert(!jq.Bootstrapping);
        ClassLib.Common.java.lang.System.initializeSystemClass();
    }
    
    public void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable) throws java.io.FileNotFoundException {
    	jq.Assert(!jq.Bootstrapping);
        java.lang.Object o = dis;
        ((ClassLib.Common.java.io.RandomAccessFile)o).open(name, writeable);
    }
    
}

/*
 * ClassLibInterface.java
 *
 * Created on December 11, 2001, 3:59 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib;

import Bootstrap.PrimordialClassLoader;
import jq;

public abstract class ClassLibInterface {

    public static final ClassLibInterface i;

    static {
        String classlibrary = System.getProperty("classlibrary");
        if (classlibrary == null) {
            String javaversion = System.getProperty("java.version");

	    String javavmversion = System.getProperty("java.vm.version");
	    String javavmvendor = System.getProperty("java.vm.vendor");
	    String javaruntimeversion = System.getProperty("java.runtime.version");
            String osarch = System.getProperty("os.arch");
            String osname = System.getProperty("os.name");

            if (osarch.equals("x86")) {
            } else if (osarch.equals("i386")) {
	    } else {
                System.err.println("Warning: architecture "+osarch+" is not yet supported.");
            }
            if (javavmvendor.equals("Sun Microsystems Inc.")) {
		if (javaruntimeversion.equals("1.3.1_01")) {
		    classlibrary = "sun13_";
		} else if (javaruntimeversion.equals("1.4.0-b92")) {
                    classlibrary = "sun14_";
		} else {
		    classlibrary = "sun13_";
		    System.err.println("Warning: class library version "+javaruntimeversion+" is not yet supported, trying default "+classlibrary);
		}
            } else if (javavmvendor.equals("IBM Corporation")) {
		if (javaruntimeversion.equals("1.3.0")) {
		    classlibrary = "ibm13_";
		} else {
		    classlibrary = "ibm13_";
		    System.err.println("Warning: class library version "+javaruntimeversion+" is not yet supported, trying default "+classlibrary);
		}
            } else {
                classlibrary = "sun13_";
                System.err.println("Warning: vm vendor "+javavmvendor+" is not yet supported, trying default "+classlibrary);
            }
            if (osname.startsWith("Windows")) {
                classlibrary += "win32";
            } else if (osname.equals("Linux")) {
                classlibrary += "linux";
            } else {
                classlibrary += "win32";
                System.err.println("Warning: OS "+osname+" is not yet supported, trying "+classlibrary);
            }
        }
        ClassLibInterface f = null;
        try {
            Class c = Class.forName("ClassLib."+classlibrary+".Interface");
            f = (ClassLibInterface)c.newInstance();
        } catch (java.lang.ClassNotFoundException x) {
            System.err.println("Cannot find class library interface "+classlibrary+": "+x);
            System.err.println("Please check the version of your virtual machine.");
            System.exit(-1);
        } catch (java.lang.InstantiationException x) {
            System.err.println("Cannot instantiate class library interface "+classlibrary+": "+x);
            System.exit(-1);
        } catch (java.lang.IllegalAccessException x) {
            System.err.println("Cannot access class library interface "+classlibrary+": "+x);
            System.exit(-1);
        }
        i = f;
    }

    protected ClassLibInterface() {}

    // java.lang.Thread
    public abstract Scheduler.jq_Thread getJQThread(java.lang.Thread t) ;

    // java.lang.Class
    public abstract java.lang.Class createNewClass(Clazz.jq_Type f) ;
    public abstract Clazz.jq_Type getJQType(java.lang.Class k) ;
    
    // java.lang.ClassLoader
    public abstract Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) ;
    public abstract void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) ;
    
    // java.lang.System
    public abstract void initializeSystemClass()
        throws java.lang.Throwable;
    
    // java.lang.reflect.Field
    public abstract java.lang.reflect.Field createNewField(Clazz.jq_Field f) ;
    public abstract Clazz.jq_Field getJQField(java.lang.reflect.Field f) ;

    // java.lang.reflect.Method
    public abstract java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) ;
    public abstract Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) ;
    
    // java.lang.reflect.Constructor
    public abstract java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) ;
    public abstract Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) ;
    
    // java.io.RandomAccessFile
    public abstract void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable)
        throws java.io.FileNotFoundException;
    public static final void open_static(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable)
        throws java.io.FileNotFoundException { ClassLibInterface.i.open(dis, name, writeable); }
    
    // java.util.zip.ZipFile
    public abstract void init_zipfile(java.util.zip.ZipFile o, java.lang.String name)
        throws java.io.IOException;
    // java.util.zip.Inflater
    public abstract void init_inflater(java.util.zip.Inflater o, boolean nowrap);

    public abstract java.util.Set bootstrapNullStaticFields() ;
    public abstract java.util.Set bootstrapNullInstanceFields() ;
    public abstract java.lang.String getImplementationClassDesc(UTF.Utf8 desc) ;
    public abstract void useJoeqClasslib(boolean b);
    
    public static final Clazz.jq_Class _class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("LClassLib/ClassLibInterface;");
    
}

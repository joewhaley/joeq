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
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Clazz.jq_Member;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Clazz.jq_NameAndDesc;
import Run_Time.SystemInterface;
import UTF.Utf8;
import Main.jq;

public abstract class ClassLibInterface {

    public static boolean USE_JOEQ_CLASSLIB;
    public final void useJoeqClasslib(boolean b) { USE_JOEQ_CLASSLIB = b; }
    
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
		    if (javaruntimeversion.startsWith("1.4")) {
			classlibrary = "sun14_";
		    } else {
			classlibrary = "sun13_";
		    }
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
    public abstract void initNewField(java.lang.reflect.Field o, Clazz.jq_Field f) ;
    public abstract Clazz.jq_Field getJQField(java.lang.reflect.Field f) ;

    // java.lang.reflect.Method
    public abstract java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) ;
    public abstract void initNewMethod(java.lang.reflect.Method o, Clazz.jq_Method f) ;
    public abstract Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) ;
    
    // java.lang.reflect.Constructor
    public abstract java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) ;
    public abstract void initNewConstructor(java.lang.reflect.Constructor o, Clazz.jq_Initializer f) ;
    public abstract Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) ;
    
    // java.io.RandomAccessFile
    public abstract void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable)
        throws java.io.FileNotFoundException;
    
    // java.util.zip.ZipFile
    public abstract void init_zipfile(java.util.zip.ZipFile o, java.lang.String name)
        throws java.io.IOException;
    public static final void init_zipfile_static(java.util.zip.ZipFile o, java.lang.String name)
        throws java.io.IOException { ClassLibInterface.i.init_zipfile(o, name); }
    // java.util.zip.Inflater
    public abstract void init_inflater(java.util.zip.Inflater o, boolean nowrap);

    public abstract java.util.Set bootstrapNullStaticFields() ;
    public abstract java.util.Set bootstrapNullInstanceFields() ;
    public abstract java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) ;
    public abstract void initializeDefaults() ;
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClassLib/ClassLibInterface;");
    
    public static /*final*/ boolean TRACE = false;
    
    // utility functions
    public static jq_NameAndDesc convertClassLibNameAndDesc(jq_Class k, jq_NameAndDesc t) {
        Utf8 d = convertClassLibDesc(t.getDesc());
        Utf8 n = t.getName();
        if (k.getDesc().toString().endsWith("/java/lang/Object;")) {
            // trim initial "_", if it exists.
            String s = n.toString();
            if (s.charAt(0) == '_') {
                n = Utf8.get(s.substring(1));
                if (TRACE) SystemInterface.debugmsg("special case for java.lang.Object: "+n+" "+d);
                return new jq_NameAndDesc(n, d);
            }
        }
        if (d == t.getDesc())
            return t;
        else
            return new jq_NameAndDesc(n, d);
    }
    
    public static Utf8 convertClassLibDesc(Utf8 desc) {
        return Utf8.get(convertClassLibDesc(desc.toString()));
    }
    
    public static String convertClassLibDesc(String desc) {
        int i = desc.indexOf("ClassLib/");
        if (i != -1) {
            for (;;) {
                int m = desc.indexOf(';', i+10);
                if (m == -1) break;
                int j = desc.indexOf('/', i+10);
                if (j == -1 || j > m) break;
                int k = desc.indexOf(';', j);
                String t = desc.substring(j+1, k).replace('/','.');
                try {
                    Class.forName(t);
                    desc = desc.substring(0, i) + desc.substring(j+1);
                } catch (ClassNotFoundException x) {
                }
                i = desc.indexOf("ClassLib/", i+1);
                if (i == -1) break;
            }
        }
        return desc;
    }
    
    public static jq_Member convertClassLibCPEntry(jq_Member t) {
        jq_NameAndDesc u1 = convertClassLibNameAndDesc(t.getDeclaringClass(), t.getNameAndDesc());
        Utf8 u2 = convertClassLibDesc(t.getDeclaringClass().getDesc());
        if (u1 == t.getNameAndDesc() && u2 == t.getDeclaringClass().getDesc())
            return t;
        else {
            jq_Class c;
            if (u2 != t.getDeclaringClass().getDesc())
                c = (jq_Class)ClassLib.ClassLibInterface.i.getOrCreateType(t.getDeclaringClass().getClassLoader(), u2);
            else
                c = t.getDeclaringClass();
            if (t instanceof jq_InstanceField) {
                return c.getOrCreateInstanceField(u1);
            } else if (t instanceof jq_StaticField) {
                return c.getOrCreateStaticField(u1);
            } else if (t instanceof jq_InstanceMethod) {
                return c.getOrCreateInstanceMethod(u1);
            } else if (t instanceof jq_StaticMethod) {
                return c.getOrCreateStaticMethod(u1);
            } else {
                jq.UNREACHABLE(); return null;
            }
        }
    }
    
    public static jq_Reference convertClassLibCPEntry(jq_Reference t) {
        Utf8 u = convertClassLibDesc(t.getDesc());
        if (u == t.getDesc())
            return t;
        else
            return (jq_Reference)ClassLib.ClassLibInterface.i.getOrCreateType(t.getClassLoader(), u);
    }

}

/*
 * ClassLibInterface.java
 *
 * Created on December 11, 2001, 3:59 PM
 *
 */

package ClassLib;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Member;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Main.jq;
import Run_Time.Debug;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ClassLibInterface {

    public static boolean USE_JOEQ_CLASSLIB;
    public static final void useJoeqClasslib(boolean b) { USE_JOEQ_CLASSLIB = b; }
    
    public static final ClassLib.Common.Interface DEFAULT;

    /* Try the three current possibilities for the ClassLibInterface.
       This would probably be rather more general with some kind of
       iterator, but it does for now. */
    static {
        ClassLib.Common.Interface f = null;
        String classlibinterface = System.getProperty("joeq.classlibinterface");
        boolean nullVM = jq.nullVM || System.getProperty("joeq.nullvm") != null;

        if (classlibinterface != null) {
            f = attemptClassLibInterface(classlibinterface);
        }
        if (nullVM) {
            f = new ClassLib.Common.NullInterfaceImpl();
        }
        if (f == null) {
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
                } else if (javavmvendor.equals("Apple Computer, Inc.")) {
                    if (javaruntimeversion.equals("1.3.1")) {
                        classlibrary = "apple13_";
                    } else {
                        classlibrary = "apple13_";
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
                } else if (osname.equals("Mac OS X")) {
                    classlibrary += "osx";
                } else {
                    classlibrary += "win32";
                    System.err.println("Warning: OS "+osname+" is not yet supported, trying "+classlibrary);
                }
            }
            f = attemptClassLibInterface("ClassLib."+classlibrary+".Interface");
        }
        if (f == null) {
            f = new ClassLib.Common.NullInterfaceImpl();
        }
        
        DEFAULT = f;
    }

    private static ClassLib.Common.Interface attemptClassLibInterface(String s) {
        try {
            Class c = Class.forName(s);
            return (ClassLib.Common.Interface)c.newInstance();
        } catch (java.lang.ClassNotFoundException x) {
            System.err.println("Cannot find class library interface "+s+": "+x);
            System.err.println("Please check the version of your virtual machine.");
        } catch (java.lang.InstantiationException x) {
            System.err.println("Cannot instantiate class library interface "+s+": "+x);
        } catch (java.lang.IllegalAccessException x) {
            System.err.println("Cannot access class library interface "+s+": "+x);
        }
        return null;
    }

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
                if (TRACE) Debug.writeln("special case for java.lang.Object: "+n+" "+d);
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
                    Class.forName(t, false, ClassLibInterface.class.getClassLoader());
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
                c = (jq_Class)PrimordialClassLoader.getOrCreateType(t.getDeclaringClass().getClassLoader(), u2);
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
            return (jq_Reference)PrimordialClassLoader.getOrCreateType(t.getClassLoader(), u);
    }
    
    public static void init_zipfile_static(java.util.zip.ZipFile zf, java.lang.String s) {
        try {
            ClassLibInterface.DEFAULT.init_zipfile(zf, s);
        } catch (java.io.IOException x) {
            System.err.println("Note: cannot reopen zip file "+s);
        }
    }
    
    public static void init_inflater_static(java.util.zip.Inflater i, boolean nowrap)
        throws java.io.IOException {
        ClassLibInterface.DEFAULT.init_inflater(i, nowrap);
    }
}

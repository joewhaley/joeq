/*
 * ClassLoader.java
 *
 * Created on January 29, 2001, 11:16 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun14_win32.java.lang;

import Clazz.jq_ClassFileConstants;
import Clazz.jq_Class;
import Clazz.jq_Type;
import Clazz.jq_Primitive;
import Clazz.jq_Array;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticMethod;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Clazz.jq_CompiledCode;
import Run_Time.Reflection;
import Run_Time.StackWalker;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;
import jq;
import java.io.ByteArrayInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Map;
import java.security.ProtectionDomain;

public abstract class ClassLoader {
    
    // additional instance field
    private final Map/*<Utf8, jq_Type>*/ desc2type = null;

    // overridden instance method
    static void addClass(java.lang.ClassLoader dis, java.lang.Class c) {}

    // overridden constructors
    protected static void __init__(java.lang.ClassLoader dis, java.lang.ClassLoader parent) {
	java.lang.SecurityManager security = java.lang.System.getSecurityManager();
	if (security != null) {
	    security.checkCreateClassLoader();
	}
        jq.assert(parent != null);
        Reflection.putfield_A(dis, _parent, parent);
        Map m = new HashMap();
        Reflection.putfield_A(dis, _desc2type, m);
        Reflection.putfield_Z(dis, _initialized, true);
    }
    protected static void __init__(java.lang.ClassLoader dis) {
	java.lang.SecurityManager security = java.lang.System.getSecurityManager();
	if (security != null) {
	    security.checkCreateClassLoader();
	}
        java.lang.ClassLoader parent = getSystemClassLoader(_class);
        jq.assert(parent != null);
        Reflection.putfield_A(dis, _parent, parent);
        Map m = new HashMap();
        Reflection.putfield_A(dis, _desc2type, m);
        Reflection.putfield_Z(dis, _initialized, true);
    }

    // overridden methods.
    public static java.lang.ClassLoader getSystemClassLoader(jq_Class clazz) {
        java.lang.ClassLoader scl = PrimordialClassLoader.loader;
        Reflection.putstatic_A(_scl, scl);
	if (scl == null) {
	    return null;
	}
	java.lang.SecurityManager sm = java.lang.System.getSecurityManager();
	if (sm != null) {
	    java.lang.ClassLoader ccl = getCallerClassLoader(clazz);
            if (ccl != null && ccl != scl) {
                try {
                    if (!Reflection.invokeinstance_Z(_isAncestor, scl, ccl)) {
                        sm.checkPermission((java.lang.RuntimePermission)Reflection.invokestatic_A(_getGetClassLoaderPerm));
                    }
                } catch (java.lang.Error x) {
                    throw x;
                } catch (java.lang.Throwable x) {
                    jq.UNREACHABLE();
                }
	    }
	}
	return scl;
    }
    
    // native method implementations.
    private static java.lang.Class defineClass0(java.lang.ClassLoader dis, java.lang.String name, byte[] b, int off, int len,
                                                ProtectionDomain pd) {
        // define a new class based on given name and class file structure
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(b, off, len));
        // TODO: what should we do about protection domain???  ignore it???
        if (name == null) throw new ClassFormatError("name cannot be null when defining class");
        if (name.startsWith("[")) throw new ClassFormatError("cannot define array class with defineClass: "+name);
        Utf8 desc = Utf8.get("L"+name.replace('.','/')+";");
        if (getType(dis, desc) != null)
            throw new ClassFormatError("class "+name+" already defined");
        jq_Class c = jq_Class.newClass(dis, desc);
        Map desc2type = (Map)Reflection.getfield_A(dis, _desc2type);
        desc2type.put(desc, c);
        c.load(in);
        //in.close();
        return Reflection.getJDKType(c);
    }
    private static void resolveClass0(java.lang.ClassLoader dis, java.lang.Class c) {
        jq_Type t = (jq_Type)Reflection.getfield_A(c, Class._jq_type);
        t.load(); t.verify(); t.prepare();
    }
    private static java.lang.Class findBootstrapClass(java.lang.ClassLoader dis, java.lang.String name) throws ClassNotFoundException {
        jq.assert(dis == PrimordialClassLoader.loader);
        if (!name.startsWith("[")) name = "L"+name+";";
        Utf8 desc = Utf8.get(name.replace('.','/'));
        jq_Type k;
        k = getOrCreateType(dis, desc);
        try {
            k.load();
        } catch (NoClassDefFoundError x) {
            unloadType(dis, k);
            throw new ClassNotFoundException(name);
        }
        return Reflection.getJDKType(k);
    }
    protected static final java.lang.Class findLoadedClass(java.lang.ClassLoader dis, java.lang.String name) {
        if (!name.startsWith("[")) name = "L"+name+";";
        Utf8 desc = Utf8.get(name.replace('.','/'));
        jq_Type t;
        t = getType(dis, desc);
        if (t == null) return null;
        t.load();
        return Reflection.getJDKType(t);
    }
    static java.lang.ClassLoader getCallerClassLoader(jq_Class clazz) {
        StackWalker sw = new StackWalker(0, Unsafe.EBP());
        sw.gotoNext(); sw.gotoNext(); sw.gotoNext();
        jq_CompiledCode cc = sw.getCode();
        if (cc == null) return null;
        return (java.lang.ClassLoader)cc.getMethod().getDeclaringClass().getClassLoader();
    }

    // additional methods
    public static jq_Type getType(java.lang.ClassLoader dis, Utf8 desc) {
        jq.assert(!jq.Bootstrapping);
        Map desc2type = (Map)Reflection.getfield_A(dis, _desc2type);
        jq_Type t = (jq_Type)desc2type.get(desc);
        return t;
    }
    public static jq_Type getOrCreateType(java.lang.ClassLoader dis, Utf8 desc) {
        if (jq.Bootstrapping) return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        Map desc2type = (Map)Reflection.getfield_A(dis, _desc2type);
        jq_Type t = (jq_Type)desc2type.get(desc);
        if (t == null) {
            if (desc.isDescriptor(jq_ClassFileConstants.TC_CLASS)) {
                desc2type.put(desc, t = jq_Class.newClass(dis, desc));
            } else {
                if (!desc.isDescriptor(jq_ClassFileConstants.TC_ARRAY))
                    jq.UNREACHABLE("bad descriptor! "+desc);
                Utf8 elementDesc = desc.getArrayElementDescriptor();
                jq_Type elementType;
                elementType = getOrCreateType(dis, elementDesc); // recursion
                desc2type.put(desc, t = jq_Array.newArray(desc, dis, elementType));
            }
        }
        return t;
    }
    public static void unloadType(java.lang.ClassLoader dis, jq_Type t) {
        if (jq.Bootstrapping) {
            PrimordialClassLoader.loader.unloadBSType(t);
            return;
        }
        Map desc2type = (Map)Reflection.getfield_A(dis, _desc2type);
        desc2type.remove(t.getDesc());
    }
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/ClassLoader;");
    public static final jq_InstanceField _desc2type = _class.getOrCreateInstanceField("desc2type", "Ljava/util/Map;");
    public static final jq_InstanceField _parent = _class.getOrCreateInstanceField("parent", "Ljava/lang/ClassLoader;");
    public static final jq_InstanceField _initialized = _class.getOrCreateInstanceField("initialized", "Z");
    public static final jq_StaticField _scl = _class.getOrCreateStaticField("scl", "Ljava/lang/ClassLoader;");
    //public static final jq_InstanceMethod _getType = _class.getOrCreateInstanceMethod("getType", "(LUTF/Utf8;)LClazz/jq_Type;");
    //public static final jq_InstanceMethod _getOrCreateType = _class.getOrCreateInstanceMethod("getOrCreateType", "(LUTF/Utf8;)LClazz/jq_Type;");
    public static final jq_InstanceMethod _isAncestor = _class.getOrCreateInstanceMethod("isAncestor", "(Ljava/lang/ClassLoader;)Z");
    public static final jq_StaticMethod _getGetClassLoaderPerm = _class.getOrCreateStaticMethod("getGetClassLoaderPerm", "()Ljava/lang/RuntimePermission;");
    public static final jq_StaticField _loadedLibraryNames = _class.getOrCreateStaticField("loadedLibraryNames", "Ljava/util/Vector;");
    public static final jq_StaticField _systemNativeLibraries = _class.getOrCreateStaticField("systemNativeLibraries", "Ljava/util/Vector;");
    public static final jq_InstanceField _nativeLibraries = _class.getOrCreateInstanceField("nativeLibraries", "Ljava/util/Vector;");
    public static final jq_StaticField _nativeLibraryContext = _class.getOrCreateStaticField("nativeLibraryContext", "Ljava/util/Stack;");
}

/*
 * ClassLoader.java
 *
 * Created on January 29, 2001, 11:16 AM
 *
 */

package ClassLib.Common.java.lang;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.security.ProtectionDomain;
import java.util.HashMap;
import java.util.Map;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_CompiledCode;
import Clazz.jq_Type;
import Main.jq;
import Memory.CodeAddress;
import Memory.StackAddress;
import Run_Time.Reflection;
import Run_Time.StackWalker;
import Run_Time.Unsafe;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ClassLoader {
    
    private boolean initialized;
    private java.lang.ClassLoader parent;
    private static ClassLoader scl;
    
    // additional instance field
    private final Map/*<Utf8, jq_Type>*/ desc2type;

    // overridden instance method
    void addClass(java.lang.Class c) {}

    // overridden constructors
    protected ClassLoader(java.lang.ClassLoader parent) {
        java.lang.SecurityManager security = java.lang.System.getSecurityManager();
        if (security != null) {
            security.checkCreateClassLoader();
        }
        jq.Assert(parent != null);
        this.parent = parent;
        Map m = new HashMap();
        this.desc2type = m;
        this.initialized = true;
    }
    protected ClassLoader() {
        java.lang.SecurityManager security = java.lang.System.getSecurityManager();
        if (security != null) {
            security.checkCreateClassLoader();
        }
        java.lang.ClassLoader parent = getSystemClassLoader();
        jq.Assert(parent != null);
        this.parent = parent;
        Map m = new HashMap();
        this.desc2type = m;
        this.initialized = true;
    }

    native boolean isAncestor(ClassLoader cl);
    static native RuntimePermission getGetClassLoaderPerm();
    public native Class loadClass(java.lang.String name);
    
    // overridden methods.
    public static java.lang.ClassLoader getSystemClassLoader() {
        java.lang.Object o = PrimordialClassLoader.loader;
        scl = (ClassLoader)o;
        if (scl == null) {
            return null;
        }
        java.lang.SecurityManager sm = java.lang.System.getSecurityManager();
        if (sm != null) {
            ClassLoader ccl = getCallerClassLoader();
            if (ccl != null && ccl != scl) {
                try {
                    if (!scl.isAncestor(ccl)) {
                        sm.checkPermission(getGetClassLoaderPerm());
                    }
                } catch (java.lang.Error x) {
                    throw x;
                } catch (java.lang.Throwable x) {
                    jq.UNREACHABLE();
                }
            }
        }
        o = scl;
        return (java.lang.ClassLoader)o;
    }
    
    // native method implementations.
    private java.lang.Class defineClass0(java.lang.String name, byte[] b, int off, int len,
                                         ProtectionDomain pd) {
        // define a new class based on given name and class file structure
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(b, off, len));
        // TODO: what should we do about protection domain???  ignore it???
        if (name == null) throw new ClassFormatError("name cannot be null when defining class");
        if (name.startsWith("[")) throw new ClassFormatError("cannot define array class with defineClass: "+name);
        Utf8 desc = Utf8.get("L"+name.replace('.','/')+";");
        if (this.getType(desc) != null)
            throw new ClassFormatError("class "+name+" already defined");
        java.lang.Object o = this;
        jq_Class c = jq_Class.newClass((java.lang.ClassLoader)o, desc);
        Map desc2type = this.desc2type;
        desc2type.put(desc, c);
        c.load(in);
        //in.close();
        return Reflection.getJDKType(c);
    }
    private void resolveClass0(Class c) {
        jq_Type t = c.jq_type;
        t.load(); t.verify(); t.prepare();
    }
    private java.lang.Class findBootstrapClass(java.lang.String name) throws ClassNotFoundException {
        java.lang.Object o = PrimordialClassLoader.loader;
        jq.Assert(this == o);
        if (!name.startsWith("[")) name = "L"+name+";";
        Utf8 desc = Utf8.get(name.replace('.','/'));
        jq_Type k;
        k = this.getOrCreateType(desc);
        try {
            k.load();
        } catch (NoClassDefFoundError x) {
            this.unloadType(k);
            throw new ClassNotFoundException(name);
        }
        return Reflection.getJDKType(k);
    }
    protected final java.lang.Class findLoadedClass(java.lang.String name) {
        if (!name.startsWith("[")) name = "L"+name+";";
        Utf8 desc = Utf8.get(name.replace('.','/'));
        jq_Type t;
        t = this.getType(desc);
        if (t == null) return null;
        t.load();
        return Reflection.getJDKType(t);
    }
    static ClassLoader getCallerClassLoader() {
        StackWalker sw = new StackWalker(CodeAddress.min(), StackAddress.getBasePointer());
        sw.gotoNext(); sw.gotoNext(); sw.gotoNext();
        jq_CompiledCode cc = sw.getCode();
        if (cc == null) return null;
        java.lang.Object o = cc.getMethod().getDeclaringClass().getClassLoader();
        return (ClassLoader)o;
    }

    // additional methods
    public jq_Type getType(Utf8 desc) {
        jq.Assert(!jq.Bootstrapping);
        Map desc2type = this.desc2type;
        jq_Type t = (jq_Type)desc2type.get(desc);
        return t;
    }
    public static jq_Type getOrCreateType(java.lang.ClassLoader loader, Utf8 desc) {
        java.lang.Object o = loader;
        return ((ClassLoader)o).getOrCreateType(desc);
    }
    public jq_Type getOrCreateType(Utf8 desc) {
        if (jq.Bootstrapping) return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        Map desc2type = this.desc2type;
        jq_Type t = (jq_Type)desc2type.get(desc);
        if (t == null) {
            if (desc.isDescriptor(jq_ClassFileConstants.TC_CLASS)) {
                java.lang.Object o = this;
                t = jq_Class.newClass((java.lang.ClassLoader)o, desc);
                desc2type.put(desc, t);
            } else {
                if (!desc.isDescriptor(jq_ClassFileConstants.TC_ARRAY))
                    jq.UNREACHABLE("bad descriptor! "+desc);
                Utf8 elementDesc = desc.getArrayElementDescriptor();
                jq_Type elementType;
                elementType = this.getOrCreateType(elementDesc); // recursion
                java.lang.Object o = this;
                t = jq_Array.newArray(desc, (java.lang.ClassLoader)o, elementType);
                desc2type.put(desc, t);
            }
        }
        return t;
    }
    public void unloadType(jq_Type t) {
        if (jq.Bootstrapping) {
            PrimordialClassLoader.loader.unloadBSType(t);
            return;
        }
        Map desc2type = this.desc2type;
        desc2type.remove(t.getDesc());
    }
    
}

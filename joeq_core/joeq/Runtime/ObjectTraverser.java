/*
 * ObjectTraverser.java
 *
 * Created on January 14, 2001, 11:38 AM
 *
 */

package Bootstrap;

import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Member;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_Type;
import Main.jq;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Scheduler.jq_Thread;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class ObjectTraverser {

    protected HashMap mapped_objects = new HashMap();
    
    public java.lang.Object mapStaticField(jq_StaticField f) {
    	return NO_OBJECT;
    }
    public java.lang.Object mapInstanceField(java.lang.Object o, jq_InstanceField f) {
        jq_Class c = f.getDeclaringClass();
        String fieldName = f.getName().toString();
        if (c == PrimordialClassLoader.getJavaLangClass()) {
            if (fieldName.equals("jq_type"))
                return Reflection.getJQType((java.lang.Class)o);
            if (fieldName.equals("signers"))
                return null;
                //return ((java.lang.Class)o).getSigners();
            if (fieldName.equals("protection_domain"))
                return null;
                //return ((java.lang.Class)o).getProtectionDomain();
        } else if (c == jq_Type._class) {
            if (o == jq_Reference.jq_NullType.NULL_TYPE)
                return null;
            if (o == Compil3r.Quad.BytecodeToQuad.jq_ReturnAddressType.INSTANCE)
                return null;
            if (fieldName.equals("class_object"))
                return Reflection.getJDKType((jq_Type)o);
        } else if (c == PrimordialClassLoader.getJavaLangReflectField()) {
            if (fieldName.equals("jq_field"))
                return Reflection.getJQMember((java.lang.reflect.Field)o);
        } else if (c == PrimordialClassLoader.getJavaLangReflectMethod()) {
            if (fieldName.equals("jq_method"))
                return Reflection.getJQMember((java.lang.reflect.Method)o);
        } else if (c == PrimordialClassLoader.getJavaLangReflectConstructor()) {
            if (fieldName.equals("jq_init"))
                return Reflection.getJQMember((java.lang.reflect.Constructor)o);
        } else if (!Clazz.jq_Member.USE_MEMBER_OBJECT_FIELD &&
                   (c == PrimordialClassLoader.loader.getBSType("LClazz/jq_Member;"))) {
            if (fieldName.equals("member_object")) {
                // reflection returns different objects every time!
                // cache one and use it from now on.
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = Reflection.getJDKMember((jq_Member)o));
                return o2;
            }
        } else if (c == PrimordialClassLoader.getJavaLangThread()) {
            if (fieldName.equals("jq_thread")) {
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = new jq_Thread((Thread)o));
                return o2;
            }
            /***
            if (fieldName.equals("threadLocals")) {
                return java.util.Collections.EMPTY_MAP;
            }
            if (fieldName.equals("inheritableThreadLocals")) {
                return java.util.Collections.EMPTY_MAP;
            }
            ***/
        } else if (c == PrimordialClassLoader.getJavaLangClassLoader()) {
            if (o == PrimordialClassLoader.loader) {
                if (fieldName.equals("parent"))
                    return null;
                if (fieldName.equals("desc2type"))
                    return getInstanceFieldValue_reflection(o, PrimordialClassLoader.class, "bs_desc2type");
            } else if (fieldName.equals("desc2type")) {
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
			    Class c2 = Reflection.getJDKType(c);
                Vector classes = (Vector)getInstanceFieldValue_reflection(o, c2, "classes");
                HashMap desc2type = new HashMap();
                Iterator i = classes.iterator();
                while (i.hasNext()) {
                    Class c3 = (Class)i.next();
                    jq_Type t = Reflection.getJQType(c3);
                    desc2type.put(t.getDesc(), t);
                }
                mapped_objects.put(o, desc2type);
                return desc2type;
            }
        } else if (c == PrimordialClassLoader.getJavaLangRefFinalizer()) {
            // null out all fields of finalizer objects.
            return null;
        } else if (TypeCheck.isAssignable(c, PrimordialClassLoader.getJavaLangThrowable())) {
            if (fieldName.equals("backtrace")) {
                // sun jvm crashes when using reflection on java.lang.Throwable.backtrace
                return null;
                /*
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = new int[0]);
                return o2;
                */
            }
        }
        else if (c == PrimordialClassLoader.loader.getBSType("Ljava/util/zip/ZipFile;")) {
            if (fieldName.equals("raf"))
                return null;
            if (fieldName.equals("entries"))
                return null;
            if (fieldName.equals("cenpos"))
                return null;
            if (fieldName.equals("pos"))
                return null;
        }
        else if (c == PrimordialClassLoader.loader.getBSType("Ljava/util/zip/Inflater;")) {
        	// Inflater objects are reinitialized on VM startup.
        	return null;
        }

    	return NO_OBJECT;
    }
    public java.lang.Object mapValue(java.lang.Object o) {
        if (o == ClassLoader.getSystemClassLoader()) {
            return PrimordialClassLoader.loader;
        }
        if (o instanceof java.util.zip.ZipFile) {
            Object o2 = mapped_objects.get(o);
            if (o2 != null) return o;
            mapped_objects.put(o, o);
            String name = ((java.util.zip.ZipFile)o).getName();
            
            // we need to reopen the ZipFile on VM startup
            Object[] args = { o, name };
            jq_Method zip_open = ClassLibInterface._class.getOrCreateStaticMethod("init_zipfile_static", "(Ljava/util/zip/ZipFile;Ljava/lang/String;)V");
            MethodInvocation mi = new MethodInvocation(zip_open, args);
            jq.on_vm_startup.add(mi);
            System.out.println("Added call to reopen zip file "+name+" on joeq startup: "+mi);
        }
        return o;
    }

    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    public static final java.lang.Object NO_OBJECT = new java.lang.Object();
    
    public Object getStaticFieldValue(jq_StaticField f) {
        java.lang.Object result = this.mapStaticField(f);
        if (result != NO_OBJECT) return this.mapValue(result);
        // get the value via real reflection.
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        return getStaticFieldValue_reflection(c, fieldName);
    }
    public Object getStaticFieldValue_reflection(Class c, String fieldName) {
        if (TRACE) out.println("Getting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        if (f2 == null) {
        	jq_Class klass = (jq_Class)Reflection.getJQType(c);
            for (Iterator i=ClassLibInterface.DEFAULT.getImplementationClassDescs(klass.getDesc()); i.hasNext(); ) {
                UTF.Utf8 u = (UTF.Utf8)i.next();
                if (TRACE) out.println("Checking mirror class "+u);
                String s = u.toString();
                jq.Assert(s.charAt(0) == 'L');
                try {
                    c = Class.forName(s.substring(1, s.length()-1).replace('/', '.'));
                    f2 = Reflection.getJDKField(c, fieldName);
                    if (f2 != null) break;
                } catch (ClassNotFoundException x) {
                    if (TRACE) out.println("Mirror class "+s+" doesn't exist");
                }
            }
        }
        jq.Assert(f2 != null, "host jdk does not contain static field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) != 0);
        try {
            return this.mapValue(f2.get(null));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public Object getInstanceFieldValue(Object base, jq_InstanceField f) {
        java.lang.Object result = this.mapInstanceField(base, f);
        if (result != NO_OBJECT) return this.mapValue(result);
        // get the value via real reflection.
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        return getInstanceFieldValue_reflection(base, c, fieldName);
    }
    public Object getInstanceFieldValue_reflection(Object base, Class c, String fieldName) {
        if (TRACE) out.println("Getting value of instance field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.Assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            return this.mapValue(f2.get(base));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public void putStaticFieldValue(jq_StaticField f, Object o) {
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        putStaticFieldValue(c, fieldName, o);
    }
    public void putStaticFieldValue(Class c, String fieldName, Object o) {
        if (TRACE) out.println("Setting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        if (f2 == null) {
        	jq_Class klass = (jq_Class)Reflection.getJQType(c);
            for (Iterator i=ClassLibInterface.DEFAULT.getImplementationClassDescs(klass.getDesc()); i.hasNext(); ) {
                UTF.Utf8 u = (UTF.Utf8)i.next();
                if (TRACE) out.println("Checking mirror class "+u);
                String s = u.toString();
                jq.Assert(s.charAt(0) == 'L');
                try {
                    c = Class.forName(s.substring(1, s.length()-1).replace('/', '.'));
                    f2 = Reflection.getJDKField(c, fieldName);
                    if (f2 != null) break;
                } catch (ClassNotFoundException x) {
                    if (TRACE) out.println("Mirror class "+s+" doesn't exist");
                }
            }
        }
        jq.Assert(f2 != null, "host jdk does not contain static field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) != 0);
        try {
            f2.set(null, o);
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
        }
    }
    
    public void putInstanceFieldValue(Object base, jq_InstanceField f, Object o) {
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        putInstanceFieldValue(base, c, fieldName, o);
    }
    public void putInstanceFieldValue(Object base, Class c, String fieldName, Object o) {
        if (TRACE) out.println("Setting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.Assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            f2.set(base, o);
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
        }
    }
}

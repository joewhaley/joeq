/*
 * ObjectTraverser.java
 *
 * Created on January 14, 2001, 11:38 AM
 *
 * @author  jwhaley
 * @version 
 */

package Bootstrap;

import ClassLib.ClassLibInterface;
import Clazz.*;
import Run_Time.*;
import Scheduler.*;
import jq;
import java.util.*;
import java.io.*;
import java.lang.reflect.*;

public class ObjectTraverser {

    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    public static final boolean SKIP_TRANSIENT_FIELDS = false;

    private Set/*jq_StaticField*/ nullStaticFields;
    private Set/*jq_InstanceField*/ nullInstanceFields;
    
    public ObjectTraverser(Set/*jq_StaticField*/ nullStaticFields, Set/*jq_InstanceField*/ nullInstanceFields) {
        this.nullStaticFields = nullStaticFields; this.nullInstanceFields = nullInstanceFields;
    }
    
    public Object getStaticFieldValue(jq_StaticField f) {
        if (nullStaticFields.contains(f)) {
            if (TRACE) out.println("Skipping null static field "+f);
            return null;
        }
        if (SKIP_TRANSIENT_FIELDS && f.isTransient()) {
            if (TRACE) out.println("Skipping transient static field "+f);
            return null;
        }
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        /*
        Object res = getMappedStaticFieldValue(c, fieldName);
        if (res != NO_OBJECT) {
            if (TRACE) out.println("Getting mapped value of static field "+c+"."+fieldName);
            return mapValue(res);
        }
         */
        if (TRACE) out.println("Getting value of static field "+c+"."+fieldName+" through reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        if (f2 == null) {
            for (Iterator i=ClassLibInterface.i.getImplementationClassDescs(f.getDeclaringClass().getDesc()); i.hasNext(); ) {
                UTF.Utf8 u = (UTF.Utf8)i.next();
                if (TRACE) out.println("Checking mirror class "+u);
                String s = u.toString();
                jq.assert(s.charAt(0) == 'L');
                try {
                    c = Class.forName(s.substring(1, s.length()-1).replace('/', '.'));
                    f2 = Reflection.getJDKField(c, fieldName);
                    if (f2 != null) break;
                } catch (ClassNotFoundException x) {
                    if (TRACE) out.println("Mirror class "+s+" doesn't exist");
                }
            }
        }
        jq.assert(f2 != null, "host jdk does not contain static field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.assert((f2.getModifiers() & Modifier.STATIC) != 0);
        try {
            return mapValue(f2.get(null));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
        
        /*
        Class c2 = c;
        while (c != null) {
            Field[] fields = c.getDeclaredFields();
            for (int i=0; i<fields.length; ++i) {
                Field f2 = fields[i];
                if (f2.getName().equals(fieldName)) {
                    f2.setAccessible(true);
                    jq.assert((f2.getModifiers() & Modifier.STATIC) != 0);
                    try {
                        return mapValue(f2.get(null));
                    } catch (IllegalAccessException x) {
                        jq.UNREACHABLE();
                        return null;
                    }
                }
            }
            c = c.getSuperclass();
        }
        jq.UNREACHABLE("host jdk does not contain static field "+c2.getName()+"."+fieldName);
        return null;
         */
    }
    
    public Object getInstanceFieldValue(Object o, jq_InstanceField f) {
        if (nullInstanceFields.contains(f)) {
            if (TRACE) out.println("Skipping null instance field "+f);
            return null;
        }
        if (SKIP_TRANSIENT_FIELDS && f.isTransient()) {
            if (TRACE) out.println("Skipping transient instance field "+f);
            return null;
        }
        jq.assert(o != null);
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        return getInstanceFieldValue(o, c, fieldName);
    }
    private Object getInstanceFieldValue(Object o, Class c, String fieldName) {
        Object res = getMappedInstanceFieldValue(o, c, fieldName);
        if (res != NO_OBJECT) {
            if (TRACE) out.println("Getting mapped value of instance field "+c+"."+fieldName);
            return mapValue(res);
        }
        if (TRACE) out.println("Getting value of instance field "+c+"."+fieldName+" through reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            return mapValue(f2.get(o));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
        /*
        Class c2 = c;
        while (c != null) {
            Field[] fields = c.getDeclaredFields();
            for (int i=0; i<fields.length; ++i) {
                Field f2 = fields[i];
                if (f2.getName().equals(fieldName)) {
                    f2.setAccessible(true);
                    jq.assert((f2.getModifiers() & Modifier.STATIC) == 0);
                    try {
                        return mapValue(f2.get(o));
                    } catch (IllegalAccessException x) {
                        jq.UNREACHABLE();
                        return null;
                    }
                }
            }
            c = c.getSuperclass();
        }
        jq.UNREACHABLE("host jdk does not contain instance field "+c2.getName()+"."+fieldName);
        return null;
         */
    }
    private static Object NO_OBJECT = new Object();
    private HashMap mapped_objects = new HashMap();
    public Object getMappedInstanceFieldValue(Object o, Class c, String fieldName) {
        if (c == Class.class) {
            if (fieldName.equals("jq_type"))
                return Reflection.getJQType((Class)o);
            if (fieldName.equals("signers"))
                return null;
                //return ((Class)o).getSigners();
            if (fieldName.equals("protection_domain"))
                return null;
                //return ((Class)o).getProtectionDomain();
        }
        if (c == jq_Type.class) {
            if (o == jq_Reference.jq_NullType.NULL_TYPE)
                return null;
            if (o == Compil3r.Quad.BytecodeToQuad.jq_ReturnAddressType.INSTANCE)
                return null;
            if (fieldName.equals("class_object"))
                return Reflection.getJDKType((jq_Type)o);
        }
        if (c == java.lang.reflect.Field.class) {
            if (fieldName.equals("jq_field"))
                return Reflection.getJQMember((java.lang.reflect.Field)o);
        }
        if (c == java.lang.reflect.Method.class) {
            if (fieldName.equals("jq_method"))
                return Reflection.getJQMember((java.lang.reflect.Method)o);
        }
        if (c == java.lang.reflect.Constructor.class) {
            if (fieldName.equals("jq_init"))
                return Reflection.getJQMember((java.lang.reflect.Constructor)o);
        }
        if (c == jq_Member.class) {
            if (fieldName.equals("member_object")) {
                // reflection returns different objects every time!
                // cache one and use it from now on.
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = Reflection.getJDKMember((jq_Member)o));
                return o2;
            }
        }
        if (c == Thread.class) {
            if (fieldName.equals("jq_thread")) {
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = new jq_Thread((Thread)o));
                return o2;
            }
        }
        if (c == ClassLoader.class) {
            if (o == PrimordialClassLoader.loader) {
                if (fieldName.equals("parent")) {
                    return null;
                }
                if (fieldName.equals("desc2type")) {
                    return getInstanceFieldValue(o, PrimordialClassLoader.class, "bs_desc2type");
                }
            }
            if (fieldName.equals("desc2type")) {
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                Vector classes = (Vector)getInstanceFieldValue(o, c, "classes");
                HashMap desc2type = new HashMap();
                Iterator i = classes.iterator();
                while (i.hasNext()) {
                    Class c2 = (Class)i.next();
                    jq_Type t = Reflection.getJQType(c2);
                    desc2type.put(t.getDesc(), t);
                }
                mapped_objects.put(o, desc2type);
                return desc2type;
            }
        }
        if (c.getName().equals("java.lang.ref.Finalizer")) {
            // don't return fields of finalizer objects.
            return null;
        }
        if (c == java.util.zip.ZipFile.class) {
            if (fieldName.equals("raf")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[0];
            }
            if (fieldName.equals("entries")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[1];
            }
            if (fieldName.equals("cenpos")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[2];
            }
            if (fieldName.equals("pos")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[3];
            }
        }
        if (c == java.util.zip.Inflater.class) {
	    if (fieldName.equals("mode")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("readAdler")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("neededBits")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("repLength")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("repDist")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("uncomprLen")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("isLastBlock")) {
		return new Boolean(false);
	    }
	    if (fieldName.equals("totalOut")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("totalIn")) {
		return new Integer(0);
	    }
	    if (fieldName.equals("nowrap")) {
		return new Boolean(false);
	    }
            if (fieldName.equals("adler")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[0];
            }
            if (fieldName.equals("input")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[1];
            }
            if (fieldName.equals("outputWindow")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[2];
            }
	    return null;
	}
        if (Throwable.class.isAssignableFrom(c)) {
            if (fieldName.equals("backtrace")) {
                // sun jvm crashes when using reflection on java.lang.Throwable.backtrace
                Object o2 = mapped_objects.get(o);
                if (o2 != null) return o2;
                mapped_objects.put(o, o2 = new int[0]);
                return o2;
            }
        }
        return NO_OBJECT;
    }
    public Object mapValue(Object o) {
        if (o == ClassLoader.getSystemClassLoader()) {
            return PrimordialClassLoader.loader;
        }
        if (o instanceof java.util.zip.ZipFile) {
            Object o2 = mapped_objects.get(o);
            if (o2 != null) return o;
            mapped_objects.put(o, o2 = new Object[4]);
            String name = ((java.util.zip.ZipFile)o).getName();
            try {
                // initialize the fields of the object
                ClassLibInterface.i.init_zipfile((java.util.zip.ZipFile)o, name);
            } catch (IOException x) {
                x.printStackTrace();
                jq.UNREACHABLE("cannot open zip file "+((java.util.zip.ZipFile)o).getName()+": "+x);
            }
            
            // we need to reopen the RandomAccessFile on VM startup
            jq.assert(((Object[])o2)[0] != null);
            Object[] args = { ((Object[])o2)[0], name, new Boolean(false) };
            jq_Method raf_open = ClassLibInterface._class.getOrCreateStaticMethod("open_static", "(Ljava/io/RandomAccessFile;Ljava/lang/String;Z)V");
            MethodInvocation mi = new MethodInvocation(raf_open, args);
            jq.on_vm_startup.add(mi);
            System.out.println("Added call to reopen zip file "+name+" on joeq startup: "+mi);
            
            return o;
        }
        if (o instanceof java.util.zip.Inflater) {
            Object o2 = mapped_objects.get(o);
            if (o2 != null) return o;
            mapped_objects.put(o, o2 = new Object[3]);
	    boolean nowrap = false; // how do we know?
	    // initialize the fields of the object
	    ClassLibInterface.i.init_inflater((java.util.zip.Inflater)o, nowrap);
	    return o;
	}
        return o;
    }

    public void putInstanceFieldValue(Object o, jq_InstanceField f, Object v) {
        if (SKIP_TRANSIENT_FIELDS && f.isTransient()) {
            if (TRACE) out.println("Skipping transient instance field "+f);
            return;
        }
        jq.assert(o != null);
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        putInstanceFieldValue(o, c, fieldName, v);
    }
    public void putInstanceFieldValue(Object o, Class c, String fieldName, Object v) {
        boolean mapped = putMappedInstanceFieldValue(o, c, fieldName, v);
        if (mapped) {
            if (TRACE) out.println("Setting mapped value of instance field "+c+"."+fieldName);
            return;
        }
        if (TRACE) out.println("Setting value of instance field "+c+"."+fieldName+" through reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            f2.set(o, v);
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
        }
        /*
        Class c2 = c;
        while (c != null) {
            Field[] fields = c.getDeclaredFields();
            for (int i=0; i<fields.length; ++i) {
                Field f2 = fields[i];
                if (f2.getName().equals(fieldName)) {
                    f2.setAccessible(true);
                    jq.assert((f2.getModifiers() & Modifier.STATIC) == 0);
                    try {
                        f2.set(o, v);
                    } catch (IllegalAccessException x) {
                        jq.UNREACHABLE();
                    }
                    return;
                }
            }
            c = c.getSuperclass();
        }
        jq.UNREACHABLE("host jdk does not contain instance field "+c2.getName()+"."+fieldName);
         */
    }
    public boolean putMappedInstanceFieldValue(Object o, Class c, String fieldName, Object v) {
        if (c == Class.class) {
            if (fieldName.equals("jq_type"))
                jq.UNREACHABLE();
        }
        if (c == jq_Type.class) {
            if (fieldName.equals("class_object"))
                jq.UNREACHABLE();
        }
        if (c == java.lang.reflect.Field.class) {
            if (fieldName.equals("jq_field"))
                jq.UNREACHABLE();
        }
        if (c == java.lang.reflect.Method.class) {
            if (fieldName.equals("jq_method"))
                jq.UNREACHABLE();
        }
        if (c == java.lang.reflect.Constructor.class) {
            if (fieldName.equals("jq_init"))
                jq.UNREACHABLE();
        }
        if (c == jq_Member.class) {
            if (fieldName.equals("member_object"))
                jq.UNREACHABLE();
        }
        if (c == Thread.class) {
            if (fieldName.equals("jq_thread")) {
                jq.assert(v instanceof jq_Thread);
                mapped_objects.put(o, v);
                return true;
            }
        }
        if (c == ClassLoader.class) {
            if (o == PrimordialClassLoader.loader) {
                if (fieldName.equals("parent"))
                    jq.UNREACHABLE();
                if (fieldName.equals("desc2type"))
                    jq.UNREACHABLE();
            }
            if (fieldName.equals("desc2type")) {
                jq.assert(v instanceof Map);
                mapped_objects.put(o, v);
                return true;
            }
        }
        if (c.getName().equals("java.lang.ref.Finalizer"))
            jq.UNREACHABLE();
        if (c == java.util.zip.ZipFile.class) {
            if (fieldName.equals("raf")) {
                jq.assert(v instanceof java.io.RandomAccessFile);
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[0] = v;
                return true;
            }
            if (fieldName.equals("entries")) {
                jq.assert(v instanceof Hashtable);
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[1] = v;
                return true;
            }
            if (fieldName.equals("cenpos")) {
                jq.assert(v instanceof Long);
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[2] = v;
                return true;
            }
            if (fieldName.equals("pos")) {
                jq.assert(v instanceof Long);
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[3] = v;
                return true;
            }
        }
        if (c == java.util.zip.Inflater.class) {
            if (fieldName.equals("adler")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[0] = v;
                return true;
            }
            if (fieldName.equals("input")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[1] = v;
                return true;
            }
            if (fieldName.equals("outputWindow")) {
                Object[] o2 = (Object[])mapped_objects.get(o);
                o2[2] = v;
                return true;
            }
	    return true;
	}
        if (Throwable.class.isAssignableFrom(c)) {
            if (fieldName.equals("backtrace"))
                jq.UNREACHABLE();
        }
        return false;
    }
}

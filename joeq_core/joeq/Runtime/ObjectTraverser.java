/*
 * ObjectTraverser.java
 *
 * Created on January 14, 2001, 11:38 AM
 *
 * @author  jwhaley
 * @version 
 */

package Bootstrap;

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
    
    public ObjectTraverser(Set/*jq_StaticField*/ nullStaticFields) {
        this.nullStaticFields = nullStaticFields;
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
    }
    
    public Object getInstanceFieldValue(Object o, jq_InstanceField f) {
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
    }
    private static Object NO_OBJECT = new Object();
        /*
    public Object getMappedStaticFieldValue(Class c, String fieldName) {
        jq_Class k = (jq_Class)Reflection.getJQType(c);
        if (fieldName.equals("_class"))
            return k;
        if (c == Class.class) {
            if (fieldName.equals("_jq_type"))
                return k.getOrCreateInstanceField("jq_type", "LClazz/jq_Type;");
        }
        if (c == ClassLoader.class) {
            if (fieldName.equals("_desc2type"))
                return k.getOrCreateInstanceField("desc2type", "Ljava/util/Map;");
        }
        if (c == System.class) {
            if (fieldName.equals("_in"))
                return k.getOrCreateStaticField("in", "Ljava/io/InputStream;");
            if (fieldName.equals("_out"))
                return k.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
            if (fieldName.equals("_err"))
                return k.getOrCreateStaticField("err", "Ljava/io/PrintStream;");
            if (fieldName.equals("_props"))
                return k.getOrCreateStaticField("props", "Ljava/util/Properties;");
            if (fieldName.equals("_initializeSystemClass"))
                return k.getOrCreateStaticMethod("initializeSystemClass", "()V");
        }
        if (c == java.util.zip.ZipFile.class) {
            return getStaticFieldValue(org.jos.java.util.zip.ZipFile.class, fieldName);
        }
        if (c.getName().equals("java.io.FileSystem")) {
            if (fieldName.equals("default_fs")) {
                Object o2 = mapped_objects.get(c);
                if (o2 != null) return o2;
                try {
                    Constructor init = Class.forName("java.io.Win32FileSystem").getConstructor(null);
                    init.setAccessible(true);
                    o2 = init.newInstance(null);
                    mapped_objects.put(c, o2);
                } catch (ClassNotFoundException x) {
                    jq.UNREACHABLE();
                } catch (NoSuchMethodException x) {
                    jq.UNREACHABLE();
                } catch (InvocationTargetException x) {
                    jq.UNREACHABLE();
                } catch (IllegalAccessException x) {
                    jq.UNREACHABLE();
                } catch (InstantiationException x) {
                    jq.UNREACHABLE();
                }
                return o2;
            }
        }
        if (fieldName.startsWith("class$L")) {
            // jikes-compiled class file, static class reference.
            fieldName = "class$"+fieldName.substring(7);
            //return getStaticFieldValue(c, fieldName);
            return c;
        }
        if (c == java.io.FileDescriptor.class) {
            if (fieldName.equals("_fd"))
                return k.getOrCreateInstanceField("fd", "I");
        }
        return NO_OBJECT;
    }
         */
    private HashMap mapped_objects = new HashMap();
    public Object getMappedInstanceFieldValue(Object o, Class c, String fieldName) {
        if (c == Class.class) {
            if (fieldName.equals("jq_type"))
                return Reflection.getJQType((Class)o);
        }
        if (c == jq_Type.class) {
            if (fieldName.equals("class_object"))
                return Reflection.getJDKType((jq_Type)o);
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
                /*
                Object[] o2 = (Object[])mapped_objects.get(o);
                if (o2 != null) {
                    if (o2[0] != null) return o2[0];
                } else {
                    mapped_objects.put(o, o2 = new Object[2]);
                }
                String name = ((java.util.zip.ZipFile)o).getName();
                return o2[0] = new java.io.RandomAccessFile(name);
                 */
                Object[] o2 = (Object[])mapped_objects.get(o);
                return o2[0];
            }
            if (fieldName.equals("entries")) {
                /*
                Object[] o2 = (Object[])mapped_objects.get(o);
                if (o2 != null) {
                    if (o2[1] != null) return o2[1];
                } else {
                    mapped_objects.put(o, o2 = new Object[2]);
                }
                Enumeration e = ((java.util.zip.ZipFile)o).entries();
                Hashtable entries = new Hashtable();
                while (e.hasNext()) {
                    java.util.zip.ZipEntry ze = (java.util.zip.ZipEntry)e.next();
                    entries.put(ze.getName(), ze);
                }
                return o2[1] = entries;
                 */
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
                ClassLib.sun13.java.util.zip.ZipFile.__init__((java.util.zip.ZipFile)o, name);
            } catch (IOException x) {
                jq.UNREACHABLE("cannot open zip file "+o+": "+x);
            }
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
        if (Throwable.class.isAssignableFrom(c)) {
            if (fieldName.equals("backtrace"))
                jq.UNREACHABLE();
        }
        return false;
    }
}

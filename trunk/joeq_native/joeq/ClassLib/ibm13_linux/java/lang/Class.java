/*
 * Class.java
 *
 * Created on January 29, 2001, 11:40 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang;

import Clazz.jq_Type;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_InstanceField;
import Clazz.jq_Initializer;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Primitive;
import Clazz.jq_StaticField;
import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;
import jq;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;

public class Class {
    
    // additional instance fields.
    public final jq_Type jq_type = null;
    private java.lang.Object[] signers;
    private java.security.ProtectionDomain protection_domain;
    
    // native method implementations.
    private static void registerNatives(jq_Class clazz) { }
    private static java.lang.Class forName0(jq_Class clazz, java.lang.String name, boolean initialize,
				            java.lang.ClassLoader loader)
        throws ClassNotFoundException
    {
        java.lang.Class k = loader.loadClass(name);
        if (initialize) {
            jq_Type t = (jq_Type)Reflection.getfield_A(k, _jq_type);
            jq.assert(t.isLoaded());
            t.verify();
            t.prepare();
            t.sf_initialize();
            t.cls_initialize();
        }
        return k;
    }
    private static java.lang.Class forName1(jq_Class clazz, java.lang.String name)
        throws ClassNotFoundException
    {
	java.lang.ClassLoader loader = clazz.getClassLoader();
	return forName0(clazz, name, true, loader);
    }

    private static java.lang.Object newInstance0(java.lang.Class dis)
        throws InstantiationException, IllegalAccessException
    {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (!jq_type.isClassType())
            throw new java.lang.InstantiationException(jq_type.getDesc()+" is not a class type");
        jq_Class jq_class = (jq_Class)jq_type;
        jq_class.load();
        if (jq_class.isAbstract())
            throw new java.lang.InstantiationException("cannot instantiate abstract "+dis);
        jq_Initializer i = jq_class.getInitializer(new jq_NameAndDesc(Utf8.get("<init>"), Utf8.get("()V")));
        if (i == null)
            throw new InstantiationException("no empty arg initializer in "+dis);
        i.checkCallerAccess(4);
        jq_class.verify(); jq_class.prepare(); jq_class.sf_initialize(); jq_class.cls_initialize(); 
        java.lang.Object o = jq_class.newInstance();
        try {
            Reflection.invokeinstance_V(i, o);
        } catch (Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            throw new ExceptionInInitializerError(x);
        }
        return o;
    }

    private static java.lang.Object newInstance2(java.lang.Class dis, java.lang.Class ccls)
        throws InstantiationException, IllegalAccessException
    {
	// what is special about ccls?
	return newInstance0(dis);
    }

    public static boolean isInstance(java.lang.Class dis, java.lang.Object obj) {
        jq_Type t = Unsafe.getTypeOf(obj);
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        jq_type.load(); jq_type.verify(); jq_type.prepare();
        return TypeCheck.isAssignable(t, jq_type);
    }
    
    public static boolean isAssignableFrom(java.lang.Class dis, java.lang.Class cls) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        jq_type.load(); jq_type.verify(); jq_type.prepare();
        jq_Type cls_jq_type = (jq_Type)Reflection.getfield_A(cls, _jq_type);
        cls_jq_type.load(); cls_jq_type.verify(); cls_jq_type.prepare();
        return TypeCheck.isAssignable(cls_jq_type, jq_type);
    }
    
    public static boolean isInterface(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        jq_type.load();
        return jq_type.isClassType() && ((jq_Class)jq_type).isInterface();
    }
    
    public static boolean isArray(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        return jq_type.isArrayType();
    }
    
    public static boolean isPrimitive(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        return jq_type.isPrimitiveType();
    }
    
    public static java.lang.String getName(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (jq_type.isArrayType()) return jq_type.getDesc().toString().replace('/','.');
        else return jq_type.getName().toString();
    }
    
    private static java.lang.ClassLoader getClassLoader0(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        return (java.lang.ClassLoader)jq_type.getClassLoader();
    }
    
    public static java.lang.Class getSuperclass(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (!jq_type.isClassType()) return null;
        jq_Class k = (jq_Class)jq_type;
        k.load(); k.verify(); k.prepare();
        if (k.getSuperclass() == null) return null;
        return Reflection.getJDKType(k.getSuperclass());
    }
    
    public static java.lang.Class[] getInterfaces(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (jq_type.isPrimitiveType()) return new java.lang.Class[0];
        jq_Class[] ins;
        jq_type.load();
        if (jq_type.isArrayType()) ins = jq_Array.array_interfaces;
        else ins = ((jq_Class)jq_type).getDeclaredInterfaces();
        java.lang.Class[] c = new java.lang.Class[ins.length];
        for (int i=0; i<ins.length; ++i) {
            c[i] = Reflection.getJDKType(ins[i]);
        }
        return c;
    }
    
    public static java.lang.Class getComponentType(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (!jq_type.isArrayType()) return null;
        return Reflection.getJDKType(((jq_Array)jq_type).getElementType());
    }
    
    public static int getModifiers(java.lang.Class dis) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        jq_type.load();
        if (jq_type.isPrimitiveType()) return jq_ClassFileConstants.ACC_PUBLIC | jq_ClassFileConstants.ACC_FINAL;
        if (jq_type.isArrayType()) return Reflection.getJDKType(((jq_Array)jq_type).getElementType()).getModifiers() | jq_ClassFileConstants.ACC_FINAL & ~jq_ClassFileConstants.ACC_INTERFACE;
        return (int)((jq_Class)jq_type).getAccessFlags();
    }

    public static java.lang.Object[] getSigners(java.lang.Class dis) {
        // TODO: correct handling of signers
        return (java.lang.Object[])Reflection.getfield_A(dis, _signers);
    }
    static void setSigners(java.lang.Class dis, java.lang.Object[] signers) {
        // TODO: correct handling of signers
        Reflection.putfield_A(dis, _signers, signers);
    }
    
    //public static java.lang.Class getDeclaringClass(java.lang.Class dis) { }
    
    private static java.security.ProtectionDomain getProtectionDomain0(java.lang.Class dis) {
        // TODO: correct handling of ProtectionDomain
        return (java.security.ProtectionDomain)Reflection.getfield_A(dis, _protection_domain);
    }
    static void setProtectionDomain0(java.lang.Class dis, java.security.ProtectionDomain pd) {
        // TODO: correct handling of ProtectionDomain
        Reflection.putfield_A(dis, _protection_domain, pd);
    }

    static java.lang.Class getPrimitiveClass(jq_Class clazz, java.lang.String name) {
        if (name.equals("int")) return Reflection.getJDKType(jq_Primitive.INT);
        if (name.equals("float")) return Reflection.getJDKType(jq_Primitive.FLOAT);
        if (name.equals("long")) return Reflection.getJDKType(jq_Primitive.LONG);
        if (name.equals("double")) return Reflection.getJDKType(jq_Primitive.DOUBLE);
        if (name.equals("boolean")) return Reflection.getJDKType(jq_Primitive.BOOLEAN);
        if (name.equals("byte")) return Reflection.getJDKType(jq_Primitive.BYTE);
        if (name.equals("char")) return Reflection.getJDKType(jq_Primitive.CHAR);
        if (name.equals("short")) return Reflection.getJDKType(jq_Primitive.SHORT);
        if (name.equals("void")) return Reflection.getJDKType(jq_Primitive.VOID);
        throw new InternalError("no such primitive type: "+name);
    }
    
    private static java.lang.reflect.Field[] getFields0(java.lang.Class dis, int which) {
        jq_Type jq_type = (jq_Type)Reflection.getfield_A(dis, _jq_type);
        if (!jq_type.isClassType()) return new java.lang.reflect.Field[0];
        jq_Class c = (jq_Class)jq_type;
        c.load();
        if (which == java.lang.reflect.Member.DECLARED) {
            jq_StaticField[] sfs = c.getDeclaredStaticFields();
            jq_InstanceField[] ifs = c.getDeclaredInstanceFields();
            int size = sfs.length + ifs.length;
            java.lang.reflect.Field[] fs = new java.lang.reflect.Field[size];
            for (int j=0; j<sfs.length; ++j) {
                fs[j] = (java.lang.reflect.Field)sfs[j].getJavaLangReflectMemberObject();
            }
            for (int j=0; j<ifs.length; ++j) {
                fs[sfs.length+j] = (java.lang.reflect.Field)ifs[j].getJavaLangReflectMemberObject();
            }
            return fs;
        } else {
            jq.assert(which == java.lang.reflect.Member.PUBLIC);
            int size = 0;
            jq_StaticField[] sfs = c.getStaticFields();
            jq_InstanceField[] ifs = c.getInstanceFields();
            for (int j=0; j<sfs.length; ++j) {
                if (sfs[j].isPublic()) ++size;
            }
            for (int j=0; j<ifs.length; ++j) {
                if (ifs[j].isPublic()) ++size;
            }
            java.lang.reflect.Field[] fs = new java.lang.reflect.Field[size];
            int current = -1;
            for (int j=0; j<sfs.length; ++j) {
                if (sfs[j].isPublic())
                    fs[++current] = (java.lang.reflect.Field)sfs[j].getJavaLangReflectMemberObject();
            }
            for (int j=0; j<ifs.length; ++j) {
                if (ifs[j].isPublic())
                    fs[++current] = (java.lang.reflect.Field)ifs[j].getJavaLangReflectMemberObject();
            }
            jq.assert(current+1 == fs.length);
            return fs;
        }
    }
    //private static java.lang.reflect.Method[] getMethods0(java.lang.Class dis, int which) {}
    //private static java.lang.reflect.Constructor[] getConstructors0(java.lang.Class dis, int which) {}
    private static java.lang.reflect.Field getField0(java.lang.Class dis, java.lang.String name, int which) {
        return null;
    }
    //private static java.lang.reflect.Method getMethod0(java.lang.Class dis, java.lang.String name, java.lang.Class[] parameterTypes, int which) {}
    //private static java.lang.reflect.Constructor getConstructor0(java.lang.Class dis, java.lang.Class[] parameterTypes, int which) {
    //private static java.lang.reflect.Class[] getDeclaredClasses0(java.lang.Class dis) {}
    
    // additional methods.
    // ONLY TO BE CALLED BY jq_Class CONSTRUCTOR!!!
    public static java.lang.Class createNewClass(jq_Class clazz, jq_Type jq_type) {
        java.lang.Class o = (java.lang.Class)_class.newInstance();
        Reflection.putfield_A(o, _jq_type, jq_type);
        return o;
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Class;");
    public static final jq_InstanceField _jq_type = _class.getOrCreateInstanceField("jq_type", "LClazz/jq_Type;");
    public static final jq_InstanceField _signers = _class.getOrCreateInstanceField("signers", "[Ljava/lang/Object;");
    public static final jq_InstanceField _protection_domain = _class.getOrCreateInstanceField("protection_domain", "[Ljava/security/ProtectionDomain;");
}

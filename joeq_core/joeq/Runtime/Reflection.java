/*
 * Reflection.java
 *
 * Created on January 12, 2001, 12:48 PM
 *
 */

package Run_Time;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_ClassInitializer;
import Clazz.jq_Field;
import Clazz.jq_Initializer;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Member;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Main.jq;
import Memory.Address;
import Memory.HeapAddress;
import UTF.Utf8;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Reflection {

    public static ObjectTraverser obj_trav;
    
    public static final jq_Reference getTypeOf(Object o) {
        if (!jq.Bootstrapping) return jq_Reference.getTypeOf(o);
        return (jq_Reference) getJQType(o.getClass());
    }
    
    // Map between our jq_Type objects and JDK Class objects
    public static final jq_Type getJQType(Class c) {
        if (!jq.Bootstrapping) return ClassLibInterface.DEFAULT.getJQType(c);
        if (c.isPrimitive()) {
            if (c == Byte.TYPE) return jq_Primitive.BYTE;
            if (c == Character.TYPE) return jq_Primitive.CHAR;
            if (c == Double.TYPE) return jq_Primitive.DOUBLE;
            if (c == Float.TYPE) return jq_Primitive.FLOAT;
            if (c == Integer.TYPE) return jq_Primitive.INT;
            if (c == Long.TYPE) return jq_Primitive.LONG;
            if (c == Short.TYPE) return jq_Primitive.SHORT;
            if (c == Boolean.TYPE) return jq_Primitive.BOOLEAN;
            if (c == Void.TYPE) return jq_Primitive.VOID;
            jq.UNREACHABLE(c.toString());
            return null;
        }
        String className = c.getName().replace('.','/');
        if (!className.startsWith("[")) className = "L"+className+";";
        className = ClassLib.ClassLibInterface.convertClassLibDesc(className);
        return PrimordialClassLoader.loader.getOrCreateBSType(className);
    }
    public static final Class getJDKType(jq_Type c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        if (c.getJavaLangClassObject() != null)
            return c.getJavaLangClassObject();
        if (c.isPrimitiveType()) 
            return getJDKType((jq_Primitive)c);
        else
            return getJDKType((jq_Reference)c);
    }
    public static final Class getJDKType(jq_Primitive c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        if (c.getJavaLangClassObject() != null)
            return c.getJavaLangClassObject();
        // cannot compare to jq_Primitive types here, as they may not
        // have been initialized yet.  so we compare descriptors instead.
        if (c.getDesc() == Utf8.BYTE_DESC) return Byte.TYPE;
        if (c.getDesc() == Utf8.CHAR_DESC) return Character.TYPE;
        if (c.getDesc() == Utf8.DOUBLE_DESC) return Double.TYPE;
        if (c.getDesc() == Utf8.FLOAT_DESC) return Float.TYPE;
        if (c.getDesc() == Utf8.INT_DESC) return Integer.TYPE;
        if (c.getDesc() == Utf8.LONG_DESC) return Long.TYPE;
        if (c.getDesc() == Utf8.SHORT_DESC) return Short.TYPE;
        if (c.getDesc() == Utf8.BOOLEAN_DESC) return Boolean.TYPE;
        if (c.getDesc() == Utf8.VOID_DESC) return Void.TYPE;
        jq.UNREACHABLE(c.getName());
        return null;
    }
    public static Class getJDKType(jq_Reference c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        if (c.getJavaLangClassObject() != null)
            return c.getJavaLangClassObject();
        try {
            return Class.forName(c.getJDKName(), false, Reflection.class.getClassLoader());
        } catch (ClassNotFoundException x) {
            if (!c.getJDKName().startsWith("ClassLib"))
                SystemInterface.debugmsg("Note: "+c.getJDKName()+" was not found in host jdk");
            return null;
        }
    }
    
    // Map between our jq_Member objects and JDK Member objects
    public static final jq_Field getJQMember(Field f) {
        if (!jq.Bootstrapping) return ClassLibInterface.DEFAULT.getJQField(f);
        jq_Class c = (jq_Class)getJQType(f.getDeclaringClass());
        jq_NameAndDesc nd = new jq_NameAndDesc(Utf8.get(f.getName()), getJQType(f.getType()).getDesc());
        nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(c, nd);
        jq_Field m = (jq_Field)c.getDeclaredMember(nd);
        if (m == null) {
            SystemInterface.debugmsg("Reference to jdk field "+f.toString()+" does not exist, creating "+c+"."+nd);
            if (Modifier.isStatic(f.getModifiers()))
                m = c.getOrCreateStaticField(nd);
            else
                m = c.getOrCreateInstanceField(nd);
        }
        return m;
    }
    public static final jq_Method getJQMember(Method f) {
        if (!jq.Bootstrapping) return ClassLibInterface.DEFAULT.getJQMethod(f);
        jq_Class c = (jq_Class)getJQType(f.getDeclaringClass());
        StringBuffer desc = new StringBuffer();
        desc.append('(');
        Class[] param_types = f.getParameterTypes();
        for (int i=0; i<param_types.length; ++i) {
            desc.append(getJQType(param_types[i]).getDesc().toString());
        }
        desc.append(')');
        desc.append(getJQType(f.getReturnType()).getDesc().toString());
        jq_NameAndDesc nd = new jq_NameAndDesc(Utf8.get(f.getName()), Utf8.get(desc.toString()));
        nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(c, nd);
        jq_Method m = (jq_Method)c.getDeclaredMember(nd);
        if (m == null) {
            SystemInterface.debugmsg("Reference to jdk method "+f.toString()+" does not exist, creating "+c+"."+nd);
            if (Modifier.isStatic(f.getModifiers()))
                m = c.getOrCreateStaticMethod(nd);
            else
                m = c.getOrCreateInstanceMethod(nd);
        }
        return m;
    }
    public static final jq_Initializer getJQMember(Constructor f) {
        if (!jq.Bootstrapping) return ClassLibInterface.DEFAULT.getJQInitializer(f);
        jq_Class c = (jq_Class)getJQType(f.getDeclaringClass());
        StringBuffer desc = new StringBuffer();
        desc.append('(');
        Class[] param_types = f.getParameterTypes();
        for (int i=0; i<param_types.length; ++i) {
            desc.append(getJQType(param_types[i]).getDesc().toString());
        }
        desc.append(")V");
        jq_NameAndDesc nd = new jq_NameAndDesc(Utf8.get("<init>"), Utf8.get(desc.toString()));
        nd = ClassLib.ClassLibInterface.convertClassLibNameAndDesc(c, nd);
        jq_Initializer m = (jq_Initializer)c.getDeclaredMember(nd);
        if (m == null) {
            SystemInterface.debugmsg("Reference to jdk constructor "+f.toString()+" does not exist, creating "+c+"."+nd);
            m = (jq_Initializer)c.getOrCreateInstanceMethod(nd);
        }
        return m;
    }
    public static boolean USE_DECLARED_FIELDS_CACHE = true;
    private static java.util.HashMap declaredFieldsCache;
    public static final Field getJDKField(Class c, String name) {
        Field[] fields = null;
        if (USE_DECLARED_FIELDS_CACHE) {
            if (declaredFieldsCache == null) declaredFieldsCache = new java.util.HashMap();
            else fields = (Field[])declaredFieldsCache.get(c);
            if (fields == null)
                declaredFieldsCache.put(c, fields = c.getDeclaredFields());
        } else {
            fields = c.getDeclaredFields();
        }
        for (int i=0; i<fields.length; ++i) {
            Field f2 = fields[i];
            if (f2.getName().equals(name)) {
                //f2.setAccessible(true);
                return f2;
            }
        }
        //jq.UNREACHABLE(c+"."+name);
        return null;
    }
    public static final Method getJDKMethod(Class c, String name, Class[] args) {
        Method[] methods = c.getDeclaredMethods();
uphere:
        for (int i=0; i<methods.length; ++i) {
            Method f2 = methods[i];
            if (f2.getName().equals(name)) {
                Class[] args2 = f2.getParameterTypes();
                if (args.length != args2.length) continue uphere;
                for (int j=0; j<args.length; ++j) {
                    if (!args[j].equals(args2[j])) continue uphere;
                }
                //f2.setAccessible(true);
                return f2;
            }
        }
        //jq.UNREACHABLE(c+"."+name+" "+args);
        return null;
    }
    public static final Constructor getJDKConstructor(Class c, Class[] args) {
        Constructor[] consts = c.getDeclaredConstructors();
uphere:
        for (int i=0; i<consts.length; ++i) {
            Constructor f2 = consts[i];
            Class[] args2 = f2.getParameterTypes();
            if (args.length != args2.length) continue uphere;
            for (int j=0; j<args.length; ++j) {
                if (!args[j].equals(args2[j])) continue uphere;
            }
            //f2.setAccessible(true);
            return f2;
        }
        //jq.UNREACHABLE(c+".<init> "+args);
        return null;
    }
    public static final Member getJDKMember(jq_Member m) {
        if (!jq.Bootstrapping) return m.getJavaLangReflectMemberObject();
        if (m.getJavaLangReflectMemberObject() != null)
            return m.getJavaLangReflectMemberObject();
        Class c = getJDKType(m.getDeclaringClass());
        if (m instanceof jq_Field) {
            Member ret = getJDKField(c, m.getName().toString());
            if (ret == null) {
                // TODO: a synthetic field, so there is no java.lang.reflect.Field object yet.
            }
            return ret;
        } else if (m instanceof jq_Initializer) {
            jq_Initializer m2 = (jq_Initializer)m;
            jq_Type[] param_types = m2.getParamTypes();
            int num_of_args = param_types.length-1; // -1 for this ptr
            Class[] args = new Class[num_of_args];
            for (int i=0; i<num_of_args; ++i) {
                args[i] = getJDKType(param_types[i+1]);
            }
            Member ret = getJDKConstructor(c, args);
            if (ret == null) {
                // TODO: a synthetic field, so there is no java.lang.reflect.Field object yet.
            }
            return ret;
        } else if (m instanceof jq_ClassInitializer) {
            return null; // <clinit> methods have no Method object
        } else {
            jq.Assert(m instanceof jq_Method);
            jq_Method m2 = (jq_Method)m;
            int offset = m2.isStatic()?0:1;
            jq_Type[] param_types = m2.getParamTypes();
            int num_of_args = param_types.length-offset;
            Class[] args = new Class[num_of_args];
            for (int i=0; i<num_of_args; ++i) {
                args[i] = getJDKType(param_types[i+offset]);
            }
            Member ret = getJDKMethod(c, m.getName().toString(), args);
            if (ret == null) {
                // TODO: a synthetic field, so there is no java.lang.reflect.Field object yet.
            }
            return ret;
        }
    }
    
    public static Class[] getArgTypesFromDesc(Utf8 desc) {
        Utf8.MethodDescriptorIterator i = desc.getParamDescriptors();
        // count them up
        int num = 0;
        while (i.hasNext()) { i.nextUtf8(); ++num; }
        // get them for real
        Class[] param_types = new Class[num];
        i = desc.getParamDescriptors();
        for (int j=0; j<num; ++j) {
            Utf8 pd = (Utf8)i.nextUtf8();
            jq_Type t = PrimordialClassLoader.loader.getOrCreateBSType(pd);
            param_types[j] = getJDKType(t);
        }
        //Utf8 rd = i.getReturnDescriptor();
        return param_types;
    }
    
    // reflective invocations.
    public static void invokestatic_V(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
        }
    }
    public static int invokestatic_I(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            return (int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return 0;
        }
    }
    public static Object invokestatic_A(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            return ((HeapAddress)Unsafe.invokeA(m.getDefaultCompiledVersion().getEntrypoint())).asObject();
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static long invokestatic_J(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return 0L;
        }
    }
    public static void invokestatic_V(jq_StaticMethod m, Object arg1) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static Object invokeinstance_A(jq_InstanceMethod m, Object dis) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            return ((HeapAddress)Unsafe.invokeA(m.getDefaultCompiledVersion().getEntrypoint())).asObject();
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static Object invokeinstance_A(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            return ((HeapAddress)Unsafe.invokeA(m.getDefaultCompiledVersion().getEntrypoint())).asObject();
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static boolean invokeinstance_Z(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            return ((int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint())) != 0;
        } else {
            jq.UNREACHABLE();
            return false;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, Object arg2) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.pushArgA(HeapAddress.addressOf(arg2));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, Object arg2, Object arg3) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.pushArgA(HeapAddress.addressOf(arg2));
            Unsafe.pushArgA(HeapAddress.addressOf(arg3));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, Object arg2, Object arg3, long arg4) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.pushArgA(HeapAddress.addressOf(arg2));
            Unsafe.pushArgA(HeapAddress.addressOf(arg3));
            Unsafe.pushArg((int)(arg4 >> 32)); // hi
            Unsafe.pushArg((int)arg4);         // lo
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, int arg2, long arg3, int arg4) throws Throwable {
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.Assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            Unsafe.pushArgA(HeapAddress.addressOf(arg1));
            Unsafe.pushArg(arg2);
            Unsafe.pushArg((int)(arg3 >> 32)); // hi
            Unsafe.pushArg((int)arg3);         // lo
            Unsafe.pushArg(arg4);
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static long invoke(jq_Method m, Object dis, Object[] args)
        throws IllegalArgumentException, InvocationTargetException
    {
        jq_Type[] params = m.getParamTypes();
        int offset;
        if (dis != null) {
            jq.Assert(!m.isStatic());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            offset = 1;
        } else {
            offset = 0;
        }
        if (args != null) {
            jq.Assert(params.length == args.length+offset);
            for (int i=0; i<args.length; ++i) {
                jq_Type c = params[i+offset];
                if (c.isAddressType()) {
                    jq.TODO();
                } else if (c.isReferenceType()) {
                    if (args[i] != null && !TypeCheck.isAssignable(jq_Reference.getTypeOf(args[i]), c))
                        throw new IllegalArgumentException(args[i].getClass()+" is not assignable to "+c);
                    Unsafe.pushArgA(HeapAddress.addressOf(args[i]));
                } else {
                    if (c == jq_Primitive.BYTE) {
                        int v = (int)unwrapToByte(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.CHAR) {
                        int v = (int)unwrapToChar(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.DOUBLE) {
                        long v = Double.doubleToRawLongBits(unwrapToDouble(args[i]));
                        Unsafe.pushArg((int)(v >> 32)); // hi
                        Unsafe.pushArg((int)v);         // lo
                    } else if (c == jq_Primitive.FLOAT) {
                        int v = Float.floatToRawIntBits(unwrapToFloat(args[i]));
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.INT) {
                        int v = unwrapToInt(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.LONG) {
                        long v = unwrapToLong(args[i]);
                        Unsafe.pushArg((int)(v >> 32)); // hi
                        Unsafe.pushArg((int)v);         // lo
                    } else if (c == jq_Primitive.SHORT) {
                        int v = (int)unwrapToShort(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.BOOLEAN) {
                        int v = unwrapToBoolean(args[i])?1:0;
                        Unsafe.pushArg(v);
                    } else jq.UNREACHABLE(c.toString());
                }
            }
        } else {
            jq.Assert(params.length == offset);
        }
        try {
            return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } catch (Throwable t) {
            throw new InvocationTargetException(t);
        }
    }
    public static Address invokeA(jq_Method m, Object dis, Object[] args)
        throws IllegalArgumentException, InvocationTargetException
    {
        jq_Type[] params = m.getParamTypes();
        int offset;
        if (dis != null) {
            jq.Assert(!m.isStatic());
            Unsafe.pushArgA(HeapAddress.addressOf(dis));
            offset = 1;
        } else {
            offset = 0;
        }
        if (args != null) {
            jq.Assert(params.length == args.length+offset);
            for (int i=0; i<args.length; ++i) {
                jq_Type c = params[i+offset];
                if (c.isAddressType()) {
                    jq.TODO();
                } else if (c.isReferenceType()) {
                    if (args[i] != null && !TypeCheck.isAssignable(jq_Reference.getTypeOf(args[i]), c))
                        throw new IllegalArgumentException(args[i].getClass()+" is not assignable to "+c);
                    Unsafe.pushArgA(HeapAddress.addressOf(args[i]));
                } else {
                    if (c == jq_Primitive.BYTE) {
                        int v = (int)unwrapToByte(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.CHAR) {
                        int v = (int)unwrapToChar(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.DOUBLE) {
                        long v = Double.doubleToRawLongBits(unwrapToDouble(args[i]));
                        Unsafe.pushArg((int)(v >> 32)); // hi
                        Unsafe.pushArg((int)v);         // lo
                    } else if (c == jq_Primitive.FLOAT) {
                        int v = Float.floatToRawIntBits(unwrapToFloat(args[i]));
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.INT) {
                        int v = unwrapToInt(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.LONG) {
                        long v = unwrapToLong(args[i]);
                        Unsafe.pushArg((int)(v >> 32)); // hi
                        Unsafe.pushArg((int)v);         // lo
                    } else if (c == jq_Primitive.SHORT) {
                        int v = (int)unwrapToShort(args[i]);
                        Unsafe.pushArg(v);
                    } else if (c == jq_Primitive.BOOLEAN) {
                        int v = unwrapToBoolean(args[i])?1:0;
                        Unsafe.pushArg(v);
                    } else jq.UNREACHABLE(c.toString());
                }
            }
        } else {
            jq.Assert(params.length == offset);
        }
        try {
            return Unsafe.invokeA(m.getDefaultCompiledVersion().getEntrypoint());
        } catch (Throwable t) {
            throw new InvocationTargetException(t);
        }
    }
    
    // unwrap functions
    public static boolean unwrapToBoolean(Object value) throws IllegalArgumentException {
        if (value instanceof Boolean) return ((Boolean)value).booleanValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to boolean");
    }
    public static byte unwrapToByte(Object value) throws IllegalArgumentException {
        if (value instanceof Byte) return ((Byte)value).byteValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to byte");
    }
    public static char unwrapToChar(Object value) throws IllegalArgumentException {
        if (value instanceof Character) return ((Character)value).charValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to char");
    }
    public static short unwrapToShort(Object value) throws IllegalArgumentException {
        if (value instanceof Short) return ((Short)value).shortValue();
        else if (value instanceof Byte) return ((Byte)value).shortValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to short");
    }
    public static int unwrapToInt(Object value) throws IllegalArgumentException {
        if (value instanceof Integer) return ((Integer)value).intValue();
        else if (value instanceof Byte) return ((Byte)value).intValue();
        else if (value instanceof Character) return (int)((Character)value).charValue();
        else if (value instanceof Short) return ((Short)value).intValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to int");
    }
    public static long unwrapToLong(Object value) throws IllegalArgumentException {
        if (value instanceof Long) return ((Long)value).longValue();
        else if (value instanceof Integer) return ((Integer)value).longValue();
        else if (value instanceof Byte) return ((Byte)value).longValue();
        else if (value instanceof Character) return (long)((Character)value).charValue();
        else if (value instanceof Short) return ((Short)value).longValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to long");
    }
    public static float unwrapToFloat(Object value) throws IllegalArgumentException {
        if (value instanceof Float) return ((Float)value).floatValue();
        else if (value instanceof Integer) return ((Integer)value).floatValue();
        else if (value instanceof Long) return ((Long)value).floatValue();
        else if (value instanceof Byte) return ((Byte)value).floatValue();
        else if (value instanceof Character) return (float)((Character)value).charValue();
        else if (value instanceof Short) return ((Short)value).floatValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to float");
    }
    public static double unwrapToDouble(Object value) throws IllegalArgumentException {
        if (value instanceof Double) return ((Double)value).doubleValue();
        else if (value instanceof Float) return ((Float)value).doubleValue();
        else if (value instanceof Integer) return ((Integer)value).doubleValue();
        else if (value instanceof Long) return ((Long)value).doubleValue();
        else if (value instanceof Byte) return ((Byte)value).doubleValue();
        else if (value instanceof Character) return (double)((Character)value).charValue();
        else if (value instanceof Short) return ((Short)value).doubleValue();
        else throw new IllegalArgumentException((value==null?null:value.getClass())+" cannot be converted to double");
    }

    public static int getfield_I(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0;
            return ((Integer)q).intValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek4();
    }
    public static long getfield_L(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0L;
            return ((Long)q).longValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek8();
    }
    public static float getfield_F(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0f;
            return ((Float)q).floatValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return Float.intBitsToFloat(getfield_I(o, f));
    }
    public static double getfield_D(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0.;
            return ((Double)q).doubleValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return Double.longBitsToDouble(getfield_L(o, f));
    }
    public static Object getfield_A(Object o, jq_InstanceField f) {
        jq.Assert(f.getType().isReferenceType() && !f.getType().isAddressType());
        if (jq.Bootstrapping) return obj_trav.getInstanceFieldValue(o, f);
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return ((HeapAddress) HeapAddress.addressOf(o).offset(f.getOffset()).peek()).asObject();
    }
    public static Address getfield_P(Object o, jq_InstanceField f) {
        jq.Assert(f.getType().isAddressType());
        if (jq.Bootstrapping) return (Address)obj_trav.getInstanceFieldValue(o, f);
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek();
    }
    public static byte getfield_B(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.BYTE);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0;
            return ((Byte)q).byteValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek1();
    }
    public static char getfield_C(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.CHAR);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0;
            return ((Character)q).charValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return (char)HeapAddress.addressOf(o).offset(f.getOffset()).peek4();
    }
    public static short getfield_S(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.SHORT);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return 0;
            return ((Short)q).shortValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek2();
    }
    public static boolean getfield_Z(Object o, jq_InstanceField f) {
        jq.Assert(f.getType() == jq_Primitive.BOOLEAN);
        if (jq.Bootstrapping) {
            Object q = obj_trav.getInstanceFieldValue(o, f);
            if (q == null) return false;
            return ((Boolean)q).booleanValue();
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        return HeapAddress.addressOf(o).offset(f.getOffset()).peek1()!=0;
    }
    public static Object getfield(Object o, jq_InstanceField f) {
        if (jq.Bootstrapping) return obj_trav.getInstanceFieldValue(o, f);
        jq_Type t = f.getType();
        if (t.isReferenceType()) return getfield_A(o, f);
        if (t == jq_Primitive.INT) return new Integer(getfield_I(o, f));
        if (t == jq_Primitive.FLOAT) return new Float(getfield_F(o, f));
        if (t == jq_Primitive.LONG) return new Long(getfield_L(o, f));
        if (t == jq_Primitive.DOUBLE) return new Double(getfield_D(o, f));
        if (t == jq_Primitive.BYTE) return new Byte(getfield_B(o, f));
        if (t == jq_Primitive.CHAR) return new Character(getfield_C(o, f));
        if (t == jq_Primitive.SHORT) return new Short(getfield_S(o, f));
        if (t == jq_Primitive.BOOLEAN) return new Boolean(getfield_Z(o, f));
        jq.UNREACHABLE();
        return null;
    }
    public static void putfield_I(Object o, jq_InstanceField f, int v) {
        jq.Assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Integer(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke4(v);
    }
    public static void putfield_L(Object o, jq_InstanceField f, long v) {
        jq.Assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Long(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke8(v);
    }
    public static void putfield_F(Object o, jq_InstanceField f, float v) {
        jq.Assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Float(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        putfield_I(o, f, Float.floatToRawIntBits(v));
    }
    public static void putfield_D(Object o, jq_InstanceField f, double v) {
        jq.Assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Double(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        putfield_L(o, f, Double.doubleToRawLongBits(v));
    }
    public static void putfield_A(Object o, jq_InstanceField f, Object v) {
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, v);
            return;
        }
        jq.Assert(v == null || TypeCheck.isAssignable(jq_Reference.getTypeOf(v), f.getType()));
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke(HeapAddress.addressOf(v));
    }
    public static void putfield_P(Object o, jq_InstanceField f, Address v) {
        jq.Assert(f.getType().isAddressType());
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, v);
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke(v);
    }
    public static void putfield_B(Object o, jq_InstanceField f, byte v) {
        jq.Assert(f.getType() == jq_Primitive.BYTE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Byte(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke1(v);
    }
    public static void putfield_C(Object o, jq_InstanceField f, char v) {
        jq.Assert(f.getType() == jq_Primitive.CHAR);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Character(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke2((short)((v<<16)>>16));
    }
    public static void putfield_S(Object o, jq_InstanceField f, short v) {
        jq.Assert(f.getType() == jq_Primitive.SHORT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Short(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke2(v);
    }
    public static void putfield_Z(Object o, jq_InstanceField f, boolean v) {
        jq.Assert(f.getType() == jq_Primitive.BOOLEAN);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Boolean(v));
            return;
        }
        jq.Assert(TypeCheck.isAssignable(jq_Reference.getTypeOf(o), f.getDeclaringClass()));
        HeapAddress.addressOf(o).offset(f.getOffset()).poke1(v?(byte)1:(byte)0);
    }
    
    public static int getstatic_I(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0;
            return ((Integer)o).intValue();
        }
        return f.getAddress().peek4();
    }
    public static long getstatic_L(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0L;
            return ((Long)o).longValue();
        }
        return f.getAddress().peek8();
    }
    public static float getstatic_F(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0L;
            return ((Float)o).floatValue();
        }
        return Float.intBitsToFloat(getstatic_I(f));
    }
    public static double getstatic_D(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0L;
            return ((Double)o).doubleValue();
        }
        return Double.longBitsToDouble(getstatic_L(f));
    }
    public static Object getstatic_A(jq_StaticField f) {
        jq.Assert(f.getType().isReferenceType() && !f.getType().isAddressType());
        if (jq.Bootstrapping) return obj_trav.getStaticFieldValue(f);
        return ((HeapAddress) f.getAddress().peek()).asObject();
    }
    public static Address getstatic_P(jq_StaticField f) {
        jq.Assert(f.getType().isAddressType());
        if (jq.Bootstrapping) {
            Address a = (Address)obj_trav.getStaticFieldValue(f);
            //if (a == null) return HeapAddress.getNull();
            return a;
        }
        return f.getAddress().peek();
    }
    public static boolean getstatic_Z(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.BOOLEAN);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return false;
            return ((Boolean)o).booleanValue();
        }
        return f.getAddress().peek4()!=0;
    }
    public static byte getstatic_B(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.BYTE);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0;
            return ((Byte)o).byteValue();
        }
        return f.getAddress().peek1();
    }
    public static short getstatic_S(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.SHORT);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0;
            return ((Short)o).shortValue();
        }
        return f.getAddress().peek2();
    }
    public static char getstatic_C(jq_StaticField f) {
        jq.Assert(f.getType() == jq_Primitive.CHAR);
        if (jq.Bootstrapping) {
            Object o = obj_trav.getStaticFieldValue(f);
            if (o == null) return 0;
            return ((Character)o).charValue();
        }
        return (char)f.getAddress().peek4();
    }
    public static void putstatic_I(jq_StaticField f, int v) {
        jq.Assert(f.getType() == jq_Primitive.INT);
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_L(jq_StaticField f, long v) {
        jq.Assert(f.getType() == jq_Primitive.LONG);
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_F(jq_StaticField f, float v) {
        jq.Assert(f.getType() == jq_Primitive.FLOAT);
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_D(jq_StaticField f, double v) {
        jq.Assert(f.getType() == jq_Primitive.DOUBLE);
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_A(jq_StaticField f, Object v) {
        jq.Assert(v == null || TypeCheck.isAssignable(jq_Reference.getTypeOf(v), f.getType()));
        jq.Assert(!f.getType().isAddressType());
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_P(jq_StaticField f, Address v) {
        jq.Assert(f.getType().isAddressType());
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_Z(jq_StaticField f, boolean v) {
        jq.Assert(f.getType() == jq_Primitive.BOOLEAN);
        f.getDeclaringClass().setStaticData(f, v?1:0);
    }
    public static void putstatic_B(jq_StaticField f, int v) {
        jq.Assert(f.getType() == jq_Primitive.BYTE);
        f.getDeclaringClass().setStaticData(f, (int)v);
    }
    public static void putstatic_S(jq_StaticField f, short v) {
        jq.Assert(f.getType() == jq_Primitive.SHORT);
        f.getDeclaringClass().setStaticData(f, (int)v);
    }
    public static void putstatic_C(jq_StaticField f, char v) {
        jq.Assert(f.getType() == jq_Primitive.CHAR);
        f.getDeclaringClass().setStaticData(f, (int)v);
    }
    
    public static int arraylength(Object o) {
        jq.Assert(getTypeOf(o).isArrayType());
        if (jq.Bootstrapping) return Array.getLength(o);
        return HeapAddress.addressOf(o).offset(Allocator.ObjectLayout.ARRAY_LENGTH_OFFSET).peek4();
    }
    public static Object arrayload_A(Object[] o, int i) {
        if (jq.Bootstrapping) return obj_trav.mapValue(o[i]);
        return o[i];
    }
    public static Address arrayload_R(Address[] o, int i) {
        return o[i];
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Reflection;");
    public static final jq_StaticField _obj_trav = _class.getOrCreateStaticField("obj_trav", "LBootstrap/ObjectTraverser;");
    public static final jq_StaticField _declaredFieldsCache = _class.getOrCreateStaticField("declaredFieldsCache", "Ljava/util/HashMap;");

}

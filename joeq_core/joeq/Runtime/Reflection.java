/*
 * Reflection.java
 *
 * Created on January 12, 2001, 12:48 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Bootstrap.PrimordialClassLoader;
import Bootstrap.ObjectTraverser;
import Clazz.jq_Type;
import Clazz.jq_Class;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_Method;
import Clazz.jq_StaticMethod;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticField;
import Clazz.jq_InstanceField;
import jq;

public abstract class Reflection {

    public static ObjectTraverser obj_trav;
    
    // Map between our jq_Type objects and JDK Class objects
    public static final jq_Type getJQType(Class c) {
        if (!jq.Bootstrapping) return (jq_Type)getfield_A(c, ClassLib.sun13.java.lang.Class._jq_type);
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
        //if (c == org.jos.java.util.zip.ZipFile.class)
        //    return PrimordialClassLoader.loader.getOrCreateType("Ljava/util/zip/ZipFile;");
        String className = c.getName().replace('.','/');
        if (!className.startsWith("[")) className = "L"+className+";";
        return PrimordialClassLoader.loader.getOrCreateBSType(className);
    }
    
    public static final Class getJDKType(jq_Type c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        if (c.isPrimitiveType()) 
            return getJDKType((jq_Primitive)c);
        else
            return getJDKType((jq_Reference)c);
    }
    public static final Class getJDKType(jq_Primitive c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        if (c == jq_Primitive.BYTE) return Byte.TYPE;
        if (c == jq_Primitive.CHAR) return Character.TYPE;
        if (c == jq_Primitive.DOUBLE) return Double.TYPE;
        if (c == jq_Primitive.FLOAT) return Float.TYPE;
        if (c == jq_Primitive.INT) return Integer.TYPE;
        if (c == jq_Primitive.LONG) return Long.TYPE;
        if (c == jq_Primitive.SHORT) return Short.TYPE;
        if (c == jq_Primitive.BOOLEAN) return Boolean.TYPE;
        if (c == jq_Primitive.VOID) return Void.TYPE;
        jq.UNREACHABLE(c.getName());
        return null;
    }
    public static Class getJDKType(jq_Reference c) {
        if (!jq.Bootstrapping) return c.getJavaLangClassObject();
        try {
            return Class.forName(c.getJDKName());
        } catch (ClassNotFoundException x) {
            jq.UNREACHABLE(c.getName());
            return null;
        }
    }
    
    public static void invokestatic_V(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
        }
    }
    public static int invokestatic_I(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            return (int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return 0;
        }
    }
    public static Object invokestatic_A(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            return Unsafe.asObject((int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint()));
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static long invokestatic_J(jq_StaticMethod m) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return 0L;
        }
    }
    public static void invokestatic_V(jq_StaticMethod m, Object arg1) throws Throwable {
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static Object invokeinstance_A(jq_InstanceMethod m, Object dis) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            return Unsafe.asObject((int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint()));
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static Object invokeinstance_A(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            return Unsafe.asObject((int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint()));
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    public static boolean invokeinstance_Z(jq_InstanceMethod m, Object dis, Object arg1) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            return ((int)Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint())) != 0;
        } else {
            jq.UNREACHABLE();
            return false;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, Object arg2) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            Unsafe.pushArg(Unsafe.addressOf(arg2));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, Object arg2, Object arg3) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
            Unsafe.pushArg(Unsafe.addressOf(arg2));
            Unsafe.pushArg(Unsafe.addressOf(arg3));
            Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            return;
        } else {
            jq.UNREACHABLE();
            return;
        }
    }
    public static void invokeinstance_V(jq_InstanceMethod m, Object dis, Object arg1, int arg2, long arg3, int arg4) throws Throwable {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(dis), m.getDeclaringClass()));
        if (!jq.Bootstrapping) {
            jq.assert(m.getDeclaringClass().isClsInitRunning());
            Unsafe.pushArg(Unsafe.addressOf(dis));
            Unsafe.pushArg(Unsafe.addressOf(arg1));
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
    public static long invoke(jq_Method m, Object[] args) throws Throwable {
        jq_Type[] params = m.getParamTypes();
        jq.assert(params.length == args.length);
        for (int i=0; i<params.length; ++i) {
            jq_Type c = params[i];
            if (c.isReferenceType()) {
                Unsafe.pushArg(Unsafe.addressOf(args[i]));
            } else {
                if (c == jq_Primitive.BYTE) {
                    int v = (int)((Byte)args[i]).byteValue();
                    Unsafe.pushArg(v);
                } else if (c == jq_Primitive.CHAR) {
                    int v = (int)((Character)args[i]).charValue();
                    Unsafe.pushArg(v);
                } else if (c == jq_Primitive.DOUBLE) {
                    long v = Double.doubleToRawLongBits(((Double)args[i]).doubleValue());
                    Unsafe.pushArg((int)(v >> 32)); // hi
                    Unsafe.pushArg((int)v);         // lo
                } else if (c == jq_Primitive.FLOAT) {
                    int v = Float.floatToRawIntBits(((Float)args[i]).floatValue());
                    Unsafe.pushArg(v);
                } else if (c == jq_Primitive.INT) {
                    int v = ((Integer)args[i]).intValue();
                    Unsafe.pushArg(v);
                } else if (c == jq_Primitive.LONG) {
                    long v = ((Long)args[i]).longValue();
                    Unsafe.pushArg((int)(v >> 32)); // hi
                    Unsafe.pushArg((int)v);         // lo
                } else if (c == jq_Primitive.SHORT) {
                    int v = ((Short)args[i]).shortValue();
                    Unsafe.pushArg(v);
                } else if (c == jq_Primitive.BOOLEAN) {
                    int v = ((Boolean)args[i]).booleanValue()?1:0;
                    Unsafe.pushArg(v);
                } else jq.UNREACHABLE(c.toString());
            }
        }
        return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
    }
    

    public static int getfield_I(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) return ((Integer)obj_trav.getInstanceFieldValue(o, f)).intValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return Unsafe.peek(Unsafe.addressOf(o)+f.getOffset());
    }
    public static long getfield_L(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) return ((Long)obj_trav.getInstanceFieldValue(o, f)).longValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        int lo=Unsafe.peek(Unsafe.addressOf(o)+f.getOffset());
        int hi=Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()+4);
        return jq.twoIntsToLong(lo, hi);
    }
    public static float getfield_F(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) return ((Float)obj_trav.getInstanceFieldValue(o, f)).floatValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return Float.intBitsToFloat(getfield_I(o, f));
    }
    public static double getfield_D(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) return ((Double)obj_trav.getInstanceFieldValue(o, f)).doubleValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return Double.longBitsToDouble(getfield_L(o, f));
    }
    public static Object getfield_A(Object o, jq_InstanceField f) {
        jq.assert(f.getType().isReferenceType());
        if (jq.Bootstrapping) return obj_trav.getInstanceFieldValue(o, f);
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return Unsafe.asObject(Unsafe.peek(Unsafe.addressOf(o)+f.getOffset()));
    }
    public static byte getfield_B(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.BYTE);
        if (jq.Bootstrapping) return ((Byte)obj_trav.getInstanceFieldValue(o, f)).byteValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return (byte)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset());
    }
    public static char getfield_C(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.CHAR);
        if (jq.Bootstrapping) return ((Character)obj_trav.getInstanceFieldValue(o, f)).charValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return (char)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset());
    }
    public static short getfield_S(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.SHORT);
        if (jq.Bootstrapping) return ((Short)obj_trav.getInstanceFieldValue(o, f)).shortValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return (short)Unsafe.peek(Unsafe.addressOf(o)+f.getOffset());
    }
    public static boolean getfield_Z(Object o, jq_InstanceField f) {
        jq.assert(f.getType() == jq_Primitive.BOOLEAN);
        if (jq.Bootstrapping) return ((Boolean)obj_trav.getInstanceFieldValue(o, f)).booleanValue();
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        return (Unsafe.peek(Unsafe.addressOf(o)+f.getOffset())&0xFF)!=0;
    }
    public static void putfield_I(Object o, jq_InstanceField f, int v) {
        jq.assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Integer(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), v);
    }
    public static void putfield_L(Object o, jq_InstanceField f, long v) {
        jq.assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Long(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), (int)v);
        Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset()+4, (int)(v>>32));
    }
    public static void putfield_F(Object o, jq_InstanceField f, float v) {
        jq.assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Float(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        putfield_I(o, f, Float.floatToRawIntBits(v));
    }
    public static void putfield_D(Object o, jq_InstanceField f, double v) {
        jq.assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Double(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        putfield_L(o, f, Double.doubleToRawLongBits(v));
    }
    public static void putfield_A(Object o, jq_InstanceField f, Object v) {
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, v);
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(v), f.getType()));
        Unsafe.poke4(Unsafe.addressOf(o)+f.getOffset(), Unsafe.addressOf(v));
    }
    public static void putfield_B(Object o, jq_InstanceField f, byte v) {
        jq.assert(f.getType() == jq_Primitive.BYTE);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Byte(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke1(Unsafe.addressOf(o)+f.getOffset(), v);
    }
    public static void putfield_C(Object o, jq_InstanceField f, char v) {
        jq.assert(f.getType() == jq_Primitive.CHAR);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Character(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke2(Unsafe.addressOf(o)+f.getOffset(), (short)((v<<16)>>16));
    }
    public static void putfield_S(Object o, jq_InstanceField f, short v) {
        jq.assert(f.getType() == jq_Primitive.SHORT);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Short(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke2(Unsafe.addressOf(o)+f.getOffset(), v);
    }
    public static void putfield_Z(Object o, jq_InstanceField f, boolean v) {
        jq.assert(f.getType() == jq_Primitive.BOOLEAN);
        if (jq.Bootstrapping) {
            obj_trav.putInstanceFieldValue(o, f, new Boolean(v));
            return;
        }
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(o), f.getDeclaringClass()));
        Unsafe.poke1(Unsafe.addressOf(o)+f.getOffset(), v?(byte)1:(byte)0);
    }
    
    public static int getstatic_I(jq_StaticField f) {
        jq.assert(f.getType() == jq_Primitive.INT || f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) return ((Integer)obj_trav.getStaticFieldValue(f)).intValue();
        return Unsafe.peek(f.getAddress());
    }
    public static long getstatic_L(jq_StaticField f) {
        jq.assert(f.getType() == jq_Primitive.LONG || f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) return ((Long)obj_trav.getStaticFieldValue(f)).longValue();
        int lo=Unsafe.peek(f.getAddress()); int hi=Unsafe.peek(f.getAddress()+4);
        return jq.twoIntsToLong(lo, hi);
    }
    public static float getstatic_F(jq_StaticField f) {
        jq.assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) return ((Float)obj_trav.getStaticFieldValue(f)).floatValue();
        return Float.intBitsToFloat(getstatic_I(f));
    }
    public static double getstatic_D(jq_StaticField f) {
        jq.assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) return ((Double)obj_trav.getStaticFieldValue(f)).doubleValue();
        return Double.longBitsToDouble(getstatic_L(f));
    }
    public static Object getstatic_A(jq_StaticField f) {
        jq.assert(f.getType().isReferenceType());
        if (jq.Bootstrapping) return obj_trav.getStaticFieldValue(f);
        return Unsafe.asObject(Unsafe.peek(f.getAddress()));
    }
    public static void putstatic_I(jq_StaticField f, int v) {
        jq.assert(f.getType() == jq_Primitive.INT);
        if (jq.Bootstrapping) {
            f.getDeclaringClass().setStaticData(f, v);
            return;
        }
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_L(jq_StaticField f, long v) {
        jq.assert(f.getType() == jq_Primitive.LONG);
        if (jq.Bootstrapping) {
            f.getDeclaringClass().setStaticData(f, v);
            return;
        }
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_F(jq_StaticField f, float v) {
        jq.assert(f.getType() == jq_Primitive.FLOAT);
        if (jq.Bootstrapping) {
            f.getDeclaringClass().setStaticData(f, v);
            return;
        }
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_D(jq_StaticField f, double v) {
        jq.assert(f.getType() == jq_Primitive.DOUBLE);
        if (jq.Bootstrapping) {
            f.getDeclaringClass().setStaticData(f, v);
            return;
        }
        f.getDeclaringClass().setStaticData(f, v);
    }
    public static void putstatic_A(jq_StaticField f, Object v) {
        jq.assert(TypeCheck.isAssignable(Unsafe.getTypeOf(v), f.getType()));
        if (jq.Bootstrapping) {
            f.getDeclaringClass().setStaticData(f, v);
            return;
        }
        f.getDeclaringClass().setStaticData(f, v);
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Reflection;");
    public static final jq_StaticMethod _invokestatic_noargs = _class.getOrCreateStaticMethod("invokestatic_J", "(LClazz/jq_StaticMethod;)J");
    public static final jq_StaticField _obj_trav = _class.getOrCreateStaticField("obj_trav", "LBootstrap/ObjectTraverser;");

}

/*
 * Field.java
 *
 * Created on April 13, 2001, 6:19 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang.reflect;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.*;
import jq;

public abstract class Field {
    
    // additional instance field.
    public final jq_Field jq_field = null;
    
    // overridden implementations.
    public static java.lang.Class getDeclaringClass(java.lang.reflect.Field dis) {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        return jq_f.getDeclaringClass().getJavaLangClassObject();
    }
    public static String getName(java.lang.reflect.Field dis) {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        return jq_f.getName().toString();
    }
    public static int getModifiers(java.lang.reflect.Field dis) {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        return jq_f.getAccessFlags();
    }
    public static Class getType(java.lang.reflect.Field dis) {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        return jq_f.getType().getJavaLangClassObject();
    }
    public static boolean equals(java.lang.reflect.Field dis, Object obj) {
        return dis == obj;
    }
    
    // native method implementations.
    public static java.lang.Object get(java.lang.reflect.Field dis, java.lang.Object obj)
	throws java.lang.IllegalArgumentException, java.lang.IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t.isReferenceType()) return Reflection.getstatic_A(sf);
            if (t == jq_Primitive.INT) return new Integer(Reflection.getstatic_I(sf));
            if (t == jq_Primitive.FLOAT) return new Float(Reflection.getstatic_F(sf));
            if (t == jq_Primitive.LONG) return new Long(Reflection.getstatic_L(sf));
            if (t == jq_Primitive.DOUBLE) return new Double(Reflection.getstatic_D(sf));
            if (t == jq_Primitive.BOOLEAN) return new Boolean(Reflection.getstatic_Z(sf));
            if (t == jq_Primitive.BYTE) return new Byte(Reflection.getstatic_B(sf));
            if (t == jq_Primitive.SHORT) return new Short(Reflection.getstatic_S(sf));
            if (t == jq_Primitive.CHAR) return new Character(Reflection.getstatic_C(sf));
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t.isReferenceType()) return Reflection.getfield_A(obj, f);
            if (t == jq_Primitive.INT) return new Integer(Reflection.getfield_I(obj, f));
            if (t == jq_Primitive.FLOAT) return new Float(Reflection.getfield_F(obj, f));
            if (t == jq_Primitive.LONG) return new Long(Reflection.getfield_L(obj, f));
            if (t == jq_Primitive.DOUBLE) return new Double(Reflection.getfield_D(obj, f));
            if (t == jq_Primitive.BOOLEAN) return new Boolean(Reflection.getfield_Z(obj, f));
            if (t == jq_Primitive.BYTE) return new Byte(Reflection.getfield_B(obj, f));
            if (t == jq_Primitive.SHORT) return new Short(Reflection.getfield_S(obj, f));
            if (t == jq_Primitive.CHAR) return new Character(Reflection.getfield_C(obj, f));
        }
        jq.UNREACHABLE();
        return null;
    }

    public static boolean getBoolean(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.BOOLEAN) return Reflection.getstatic_Z(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.BOOLEAN) return Reflection.getfield_Z(obj, f);
        }
        jq.UNREACHABLE();
        return false;
    }
    
    public static byte getByte(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.BYTE) return Reflection.getstatic_B(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.BYTE) return Reflection.getfield_B(obj, f);
        }
        jq.UNREACHABLE();
        return (byte)0;
    }
    
    public static char getChar(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.CHAR) return Reflection.getstatic_C(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.CHAR) return Reflection.getfield_C(obj, f);
        }
        jq.UNREACHABLE();
        return (char)0;
    }
    
    // byte -> short
    public static short getShort(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.SHORT) return Reflection.getstatic_S(sf);
            if (t == jq_Primitive.BYTE) return (short)Reflection.getstatic_B(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.SHORT) return Reflection.getfield_S(obj, f);
            if (t == jq_Primitive.BYTE) return (short)Reflection.getfield_B(obj, f);
        }
        jq.UNREACHABLE();
        return (short)0;
    }
    
    // byte -> int
    // char -> int
    // short -> int
    public static int getInt(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.INT) return Reflection.getstatic_I(sf);
            if (t == jq_Primitive.BYTE) return (int)Reflection.getstatic_B(sf);
            if (t == jq_Primitive.SHORT) return (int)Reflection.getstatic_S(sf);
            if (t == jq_Primitive.CHAR) return (int)Reflection.getstatic_C(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.INT) return Reflection.getfield_I(obj, f);
            if (t == jq_Primitive.BYTE) return (int)Reflection.getfield_B(obj, f);
            if (t == jq_Primitive.SHORT) return (int)Reflection.getfield_S(obj, f);
            if (t == jq_Primitive.CHAR) return (int)Reflection.getfield_C(obj, f);
        }
        jq.UNREACHABLE();
        return 0;
    }
    
    // byte -> long
    // char -> long
    // short -> long
    // int -> long
    public static long getLong(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.LONG) return Reflection.getstatic_L(sf);
            if (t == jq_Primitive.BYTE) return (long)Reflection.getstatic_B(sf);
            if (t == jq_Primitive.SHORT) return (long)Reflection.getstatic_S(sf);
            if (t == jq_Primitive.CHAR) return (long)Reflection.getstatic_C(sf);
            if (t == jq_Primitive.INT) return (long)Reflection.getstatic_I(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.LONG) return Reflection.getfield_L(obj, f);
            if (t == jq_Primitive.BYTE) return (long)Reflection.getfield_B(obj, f);
            if (t == jq_Primitive.SHORT) return (long)Reflection.getfield_S(obj, f);
            if (t == jq_Primitive.CHAR) return (long)Reflection.getfield_C(obj, f);
            if (t == jq_Primitive.INT) return (long)Reflection.getfield_I(obj, f);
        }
        jq.UNREACHABLE();
        return 0L;
    }
    
    // byte -> float
    // char -> float
    // short -> float
    // int -> float
    // long -> float
    public static float getFloat(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.FLOAT) return Reflection.getstatic_F(sf);
            if (t == jq_Primitive.BYTE) return (float)Reflection.getstatic_B(sf);
            if (t == jq_Primitive.SHORT) return (float)Reflection.getstatic_S(sf);
            if (t == jq_Primitive.CHAR) return (float)Reflection.getstatic_C(sf);
            if (t == jq_Primitive.INT) return (float)Reflection.getstatic_I(sf);
            if (t == jq_Primitive.LONG) return (float)Reflection.getstatic_L(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.FLOAT) return Reflection.getfield_F(obj, f);
            if (t == jq_Primitive.BYTE) return (float)Reflection.getfield_B(obj, f);
            if (t == jq_Primitive.SHORT) return (float)Reflection.getfield_S(obj, f);
            if (t == jq_Primitive.CHAR) return (float)Reflection.getfield_C(obj, f);
            if (t == jq_Primitive.INT) return (float)Reflection.getfield_I(obj, f);
            if (t == jq_Primitive.LONG) return (float)Reflection.getfield_L(obj, f);
        }
        jq.UNREACHABLE();
        return 0F;
    }
    
    // byte -> double
    // char -> double
    // short -> double
    // int -> double
    // long -> double
    // float -> double
    public static double getDouble(java.lang.reflect.Field dis, java.lang.Object obj)
	throws IllegalArgumentException, IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.DOUBLE) return Reflection.getstatic_D(sf);
            if (t == jq_Primitive.BYTE) return (double)Reflection.getstatic_B(sf);
            if (t == jq_Primitive.SHORT) return (double)Reflection.getstatic_S(sf);
            if (t == jq_Primitive.CHAR) return (double)Reflection.getstatic_C(sf);
            if (t == jq_Primitive.INT) return (double)Reflection.getstatic_I(sf);
            if (t == jq_Primitive.LONG) return (double)Reflection.getstatic_L(sf);
            if (t == jq_Primitive.FLOAT) return (double)Reflection.getstatic_L(sf);
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.DOUBLE) return Reflection.getfield_F(obj, f);
            if (t == jq_Primitive.BYTE) return (double)Reflection.getfield_B(obj, f);
            if (t == jq_Primitive.SHORT) return (double)Reflection.getfield_S(obj, f);
            if (t == jq_Primitive.CHAR) return (double)Reflection.getfield_C(obj, f);
            if (t == jq_Primitive.INT) return (double)Reflection.getfield_I(obj, f);
            if (t == jq_Primitive.LONG) return (double)Reflection.getfield_L(obj, f);
            if (t == jq_Primitive.FLOAT) return (double)Reflection.getfield_L(obj, f);
        }
        jq.UNREACHABLE();
        return 0F;
    }
    
    public static void set(java.lang.reflect.Field dis, java.lang.Object obj, java.lang.Object value)
	throws java.lang.IllegalArgumentException, java.lang.IllegalAccessException
    {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t.isReferenceType()) Reflection.putstatic_A(sf, value);
            else if (t == jq_Primitive.INT) {
                int val = Reflection.unwrapToInt(value);
                Reflection.putstatic_I(sf, val);
            }
            else if (t == jq_Primitive.FLOAT) {
                float val = Reflection.unwrapToFloat(value);
                Reflection.putstatic_F(sf, val);
            }
            else if (t == jq_Primitive.LONG) {
                long val = Reflection.unwrapToLong(value);
                Reflection.putstatic_L(sf, val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                double val = Reflection.unwrapToDouble(value);
                Reflection.putstatic_D(sf, val);
            }
            else if (t == jq_Primitive.BOOLEAN) {
                boolean val = Reflection.unwrapToBoolean(value);
                Reflection.putstatic_Z(sf, val);
            }
            else if (t == jq_Primitive.BYTE) {
                byte val = Reflection.unwrapToByte(value);
                Reflection.putstatic_B(sf, val);
            }
            else if (t == jq_Primitive.SHORT) {
                short val = Reflection.unwrapToShort(value);
                Reflection.putstatic_S(sf, val);
            }
            else if (t == jq_Primitive.CHAR) {
                char val = Reflection.unwrapToChar(value);
                Reflection.putstatic_C(sf, val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t.isReferenceType()) {
                Reflection.getfield_A(obj, f);
            }
            else if (t == jq_Primitive.INT) {
                int val = Reflection.unwrapToInt(value);
                Reflection.putfield_I(obj, f, val);
            }
            else if (t == jq_Primitive.FLOAT) {
                float val = Reflection.unwrapToFloat(value);
                Reflection.putfield_F(obj, f, val);
            }
            else if (t == jq_Primitive.LONG) {
                long val = Reflection.unwrapToLong(value);
                Reflection.putfield_L(obj, f, val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                double val = Reflection.unwrapToDouble(value);
                Reflection.putfield_D(obj, f, val);
            }
            else if (t == jq_Primitive.BOOLEAN) {
                boolean val = Reflection.unwrapToBoolean(value);
                Reflection.putfield_Z(obj, f, val);
            }
            else if (t == jq_Primitive.BYTE) {
                byte val = Reflection.unwrapToByte(value);
                Reflection.putfield_B(obj, f, val);
            }
            else if (t == jq_Primitive.SHORT) {
                short val = Reflection.unwrapToShort(value);
                Reflection.putfield_S(obj, f, val);
            }
            else if (t == jq_Primitive.CHAR) {
                char val = Reflection.unwrapToChar(value);
                Reflection.putfield_C(obj, f, val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setBoolean(java.lang.reflect.Field dis, java.lang.Object obj, boolean val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.BOOLEAN) {
                Reflection.putstatic_Z(sf, val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.BOOLEAN) {
                Reflection.putfield_Z(obj, f, val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setByte(java.lang.reflect.Field dis, java.lang.Object obj, byte val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.BYTE) {
                Reflection.putstatic_B(sf, val);
            }
            else if (t == jq_Primitive.SHORT) {
                Reflection.putstatic_S(sf, (short)val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putstatic_I(sf, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putstatic_L(sf, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.BYTE) {
                Reflection.putfield_B(obj, f, val);
            }
            else if (t == jq_Primitive.SHORT) {
                Reflection.putfield_S(obj, f, (short)val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putfield_I(obj, f, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putfield_L(obj, f, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setChar(java.lang.reflect.Field dis, java.lang.Object obj, char val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.CHAR) {
                Reflection.putstatic_C(sf, val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putstatic_I(sf, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putstatic_L(sf, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.CHAR) {
                Reflection.putfield_C(obj, f, val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putfield_I(obj, f, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putfield_L(obj, f, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setShort(java.lang.reflect.Field dis, java.lang.Object obj, short val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.SHORT) {
                Reflection.putstatic_S(sf, val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putstatic_I(sf, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putstatic_L(sf, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.SHORT) {
                Reflection.putfield_S(obj, f, val);
            }
            else if (t == jq_Primitive.INT) {
                Reflection.putfield_I(obj, f, (int)val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putfield_L(obj, f, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setInt(java.lang.reflect.Field dis, java.lang.Object obj, int val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.INT) {
                Reflection.putstatic_I(sf, val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putstatic_L(sf, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.INT) {
                Reflection.putfield_I(obj, f, val);
            }
            else if (t == jq_Primitive.LONG) {
                Reflection.putfield_L(obj, f, (long)val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setFloat(java.lang.reflect.Field dis, java.lang.Object obj, float val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setLong(java.lang.reflect.Field dis, java.lang.Object obj, long val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.LONG) {
                Reflection.putstatic_L(sf, val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putstatic_F(sf, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, (double)val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.LONG) {
                Reflection.putfield_L(obj, f, val);
            }
            else if (t == jq_Primitive.FLOAT) {
                Reflection.putfield_F(obj, f, (float)val);
            }
            else if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, (double)val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    public static void setDouble(java.lang.reflect.Field dis, java.lang.Object obj, double val)
        throws IllegalArgumentException, IllegalAccessException {
        jq_Field jq_f = (jq_Field)Reflection.getfield_A(dis, _jq_field);
        jq_Type t = jq_f.getType();
        if (jq_f.isStatic()) {
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_StaticField sf = (jq_StaticField)jq_f;
            if (t == jq_Primitive.DOUBLE) {
                Reflection.putstatic_D(sf, val);
            }
            else jq.UNREACHABLE();
        } else {
            jq_Reference obj_t = Unsafe.getTypeOf(obj);
            if (!TypeCheck.isAssignable(obj_t, jq_f.getDeclaringClass())) {
                throw new IllegalArgumentException();
            }
            if (!dis.isAccessible()) jq_f.checkCallerAccess(3);
            if (!jq_f.isFinal()) {
                throw new IllegalAccessException();
            }
            jq_InstanceField f = (jq_InstanceField)jq_f;
            if (t == jq_Primitive.DOUBLE) {
                Reflection.putfield_D(obj, f, val);
            }
            else jq.UNREACHABLE();
        }
    }
    
    // additional methods.
    // ONLY TO BE CALLED BY jq_Member CONSTRUCTOR!!!
    public static java.lang.reflect.Field createNewField(jq_Class clazz, jq_Field jq_field) {
        java.lang.reflect.Field o = (java.lang.reflect.Field)_class.newInstance();
        Reflection.putfield_A(o, _jq_field, jq_field);
        return o;
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _jq_field;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Field;");
        _jq_field = _class.getOrCreateInstanceField("jq_field", "LClazz/jq_Field;");
    }
}

/*
 * Array.java
 *
 * Created on January 29, 2001, 1:25 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang.reflect;

import Run_Time.Unsafe;
import Run_Time.Reflection;
import Allocator.ObjectLayout;
import Allocator.HeapAllocator;
import ClassLib.ClassLibInterface;
import Clazz.jq_Array;
import Clazz.jq_Type;
import Clazz.jq_Class;
import Bootstrap.PrimordialClassLoader;

abstract class Array {

    public static int getLength(jq_Class clazz, Object array) throws IllegalArgumentException {
        if (!Unsafe.getTypeOf(array).isArrayType())throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
        return Unsafe.peek(Unsafe.addressOf(array)+ObjectLayout.ARRAY_LENGTH_OFFSET);
    }
    public static Object get(jq_Class clazz, Object array, int index)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof Object[]) return ((Object[])array)[index];
        if (array instanceof int[]) return new Integer(((int[])array)[index]);
        if (array instanceof long[]) return new Long(((long[])array)[index]);
        if (array instanceof float[]) return new Float(((float[])array)[index]);
        if (array instanceof double[]) return new Double(((double[])array)[index]);
        if (array instanceof boolean[]) return new Boolean(((boolean[])array)[index]);
        if (array instanceof byte[]) return new Byte(((byte[])array)[index]);
        if (array instanceof short[]) return new Short(((short[])array)[index]);
        if (array instanceof char[]) return new Character(((char[])array)[index]);
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static boolean getBoolean(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof boolean[]) return ((boolean[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static byte getByte(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof byte[]) return ((byte[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static char getChar(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof char[]) return ((char[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static short getShort(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof short[]) return ((short[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static int getInt(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof int[]) return ((int[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static long getLong(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof long[]) return ((long[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static float getFloat(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof float[]) return ((float[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static double getDouble(jq_Class clazz, Object array, int index) {
        if (array == null) throw new NullPointerException();
        if (array instanceof double[]) return ((double[])array)[index];
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void set(jq_Class clazz, Object array, int index, Object value)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof Object[]) {
            ((Object[])array)[index] = value;
            return;
        }
        if (array instanceof boolean[]) {
            boolean v;
            if (value instanceof Boolean) v = ((Boolean)value).booleanValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((boolean[])array)[index] = v;
            return;
        }
        if (array instanceof byte[]) {
            byte v;
            if (value instanceof Byte) v = ((Byte)value).byteValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((byte[])array)[index] = v;
            return;
        }
        if (array instanceof short[]) {
            short v;
            if (value instanceof Short) v = ((Short)value).shortValue();
            else if (value instanceof Byte) v = (short)((Byte)value).byteValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((short[])array)[index] = v;
            return;
        }
        if (array instanceof char[]) {
            char v;
            if (value instanceof Character) v = ((Character)value).charValue();
            else if (value instanceof Byte) v = (char)((Byte)value).byteValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((char[])array)[index] = v;
            return;
        }
        if (array instanceof int[]) {
            int v;
            if (value instanceof Integer) v = ((Integer)value).intValue();
            else if (value instanceof Character) v = (int)((Character)value).charValue();
            else if (value instanceof Short) v = (int)((Short)value).shortValue();
            else if (value instanceof Byte) v = (int)((Byte)value).byteValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((int[])array)[index] = v;
            return;
        }
        if (array instanceof long[]) {
            long v;
            if (value instanceof Long) v = ((Long)value).longValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((long[])array)[index] = v;
            return;
        }
        if (array instanceof float[]) {
            float v;
            if (value instanceof Float) v = ((Float)value).floatValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((float[])array)[index] = v;
            return;
        }
        if (array instanceof double[]) {
            double v;
            if (value instanceof Double) v = ((Double)value).doubleValue();
            else throw new IllegalArgumentException("cannot store value of type "+Unsafe.getTypeOf(value)+" into array of type "+Unsafe.getTypeOf(array));
            ((double[])array)[index] = v;
            return;
        }
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setBoolean(jq_Class clazz, Object array, int index, boolean z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof boolean[]) ((boolean[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setByte(jq_Class clazz, Object array, int index, byte z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof byte[]) ((byte[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setChar(jq_Class clazz, Object array, int index, char z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof char[]) ((char[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setShort(jq_Class clazz, Object array, int index, short z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof short[]) ((short[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setInt(jq_Class clazz, Object array, int index, int z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof int[]) ((int[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setLong(jq_Class clazz, Object array, int index, long z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof long[]) ((long[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setFloat(jq_Class clazz, Object array, int index, float z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof float[]) ((float[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    public static void setDouble(jq_Class clazz, Object array, int index, double z)
    throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (array == null) throw new NullPointerException();
        if (array instanceof double[]) ((double[])array)[index] = z;
        throw new IllegalArgumentException(Unsafe.getTypeOf(array).toString());
    }
    private static Object newArray(jq_Class clazz, Class componentType, int length)
    throws NegativeArraySizeException {
        jq_Type t = ClassLibInterface.i.getJQType(componentType);
        jq_Array a = t.getArrayTypeForElementType();
        a.load(); a.verify(); a.prepare(); a.sf_initialize(); a.cls_initialize();
        return a.newInstance(length);
    }
    private static Object multiNewArray(jq_Class clazz, Class componentType, int[] dimensions)
    throws IllegalArgumentException, NegativeArraySizeException {
        jq_Type a = ClassLibInterface.i.getJQType(componentType);
        if (dimensions.length == 0) throw new IllegalArgumentException();
        for (int i=0; i<dimensions.length; ++i) {
            a = a.getArrayTypeForElementType();
            a.load(); a.verify(); a.prepare(); a.sf_initialize(); a.cls_initialize();
        }
        return HeapAllocator.multinewarray_helper(dimensions, 0, (jq_Array)a);
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Array;");
}

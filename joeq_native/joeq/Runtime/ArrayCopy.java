/*
 * ArrayCopy.java
 *
 * Created on January 12, 2001, 12:12 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

public abstract class ArrayCopy {

    public static void arraycopy(Object src, int src_position,
                                 Object dst, int dst_position,
                                 int length) {
        if (dst instanceof Object[])
            if (src instanceof Object[])
                ArrayCopy.arraycopy((Object[])src, src_position, (Object[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof byte[])
            if (src instanceof byte[])
                ArrayCopy.arraycopy((byte[])src, src_position, (byte[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof char[])
            if (src instanceof char[])
                ArrayCopy.arraycopy((char[])src, src_position, (char[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof short[])
            if (src instanceof short[])
                ArrayCopy.arraycopy((short[])src, src_position, (short[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof int[])
            if (src instanceof int[])
                ArrayCopy.arraycopy((int[])src, src_position, (int[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof float[])
            if (src instanceof float[])
                ArrayCopy.arraycopy((float[])src, src_position, (float[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof long[])
            if (src instanceof long[])
                ArrayCopy.arraycopy((long[])src, src_position, (long[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else if (dst instanceof double[])
            if (src instanceof double[])
                ArrayCopy.arraycopy((double[])src, src_position, (double[])dst, dst_position, length);
            else
                throw new ArrayStoreException();
        else
            throw new ArrayStoreException();
    }
    
    public static void arraycopy(Object[] src, int src_position,
                                 Object[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(byte[] src, int src_position,
                                 byte[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(char[] src, int src_position,
                                 char[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(short[] src, int src_position,
                                 short[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(int[] src, int src_position,
                                 int[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(float[] src, int src_position,
                                 float[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(long[] src, int src_position,
                                 long[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    public static void arraycopy(double[] src, int src_position,
                                 double[] dst, int dst_position,
                                 int length) {
        if ((src == dst) && (dst_position < src_position) && (dst_position+length > src_position)) {
            // overlapping case
            for (int i=length-1; i>=0; --i) {
                dst[dst_position+i] = src[src_position+i];
            }
            return;
        }
        for (int i=0; i<length; ++i) {
            dst[dst_position+i] = src[src_position+i];
        }
    }
    
}

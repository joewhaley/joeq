package Util;

/**
 * Utility methods to convert between primitive data types.
 * 
 * @author John Whaley
 * @version $Id$
 */
public abstract class Convert {
    
    /**
     * Convert two bytes to a char.
     * 
     * @param b1 first byte
     * @param b2 second byte
     * @return char result
     */
    public static char twoBytesToChar(byte b1, byte b2) {
        return (char) ((b1 << 8) | (b2 & 0xFF));
    }

    /**
     * Convert two bytes to a short.
     * 
     * @param b1 first byte
     * @param b2 second byte
     * @return short result
     */
    public static short twoBytesToShort(byte b1, byte b2) {
        return (short) ((b1 << 8) | (b2 & 0xFF));
    }

    /**
     * Convert two chars to an int.
     * 
     * @param c1 first char
     * @param c2 second char
     * @return int result
     */
    public static int twoCharsToInt(char c1, char c2) {
        return (c1 << 16) | c2;
    }

    /**
     * Convert four bytes to an int.
     */
    public static int fourBytesToInt(byte b1, byte b2, byte b3, byte b4) {
        return (b1 << 24) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 8) | (b4 & 0xFF);
    }

    /**
     * Convert eight bytes to a long.
     */
    public static long eightBytesToLong(byte b1, byte b2, byte b3, byte b4, byte b5, byte b6, byte b7, byte b8) {
        int hi = fourBytesToInt(b1, b2, b3, b4);
        int lo = fourBytesToInt(b5, b6, b7, b8);
        return twoIntsToLong(lo, hi);
    }

    /**
     * Convert two bytes at the given position in an array to a char.
     */
    public static char twoBytesToChar(byte[] b, int i) {
        return Convert.twoBytesToChar(b[i], b[i + 1]);
    }

    /**
     * Convert two bytes at the given position in an array to a short.
     */
    public static short twoBytesToShort(byte[] b, int i) {
        return twoBytesToShort(b[i], b[i + 1]);
    }

    /**
     * Convert four bytes at the given position in an array to an int.
     */
    public static int fourBytesToInt(byte[] b, int i) {
        return fourBytesToInt(b[i], b[i + 1], b[i + 2], b[i + 3]);
    }

    /**
     * Convert eight bytes at the given position in an array to a long.
     */
    public static long eightBytesToLong(byte[] b, int i) {
        return eightBytesToLong(b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7]);
    }

    /**
     * Convert two ints to a long.
     */
    public static long twoIntsToLong(int lo, int hi) {
        return (((long) lo) & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    /**
     * Convert a char to two bytes, putting the result at the given position in
     * the given array.
     */
    public static void charToTwoBytes(char i, byte[] b, int index) {
        b[index] = (byte) (i >> 8);
        b[index + 1] = (byte) (i);
    }

    /**
     * Convert an int to four bytes, putting the result at the given position in
     * the given array.
     */
    public static void intToFourBytes(int i, byte[] b, int index) {
        b[index] = (byte) (i >> 24);
        b[index + 1] = (byte) (i >> 16);
        b[index + 2] = (byte) (i >> 8);
        b[index + 3] = (byte) (i);
    }

    /**
     * Convert a long to eight bytes, putting the result at the given position
     * in the given array.
     */
    public static void longToEightBytes(long i, byte[] b, int index) {
        b[index] = (byte) (i >> 56);
        b[index + 1] = (byte) (i >> 48);
        b[index + 2] = (byte) (i >> 40);
        b[index + 3] = (byte) (i >> 32);
        b[index + 4] = (byte) (i >> 24);
        b[index + 5] = (byte) (i >> 16);
        b[index + 6] = (byte) (i >> 8);
        b[index + 7] = (byte) (i);
    }
}

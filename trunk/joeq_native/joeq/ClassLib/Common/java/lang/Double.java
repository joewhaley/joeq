/*
 * Double.java
 *
 * Created on January 29, 2001, 11:14 AM
 *
 */

package ClassLib.Common.java.lang;

import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version 
 */
abstract class Double {

    // native method implementations.
    public static long doubleToLongBits(double value) {
        if (java.lang.Double.isNaN(value)) return 0x7ff8000000000000L;
        return Unsafe.doubleToLongBits(value);
    }
    public static long doubleToRawLongBits(double value) {
        return Unsafe.doubleToLongBits(value);
    }
    public static double longBitsToDouble(long bits) {
        return Unsafe.longBitsToDouble(bits);
    }
    
}

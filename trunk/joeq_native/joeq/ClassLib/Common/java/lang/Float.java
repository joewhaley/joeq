/*
 * Float.java
 *
 * Created on January 29, 2001, 11:13 AM
 *
 */

package ClassLib.Common.java.lang;

import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
abstract class Float {
    
    // native method implementations.
    public static int floatToIntBits(float value) {
        if (java.lang.Float.isNaN(value)) return 0x7fc00000;
        return Unsafe.floatToIntBits(value);
    }
    public static int floatToRawIntBits(float value) {
        return Unsafe.floatToIntBits(value);
    }
    public static float intBitsToFloat(int bits) {
        return Unsafe.intBitsToFloat(bits);
    }

}

/*
 * Double.java
 *
 * Created on January 29, 2001, 11:14 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Run_Time.Unsafe;

abstract class Double {

    // native method implementations.
    public static long doubleToLongBits(jq_Class clazz, double value) {
        if (java.lang.Double.isNaN(value)) return 0x7ff8000000000000L;
        return Unsafe.doubleToLongBits(value);
    }
    public static long doubleToRawLongBits(jq_Class clazz, double value) {
        return Unsafe.doubleToLongBits(value);
    }
    public static double longBitsToDouble(jq_Class clazz, long bits) {
        return Unsafe.longBitsToDouble(bits);
    }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Double;");
}

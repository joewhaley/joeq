/*
 * Float.java
 *
 * Created on January 29, 2001, 11:13 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun14_win32.java.lang;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Run_Time.Unsafe;

abstract class Float {
    
    // native method implementations.
    public static int floatToIntBits(jq_Class clazz, float value) {
        if (java.lang.Float.isNaN(value)) return 0x7fc00000;
        return Unsafe.floatToIntBits(value);
    }
    public static int floatToRawIntBits(jq_Class clazz, float value) {
        return Unsafe.floatToIntBits(value);
    }
    public static float intBitsToFloat(jq_Class clazz, int bits) {
        return Unsafe.intBitsToFloat(bits);
    }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Float;");
}

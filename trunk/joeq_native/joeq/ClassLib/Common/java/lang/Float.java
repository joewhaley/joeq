// Float.java, created Thu Jul  4  4:50:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.lang;

import Run_Time.Unsafe;

/**
 * Float
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
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

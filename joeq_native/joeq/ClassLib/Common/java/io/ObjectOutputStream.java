/*
 * ObjectOutputStream.java
 *
 * Created on July 8, 2002, 12:32 AM
 */

package ClassLib.Common.java.io;

import Main.jq;
import Run_Time.Unsafe;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ObjectOutputStream {

    private static void floatsToBytes(float[] src, int srcpos, byte[] dst, int dstpos, int nfloats) {
        --srcpos;
        while (--nfloats >= 0) {
            jq.intToFourBytes(Unsafe.floatToIntBits(src[++srcpos]), dst, dstpos);
            dstpos += 4;
        }
    }
    private static void doublesToBytes(double[] src, int srcpos, byte[] dst, int dstpos, int ndoubles) {
        --srcpos;
        while (--ndoubles >= 0) {
            jq.longToEightBytes(Unsafe.doubleToLongBits(src[++srcpos]), dst, dstpos);
            dstpos += 8;
        }
    }
    private static void getPrimitiveFieldValues(java.lang.Object obj, long[] fieldIDs, char[] typecodes, byte[] data) {
        jq.TODO();
    }
    private static java.lang.Object getObjectFieldValue(java.lang.Object obj, long fieldID) {
        jq.TODO();
        return null;
    }
    
}

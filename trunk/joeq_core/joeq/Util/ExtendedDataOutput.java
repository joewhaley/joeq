/*
 * ExtendedDataOutput.java
 *
 * Created on April 3, 2001, 8:29 PM
 *
 */

package Util;

import java.io.DataOutput;
import java.io.IOException;

/**
 * @author  John Whaley
 * @version $Id$
 */
public interface ExtendedDataOutput extends DataOutput {
    void writeULong(long v) throws IOException;
    void writeUInt(int v) throws IOException;
    void writeUShort(int v) throws IOException;
    void writeUByte(int v) throws IOException;
}

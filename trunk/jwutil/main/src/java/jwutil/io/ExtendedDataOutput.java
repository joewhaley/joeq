// ExtendedDataOutput.java, created Wed Mar  5  0:26:34 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.DataOutput;
import java.io.IOException;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public interface ExtendedDataOutput extends DataOutput {
    void writeULong(long v) throws IOException;
    void writeUInt(int v) throws IOException;
    void writeUShort(int v) throws IOException;
    void writeUByte(int v) throws IOException;
}

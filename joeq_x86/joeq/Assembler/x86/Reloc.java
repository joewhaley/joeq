/*
 * Reloc.java
 *
 * Created on February 9, 2001, 1:29 PM
 *
 */

package Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Reloc {

    public static final char RELOC_ADDR32 = (char)0x0006;
    public static final char RELOC_REL32  = (char)0x0014;
    
    public abstract void dumpCOFF(DataOutput out) throws IOException;
    public abstract void patch();
}

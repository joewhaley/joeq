/*
 * Reloc.java
 *
 * Created on February 9, 2001, 1:29 PM
 *
 * @author  John Whaley
 * @version 
 */

package Assembler.x86;

import java.io.IOException;
import java.io.OutputStream;

public abstract class Reloc {

    public static final char RELOC_ADDR32 = (char)0x0006;
    public static final char RELOC_REL32  = (char)0x0014;
    
    public abstract void dump(OutputStream out) throws IOException;
}

/*
 * SymbolTableEntry.java
 *
 * Created on February 6, 2002, 6:42 PM
 */

package Linker.ELF;

import java.io.*;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class RelocAEntry extends RelocEntry {

    protected int addend;
    
    public RelocAEntry(int offset, SymbolTableEntry e, byte type, int addend) {
        super(offset, e, type);
        this.addend = addend;
    }

    public final int getAddEnd() { return addend; }
    
    public void write(ELFFile file, OutputStream out) throws IOException {
        file.write_addr(out, getOffset());
        file.write_word(out, getInfo());
        file.write_sword(out, getAddEnd());
    }
    
    public static int getEntrySize() { return 12; }
}

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
public class RelocEntry {

    public static final byte R_386_NONE = 0;
    public static final byte R_386_32   = 1;
    public static final byte R_386_PC32 = 2;
    
    protected int offset;
    protected SymbolTableEntry e;
    protected byte type;
    
    public RelocEntry(int offset, SymbolTableEntry e, byte type) {
        this.offset = offset; this.e = e; this.type = type;
    }

    public final int getOffset() { return offset; }
    public final int getSymbolTableIndex() { return e.getIndex(); }
    public final int getType() { return type; }
    public final int getInfo() { return (e.getIndex() << 8) | (type & 0xFF); }
    
    public void write(ELFFile file, OutputStream out) throws IOException {
        file.write_addr(out, getOffset());
        file.write_word(out, getInfo());
    }
    
    public static int getEntrySize() { return 8; }
}

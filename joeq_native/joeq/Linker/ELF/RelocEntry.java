/*
 * RelocEntry.java
 *
 * Created on February 6, 2002, 6:42 PM
 */

package Linker.ELF;

import java.io.IOException;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class RelocEntry implements ELFConstants {

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
    
    public void write(ELF file) throws IOException {
        file.write_addr(getOffset());
        file.write_word(getInfo());
    }
    
    public static RelocEntry read(ELF file, Section.SymTabSection s) throws IOException {
        int offset = file.read_addr();
        int info = file.read_word();
        int stindex = (info >>> 8);
        byte type = (byte)info;
        SymbolTableEntry e = s.getSymbolTableEntry(stindex);
        return new RelocEntry(offset, e, type);
    }
    
    public static int getEntrySize() { return 8; }
}

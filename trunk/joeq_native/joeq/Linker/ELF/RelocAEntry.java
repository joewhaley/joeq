/*
 * RelocAEntry.java
 *
 * Created on February 6, 2002, 6:42 PM
 */

package Linker.ELF;

import java.io.*;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class RelocAEntry extends RelocEntry {

    protected int addend;
    
    public RelocAEntry(int offset, SymbolTableEntry e, byte type, int addend) {
        super(offset, e, type);
        this.addend = addend;
    }

    public final int getAddEnd() { return addend; }
    
    public void write(ELF file) throws IOException {
        file.write_addr(getOffset());
        file.write_word(getInfo());
        file.write_sword(getAddEnd());
    }
    
    public static RelocEntry read(ELF file, Section.SymTabSection s) throws IOException {
        int offset = file.read_addr();
        int info = file.read_word();
        int addend = file.read_sword();
        int stindex = (info >>> 8);
        byte type = (byte)info;
        SymbolTableEntry e = s.getSymbolTableEntry(stindex);
        return new RelocAEntry(offset, e, type, addend);
    }
    
    public static int getEntrySize() { return 12; }
}

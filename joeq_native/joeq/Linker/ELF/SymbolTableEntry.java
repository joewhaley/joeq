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
public class SymbolTableEntry {

    // Symbol Binding
    public static final int STB_LOCAL   = 0;
    public static final int STB_GLOBAL  = 1;
    public static final int STB_WEAK    = 2;
    public static final int STB_LOPROC  = 13;
    public static final int STB_HIPROC  = 15;
    
    // Symbol Types
    public static final int STT_NOTYPE  = 0;
    public static final int STT_OBJECT  = 1;
    public static final int STT_FUNC    = 2;
    public static final int STT_SECTION = 3;
    public static final int STT_FILE    = 4;
    public static final int STT_LOPROC  = 13;
    public static final int STT_HIPROC  = 15;
    
    protected int index;
    protected String name;
    protected int value;
    protected int size;
    protected byte info;
    protected Section section;
    
    public SymbolTableEntry(String name, int value, int size, byte info, Section section) {
        this.name = name; this.value = value; this.size = size; this.info = info; this.section = section;
    }

    public final String getName() { return name; }
    public final int getValue() { return value; }
    public final int getSize() { return size; }
    public final byte getInfo() { return info; }
    public final byte getOther() { return 0; }
    public final int getSHndx() { return section.getIndex(); }

    public final void setIndex(int index) { this.index = index; }
    public final int getIndex() { return this.index; }
    
    public void write(ELFFile file, OutputStream out) throws IOException {
        file.write_symbolname(out, getName());
        file.write_addr(out, getValue());
        file.write_word(out, getSize());
        out.write((byte)getInfo());
        out.write((byte)getOther());
        file.write_half(out, getSHndx());
    }
    
    public static int getEntrySize() { return 16; }
}

/*
 * ProgramHeader.java
 *
 * Created on February 7, 2002, 2:15 PM
 */

package Linker.ELF;
import java.io.*;

/**
 *
 * @author  John Whaley
 * @version 
 */
public abstract class ProgramHeader {

    // Segment Types
    public static final int PT_NULL     = 0;
    public static final int PT_LOAD     = 1;
    public static final int PT_DYNAMIC  = 2;
    public static final int PT_INTERP   = 3;
    public static final int PT_NOTE     = 4;
    public static final int PT_SHLIB    = 5;
    public static final int PT_PHDR     = 6;
    public static final int PT_LOPROC   = 0x70000000;
    public static final int PT_HIPROC   = 0x7fffffff;

    protected int index;
    
    protected int offset;
    protected int vaddr;
    protected int paddr;
    protected int filesz;
    protected int memsz;
    protected int flags;
    protected int align;
    
    public abstract int getType();
    public int getOffset() { return offset; }
    public int getVAddr() { return vaddr; }
    public int getPAddr() { return paddr; }
    public int getFileSz() { return filesz; }
    public int getMemSz() { return memsz; }
    public int getFlags() { return flags; }
    public int getAlign() { return align; }

    public int getIndex() { return index; }
    public void setIndex(int index) { this.index = index; }
    
    public void writeHeader(ELFFile file, OutputStream out) throws IOException {
        file.write_word(out, this.getType());
        file.write_off(out, this.getOffset());
        file.write_addr(out, this.getVAddr());
        file.write_addr(out, this.getPAddr());
        file.write_word(out, this.getFileSz());
        file.write_word(out, this.getMemSz());
        file.write_word(out, this.getFlags());
        file.write_word(out, this.getAlign());
    }

    public static class NullProgramHeader extends ProgramHeader {
        protected String name;
        protected byte[] desc;
        protected int type;
        public final int getType() { return PT_NULL; }
    }
    public static class LoadProgramHeader extends ProgramHeader {
        public final int getType() { return PT_LOAD; }
    }
    public static class DynamicProgramHeader extends ProgramHeader {
        public final int getType() { return PT_DYNAMIC; }
    }
    public static class InterpProgramHeader extends ProgramHeader {
        public final int getType() { return PT_INTERP; }
    }
    public static class NoteProgramHeader extends ProgramHeader {
        public final int getType() { return PT_NOTE; }
    }
    public static class PHdrProgramHeader extends ProgramHeader {
        public final int getType() { return PT_PHDR; }
    }

    public static int getSize() { return 32; }
}

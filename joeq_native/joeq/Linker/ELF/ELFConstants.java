/*
 * ELFConstants.java
 *
 * Created on May 21, 2002, 2:17 AM
 */

package Linker.ELF;

/**
 *
 * @author  John Whaley
 * @version 
 */
public interface ELFConstants {

    public static final byte ELFMAG0    = (byte)0x7f;
    public static final byte ELFMAG1    = (byte)'E';
    public static final byte ELFMAG2    = (byte)'L';
    public static final byte ELFMAG3    = (byte)'F';
    
    // ei_class
    public static final byte ELFCLASSNONE   = (byte)0;
    public static final byte ELFCLASS32     = (byte)1;
    public static final byte ELFCLASS64     = (byte)2;
    
    // ei_data
    public static final byte ELFDATANONE   = (byte)0;
    public static final byte ELFDATA2LSB   = (byte)1;
    public static final byte ELFDATA2MSB   = (byte)2;
    
    // e_type
    public static final int ET_NONE     = 0;
    public static final int ET_REL      = 1;
    public static final int ET_EXEC     = 2;
    public static final int ET_DYN      = 3;
    public static final int ET_CORE     = 4;
    public static final int ET_LOPROC   = 0xff00;
    public static final int ET_HIPROC   = 0xffff;

    // e_machine
    public static final int EM_M32      = 1;
    public static final int EM_SPARC    = 2;
    public static final int EM_386      = 3;
    public static final int EM_68K      = 4;
    public static final int EM_88K      = 5;
    public static final int EM_860      = 7;
    public static final int EM_MIPS     = 8;
    public static final int EM_MIPS_RS4_BE = 10;
    
    // e_version
    public static final int EV_NONE       = (byte)0;
    public static final int EV_CURRENT    = (byte)1;
    
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

    // Reloc Types
    public static final byte R_386_NONE = 0;
    public static final byte R_386_32   = 1;
    public static final byte R_386_PC32 = 2;
    
    // Special Section Indexes
    public static final int SHN_UNDEF       = 0;
    public static final int SHN_LORESERVE   = 0xff00;
    public static final int SHN_LOPROC      = 0xff00;
    public static final int SHN_HIPROC      = 0xff1f;
    public static final int SHN_ABS         = 0xfff1;
    public static final int SHN_COMMON      = 0xfff2;
    public static final int SHN_HIRESERVE   = 0xffff;
    public static final int SHN_INVALID     = -1;
    
    // Section Types.
    public static final int SHT_NULL        = 0;
    public static final int SHT_PROGBITS    = 1;
    public static final int SHT_SYMTAB      = 2;
    public static final int SHT_STRTAB      = 3;
    public static final int SHT_RELA        = 4;
    public static final int SHT_HASH        = 5;
    public static final int SHT_DYNAMIC     = 6;
    public static final int SHT_NOTE        = 7;
    public static final int SHT_NOBITS      = 8;
    public static final int SHT_REL         = 9;
    public static final int SHT_SHLIB       = 10;
    public static final int SHT_DYNSYM      = 11;
    public static final int SHT_LOPROC      = 0x70000000;
    public static final int SHT_HIPROC      = 0x7fffffff;
    public static final int SHT_LOUSER      = 0x80000000;
    public static final int SHT_HIUSER      = 0xffffffff;
    
    // Section Attribute Flags
    public static final int SHF_WRITE       = 0x1;
    public static final int SHF_ALLOC       = 0x2;
    public static final int SHF_EXECINSTR   = 0x4;
    public static final int SHF_MASKPROC    = 0xf0000000;
    
    // Symbol Binding
    public static final byte STB_LOCAL   = 0;
    public static final byte STB_GLOBAL  = 1;
    public static final byte STB_WEAK    = 2;
    public static final byte STB_LOPROC  = 13;
    public static final byte STB_HIPROC  = 15;
    
    // Symbol Types
    public static final byte STT_NOTYPE  = 0;
    public static final byte STT_OBJECT  = 1;
    public static final byte STT_FUNC    = 2;
    public static final byte STT_SECTION = 3;
    public static final byte STT_FILE    = 4;
    public static final byte STT_LOPROC  = 13;
    public static final byte STT_HIPROC  = 15;
    
}

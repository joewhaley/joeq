/*
 * Section.java
 *
 * Created on February 6, 2002, 4:00 PM
 */

package Linker.ELF;

import java.io.*;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.LinkedList;
import java.util.Iterator;

/**
 * Defines a section in an ELF file.
 *
 * @author  John Whaley
 * @version 
 */
public abstract class Section {
    
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
    
    // The index of this section in the section header table.
    protected int index;
    
    protected String name;
    protected int flags;
    protected int addr;
    protected int offset;

    public int getIndex() { return index; }
    public String getName() { return name; }
    public abstract int getType();
    public final int getFlags() { return flags; }
    public final int getAddr() { return addr; }
    public final int getOffset() { return offset; }
    public abstract int getSize();
    public abstract int getLink();
    public abstract int getInfo();
    public abstract int getAddrAlign();
    public abstract int getEntSize();

    public void setIndex(int index) { this.index = index; }
    public void setName(String name) { this.name = name; }
    public void setAddr(int addr) { this.addr = addr; }
    public void setOffset(int offset) { this.offset = offset; }
    public void setWrite() { this.flags |= SHF_WRITE; }
    public void clearWrite() { this.flags &= ~SHF_WRITE; }
    public void setAlloc() { this.flags |= SHF_ALLOC; }
    public void clearAlloc() { this.flags &= ~SHF_ALLOC; }
    public void setExecInstr() { this.flags |= SHF_EXECINSTR; }
    public void clearExecInstr() { this.flags &= ~SHF_EXECINSTR; }

    public void writeHeader(ELFFile file, OutputStream out) throws IOException {
        file.write_sectionname(out, this.getName());
        file.write_word(out, this.getType());
        file.write_word(out, this.getFlags());
        file.write_addr(out, this.getAddr());
        file.write_off(out, this.getOffset());
        file.write_word(out, this.getSize());
        file.write_word(out, this.getLink());
        file.write_word(out, this.getInfo());
        file.write_word(out, this.getAddrAlign());
        file.write_word(out, this.getEntSize());
    }
    
    public abstract void writeData(ELFFile file, OutputStream out) throws IOException;
    
    /** Creates new Section */
    protected Section(String name, int flags) {
        this.name = name; this.flags = flags;
        this.index = SHN_INVALID;
    }

    public static class NullSection extends Section {
        public NullSection() { super("", 0); }
        public int getIndex() { return 0; }
        public int getType() { return SHT_NULL; }
        public int getSize() { return 0; }
        public int getLink() { return SHN_UNDEF; }
        public int getInfo() { return 0; }
        public int getAddrAlign() { return 0; }
        public int getEntSize() { return 0; }
        public void setIndex(int index) { if (index != 0) throw new InternalError(); }
        public void setName(String name) { throw new InternalError(); }
        public void setAddr(int addr) { if (addr != 0) throw new InternalError(); }
        public void setOffset(int offset) { if (offset != 0) throw new InternalError(); }
        public void setWrite() { throw new InternalError(); }
        public void setAlloc() { throw new InternalError(); }
        public void setExecInstr() { throw new InternalError(); }
        public void writeData(ELFFile file, OutputStream out) throws IOException { }
    }
    
    public abstract static class ProgBitsSection extends Section {
        public ProgBitsSection(String name, int flags) {
            super(name, flags);
        }
        public final int getType() { return SHT_PROGBITS; }
        public final int getLink() { return SHN_UNDEF; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }
    }
    
    public static class SymTabSection extends Section {
        List/*<SymbolTableEntry>*/ symbols;
        protected int link, info;
        public SymTabSection(String name, int flags, int link, int info) {
            super(name, flags);
            this.link = link; this.info = info;
            this.symbols = new LinkedList();
        }
        public void addSymbol(SymbolTableEntry e) { symbols.add(e); }
        public int getSize() { return symbols.size() * SymbolTableEntry.getEntrySize(); }
        public int getAddrAlign() { return 0; }
        public final int getType() { return SHT_SYMTAB; }
        public final int getLink() { return link; }
        public final int getInfo() { return info; }
        public final int getEntSize() { return SymbolTableEntry.getEntrySize(); }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            Iterator i = symbols.iterator();
            while (i.hasNext()) {
                SymbolTableEntry e = (SymbolTableEntry)i.next();
                e.write(file, out);
            }
        }
    }
    
    public static class DynSymSection extends Section {
        List/*<SymbolTableEntry>*/ symbols;
        protected int link, info;
        public DynSymSection(String name, int flags, int link, int info) {
            super(name, flags);
            this.link = link; this.info = info;
            this.symbols = new LinkedList();
        }
        public void addSymbol(SymbolTableEntry e) { symbols.add(e); }
        public int getSize() { return symbols.size() * SymbolTableEntry.getEntrySize(); }
        public int getAddrAlign() { return 0; }
        public final int getType() { return SHT_DYNSYM; }
        public final int getLink() { return link; }
        public final int getInfo() { return info; }
        public final int getEntSize() { return SymbolTableEntry.getEntrySize(); }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            Iterator i = symbols.iterator();
            while (i.hasNext()) {
                SymbolTableEntry e = (SymbolTableEntry)i.next();
                e.write(file, out);
            }
        }
    }
    
    public static class StrTabSection extends Section {
        protected Map/*<String, Integer>*/ string_map;
        protected byte[] table;
        public StrTabSection(String name, int flags) {
            super(name, flags);
            string_map = new HashMap();
        }
        public final int getType() { return SHT_STRTAB; }
        public final int getLink() { return SHN_UNDEF; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }

        public void addString(String s) { string_map.put(s, null); }

        public void pack() {
            int size = 1;
            Iterator i = string_map.entrySet().iterator();
            while (i.hasNext()) {
                Map.Entry e = (Map.Entry)i.next();
                String s = (String)e.getKey();
                e.setValue(new Integer(size));
                size += s.length() + 1;
            }
            if (size == 1) size = 0;
            // todo: combine strings that have the same endings.
            table = new byte[size];
            int index = 0;
            i = string_map.entrySet().iterator();
            while (i.hasNext()) {
                Map.Entry e = (Map.Entry)i.next();
                String s = (String)e.getKey();
                s.getBytes(0, s.length(), table, index);
                index += s.length() + 1;
            }
        }
        
        public int getStringIndex(String s) {
            Integer i = (Integer)string_map.get(s);
            if (i == null)
                return 0;
            return i.intValue();
        }
        public int getSize() { return table.length; }
        public int getAddrAlign() { return 0; }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            out.write(table);
        }
    }
    
    public static class RelASection extends Section {
        protected List/*<RelocAEntry>*/ relocs;
        protected int symbolTableIndex;
        protected int sectionIndex;
        public RelASection(String name, int flags, int symbolTableIndex, int sectionIndex) {
            super(name, flags);
            this.symbolTableIndex = symbolTableIndex; this.sectionIndex = sectionIndex;
            this.relocs = new LinkedList();
        }
        public final int getType() { return SHT_RELA; }
        public final int getLink() { return symbolTableIndex; }
        public final int getInfo() { return sectionIndex; }
        public final int getEntSize() { return RelocAEntry.getEntrySize(); }
        public int getSize() { return relocs.size() * RelocAEntry.getEntrySize(); }
        public int getAddrAlign() { return 0; }
        public void addReloc(RelocAEntry e) { relocs.add(e); }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            Iterator i = relocs.iterator();
            while (i.hasNext()) {
                RelocAEntry e = (RelocAEntry)i.next();
                e.write(file, out);
            }
        }
    }
    
    public static class HashSection extends Section {
        protected int sectionIndex;
        public HashSection(String name, int flags, int sectionIndex) {
            super(name, flags);
            this.sectionIndex = sectionIndex;
        }
        public final int getType() { return SHT_HASH; }
        public final int getLink() { return sectionIndex; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }
        public int getSize() { return 0; } // WRITE ME
        public int getAddrAlign() { return 0; }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            // WRITE ME
        }
    }
    
    public static class DynamicSection extends Section {
        protected int stringTableIndex;
        public DynamicSection(String name, int flags, int stringTableIndex) {
            super(name, flags);
            this.stringTableIndex = stringTableIndex;
        }
        public final int getType() { return SHT_DYNAMIC; }
        public final int getLink() { return stringTableIndex; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }
        public int getSize() { return 0; } // WRITE ME
        public int getAddrAlign() { return 0; }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            // WRITE ME
        }
    }
    
    public static class NoteSection extends Section {
        protected String notename;
        protected byte[] notedesc;
        protected int notetype;
        public NoteSection(String sectionname, int flags, String notename, byte[] notedesc, int notetype) {
            super(sectionname, flags);
            this.notename = notename; this.notedesc = notedesc; this.notetype = notetype;
        }
        public final int getType() { return SHT_NOTE; }
        public final int getLink() { return SHN_UNDEF; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }
        protected int getNameLength() { return (notename.length()+4)&~3; }
        public int getSize() { return 12 + getNameLength() + notedesc.length; }
        public int getAddrAlign() { return 0; }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            file.write_word(out, getNameLength());
            file.write_word(out, notedesc.length);
            file.write_word(out, notetype);
            byte[] notename_b = new byte[getNameLength()];
            notename.getBytes(0, notename.length(), notename_b, 0);
            out.write(notename_b);
            out.write(notedesc);
        }
    }
    
    public static class NoBitsSection extends Section {
        protected int size; protected int addralign;
        public NoBitsSection(String name, int flags, int size, int addralign) {
            super(name, flags);
            this.size = size; this.addralign = addralign;
        }
        public final int getType() { return SHT_NOBITS; }
        public final int getLink() { return SHN_UNDEF; }
        public final int getInfo() { return 0; }
        public final int getEntSize() { return 0; }
        public int getSize() { return size; }
        public int getAddrAlign() { return addralign; }
        public void writeData(ELFFile file, OutputStream out) throws IOException { }
    }
    
    public static class RelSection extends Section {
        protected List/*<RelocEntry>*/ relocs;
        protected SymTabSection symbolTable;
        protected Section targetSection;
        public RelSection(String name, int flags, SymTabSection symbolTable, Section targetSection) {
            super(name, flags);
            this.symbolTable = symbolTable; this.targetSection = targetSection;
            this.relocs = new LinkedList();
        }
        public final int getType() { return SHT_REL; }
        public final int getLink() { return symbolTable.getIndex(); }
        public final int getInfo() { return targetSection.getIndex(); }
        public final int getEntSize() { return RelocEntry.getEntrySize(); }
        public int getSize() { return relocs.size() * RelocEntry.getEntrySize(); }
        public int getAddrAlign() { return 0; }
        public void addReloc(RelocEntry e) { relocs.add(e); }
        public void writeData(ELFFile file, OutputStream out) throws IOException {
            Iterator i = relocs.iterator();
            while (i.hasNext()) {
                RelocEntry e = (RelocEntry)i.next();
                e.write(file, out);
            }
        }
    }
    
    public static int getHeaderSize() { return 40; }
}

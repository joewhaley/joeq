/*
 * ELFFile.java
 *
 * Created on February 6, 2002, 8:00 PM
 */

package Linker.ELF;
import java.io.*;
import java.util.List;
import java.util.LinkedList;
import java.util.Iterator;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class ELFFile {

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

    protected byte ei_class;
    protected byte ei_data;
    protected int e_type;
    protected int e_machine;
    protected int e_version;
    protected int e_entry;
    protected int e_phoff;
    protected int e_shoff;
    protected int e_flags;
    
    protected List/*<ProgramHeader>*/ program_headers;
    protected List/*<Section>*/ sections;
    protected Section.StrTabSection section_header_string_table;
    protected Section.StrTabSection symbol_string_table;
    
    /** Creates new ELFFile */
    public ELFFile(byte data, int type, int machine, int entry) {
        ei_class = ELFCLASS32;
        ei_data = data; // ELFDATA2LSB
        e_type = type;
        e_machine = machine;
        e_version = EV_CURRENT;
        program_headers = new LinkedList();
        sections = new LinkedList();
    }

    public void setSectionHeaderStringTable(Section.StrTabSection shstrtab) {
        this.section_header_string_table = shstrtab;
    }
    
    public void setSymbolStringTable(Section.StrTabSection symstrtab) {
        this.symbol_string_table = symstrtab;
    }

    static int getSectionIndex(Section s) {
        if (s == null)
            return Section.SHN_UNDEF;
        return s.getIndex();
    }
    
    public void addSection(Section s) {
        sections.add(s);
    }
    public void removeSection(Section s) {
        sections.remove(s);
    }
    public void addProgramHeader(ProgramHeader p) {
        program_headers.add(p);
    }
    public void removeProgramHeader(ProgramHeader p) {
        program_headers.remove(p);
    }
    
    public void renumber() {
        // number all program headers, set program header offset.
        Iterator pi = program_headers.iterator();
        if (!pi.hasNext()) {
            e_phoff = 0;
            e_shoff = ELFFile.getHeaderSize();
        } else {
            e_phoff = ELFFile.getHeaderSize();
            int pindex = -1;
            while (pi.hasNext()) {
                ProgramHeader p = (ProgramHeader)pi.next();
                p.setIndex(++pindex);
            }
            e_shoff = e_phoff + ((pindex+1) * ProgramHeader.getSize());
        }
        // number all sections and calculate section header offset.
        // also add section header names to the section header string table.
        Iterator si = sections.iterator();
        int sindex = -1; 
        while (si.hasNext()) {
            Section s = (Section)si.next();
            s.setIndex(++sindex);
            s.setOffset(e_shoff);
            if (section_header_string_table != null)
                section_header_string_table.addString(s.getName());
            e_shoff += s.getSize();
        }
        // pack string tables.
        if (section_header_string_table != null)
            section_header_string_table.pack();
        if (symbol_string_table != null)
            symbol_string_table.pack();
    }
    
    public boolean isLittleEndian() { return ei_data == ELFDATA2LSB; }
    public boolean isBigEndian() { return ei_data == ELFDATA2MSB; }
    public void setLittleEndian() { ei_data = ELFDATA2LSB; }
    public void setBigEndian() { ei_data = ELFDATA2MSB; }
    
    public void write(OutputStream out) throws IOException {
        renumber();
        writeHeader(out);
        writeProgramHeaderTable(out);
        Iterator si = sections.iterator();
        while (si.hasNext()) {
            Section s = (Section)si.next();
            s.writeData(this, out);
        }
        writeSectionHeaderTable(out);
    }

    void writeProgramHeaderTable(OutputStream out) throws IOException {
        Iterator pi = program_headers.iterator();
        while (pi.hasNext()) {
            ProgramHeader p = (ProgramHeader)pi.next();
            p.writeHeader(this, out);
        }
    }
    
    void writeSectionHeaderTable(OutputStream out) throws IOException {
        Iterator si = sections.iterator();
        while (si.hasNext()) {
            Section s = (Section)si.next();
            s.writeHeader(this, out);
        }
    }
    
    void writeHeader(OutputStream out) throws IOException {
        writeIdent(out);
        write_half(out, e_type);
        write_half(out, e_machine);
        write_word(out, e_version);
        write_addr(out, e_entry);
        write_off(out, e_phoff);
        write_off(out, e_shoff);
        write_word(out, e_flags);
        write_half(out, getHeaderSize());
        write_half(out, ProgramHeader.getSize());
        write_half(out, program_headers.size());
        write_half(out, Section.getHeaderSize());
        write_half(out, sections.size());
        write_half(out, getSectionIndex(section_header_string_table));
    }
    
    void writeIdent(OutputStream out) throws IOException {
        writeMagicNumber(out);
        out.write(ei_class);
        out.write(ei_data);
        out.write((byte)e_version);
        for (int i=7; i<16; ++i)
            out.write((byte)0);
    }

    void writeMagicNumber(OutputStream out) throws IOException {
        out.write(ELFMAG0);
        out.write(ELFMAG1);
        out.write(ELFMAG2);
        out.write(ELFMAG3);
    }
    
    public static int getHeaderSize() { return 52; }
    
    void write_half(OutputStream out, int v) throws IOException {
        if (isLittleEndian()) {
            out.write((byte)v);
            out.write((byte)(v>>8));
        } else {
            out.write((byte)(v>>8));
            out.write((byte)v);
        }
    }
    
    void write_word(OutputStream out, int v) throws IOException {
        if (isLittleEndian()) {
            out.write((byte)v);
            out.write((byte)(v>>8));
            out.write((byte)(v>>16));
            out.write((byte)(v>>24));
        } else {
            out.write((byte)(v>>24));
            out.write((byte)(v>>16));
            out.write((byte)(v>>8));
            out.write((byte)v);
        }
    }
    
    void write_sword(OutputStream out, int v) throws IOException {
        write_word(out, v);
    }
    
    void write_off(OutputStream out, int v) throws IOException {
        write_word(out, v);
    }
    
    void write_addr(OutputStream out, int v) throws IOException {
        write_word(out, v);
    }
    
    void write_sectionname(OutputStream out, String s) throws IOException {
        int value;
        if (section_header_string_table == null)
            value = 0;
        else
            value = section_header_string_table.getStringIndex(s);
        write_word(out, value);
    }
    
    void write_symbolname(OutputStream out, String s) throws IOException {
        int value;
        if (symbol_string_table == null)
            value = 0;
        else
            value = symbol_string_table.getStringIndex(s);
        write_word(out, value);
    }
}

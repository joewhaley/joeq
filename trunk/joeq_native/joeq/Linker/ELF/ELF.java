/*
 * ELF.java
 *
 * Created on May 21, 2002, 3:10 AM
 */

package Linker.ELF;

import java.io.*;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Iterator;
import Main.jq;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ELF implements ELFConstants {

    protected byte ei_class;
    protected byte ei_data;
    protected int e_type;
    protected int e_machine;
    protected int e_version;
    protected int e_entry;
    //protected int e_phoff;
    //protected int e_shoff;
    protected int e_flags;
    
    protected List/*<ProgramHeader>*/ program_headers;
    protected List/*<Section>*/ sections;
    protected Section.StrTabSection section_header_string_table;
    
    /** Creates new ELFFile */
    public ELF(byte data, int type, int machine, int entry) {
        ei_class = ELFCLASS32;
        ei_data = data; // ELFDATA2LSB
        e_type = type;
        e_machine = machine;
        e_version = EV_CURRENT;
        program_headers = new LinkedList();
        sections = new LinkedList();
    }

    public Section.StrTabSection getSectionHeaderStringTable() {
        return section_header_string_table;
    }
    
    public void setSectionHeaderStringTable(Section.StrTabSection shstrtab) {
        this.section_header_string_table = shstrtab;
    }
    
    int getSectionIndex(Section s) {
        if (s == null)
            return SHN_UNDEF;
        return s.getIndex();
    }
    
    Section getSection(int i) {
        if (i == SHN_ABS) return Section.AbsSection.INSTANCE;
        return (Section)sections.get(i);
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
    
    public boolean isLittleEndian() { return ei_data == ELFDATA2LSB; }
    public boolean isBigEndian() { return ei_data == ELFDATA2MSB; }
    public void setLittleEndian() { ei_data = ELFDATA2LSB; }
    public void setBigEndian() { ei_data = ELFDATA2MSB; }
    
    public void write() throws IOException {
        // sanity check - sections should include the string tables.
        if (section_header_string_table != null)
            jq.Assert(sections.contains(section_header_string_table));
        
        // add section header names to the section header string table.
        if (section_header_string_table != null) {
            Iterator si = sections.iterator();
            while (si.hasNext()) {
                Section s = (Section)si.next();
                section_header_string_table.addString(s.getName());
            }
        }
        
        // file offsets for the program header table and the section table.
        int e_phoff, e_shoff, soff;
        
        // calculate program header table offset.
        if (program_headers.isEmpty()) {
            e_phoff = 0;
            soff = e_shoff = ELF.getHeaderSize();
        } else {
            e_phoff = ELF.getHeaderSize();
            soff = e_shoff = e_phoff + (program_headers.size() * ProgramHeader.getSize());
        }
        
        // pack all sections and calculate section header offset.
        Iterator si = sections.iterator();
        Section s = (Section)si.next();
        jq.Assert(s instanceof Section.NullSection);
        int i = 0;
        while (si.hasNext()) {
            s = (Section)si.next();
            if (s instanceof Section.StrTabSection) {
                Section.StrTabSection ss = (Section.StrTabSection)s;
                if (ss.getNumberOfEntries() < 10000)
                    ss.super_pack();
                else
                    ss.pack();
            } else if (s instanceof Section.SymTabSection) {
                Section.SymTabSection ss = (Section.SymTabSection)s;
                ss.setIndices();
            }
            if (!(s instanceof Section.NoBitsSection))
                e_shoff += s.getSize();
            s.setIndex(++i);
        }
        
        // now, actually do the writing.
        // write the header.
        writeHeader(e_phoff, e_shoff);
        
        // write the program header table
        Iterator pi = program_headers.iterator();
        while (pi.hasNext()) {
            ProgramHeader p = (ProgramHeader)pi.next();
            p.writeHeader(this);
        }
        
        // write the section data
        si = sections.iterator();
        while (si.hasNext()) {
            s = (Section)si.next();
            s.writeData(this);
        }
        
        // write the section header table
        si = sections.iterator();
        while (si.hasNext()) {
            s = (Section)si.next();
            s.writeHeader(this, soff);
            if (!(s instanceof Section.NoBitsSection))
                soff += s.getSize();
        }
    }

    void writeHeader(int e_phoff, int e_shoff) throws IOException {
        writeIdent();
        write_half(e_type);
        write_half(e_machine);
        write_word(e_version);
        write_addr(e_entry);
        write_off(e_phoff);
        write_off(e_shoff);
        write_word(e_flags);
        write_half(getHeaderSize());
        write_half(ProgramHeader.getSize());
        write_half(program_headers.size());
        write_half(Section.getHeaderSize());
        write_half(sections.size());
        write_half(getSectionIndex(section_header_string_table));
    }
    
    void writeIdent() throws IOException {
        writeMagicNumber();
        write_byte(ei_class);
        write_byte(ei_data);
        write_byte((byte)e_version);
        for (int i=7; i<16; ++i)
            write_byte((byte)0);
    }

    void writeMagicNumber() throws IOException {
        write_byte(ELFMAG0);
        write_byte(ELFMAG1);
        write_byte(ELFMAG2);
        write_byte(ELFMAG3);
    }
    
    public static int getHeaderSize() { return 52; }
    
    abstract void write_byte(byte v) throws IOException;
    abstract void write_bytes(byte[] v) throws IOException;
    abstract void write_half(int v) throws IOException;
    abstract void write_word(int v) throws IOException;
    abstract void write_sword(int v) throws IOException;
    abstract void write_off(int v) throws IOException;
    abstract void write_addr(int v) throws IOException;
    abstract void write_sectionname(String s) throws IOException;
    
    abstract void set_position(int offset) throws IOException;
    abstract byte read_byte() throws IOException;
    abstract void read_bytes(byte[] b) throws IOException;
    abstract int read_half() throws IOException;
    abstract int read_word() throws IOException;
    abstract int read_sword() throws IOException;
    abstract int read_off() throws IOException;
    abstract int read_addr() throws IOException;
}

/*
 * ELFRandomAccessFile.java
 *
 * Created on February 6, 2002, 8:00 PM
 */

package Linker.ELF;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class ELFRandomAccessFile extends ELF {
    
    protected RandomAccessFile file;
    protected List section_headers;
    public ELFRandomAccessFile(byte data, int type, int machine, int entry, RandomAccessFile file) {
        super(data, type, machine, entry);
        this.file = file;
    }
    
    void readHeader() throws IOException {
        
        byte mag0 = read_byte();
        if (mag0 != ELFMAG0) throw new IOException();
        byte mag1 = read_byte();
        if (mag1 != ELFMAG1) throw new IOException();
        byte mag2 = read_byte();
        if (mag2 != ELFMAG2) throw new IOException();
        byte mag3 = read_byte();
        if (mag3 != ELFMAG3) throw new IOException();
        this.ei_class = read_byte();
        this.ei_data = read_byte();
        byte e_version2 = read_byte();
        for (int i=7; i<16; ++i) {
            byte b = read_byte();
            if (b != 0) throw new IOException();
        }
        this.e_type = read_half();
        this.e_machine = read_half();
        this.e_version = read_word();
        if (e_version2 != (byte)e_version) throw new IOException();
        this.e_entry = read_addr();
        int e_phoff = read_off();
        int e_shoff = read_off();
        this.e_flags = read_word();
        int headersize = read_half();
        if (headersize != ELF.getHeaderSize()) throw new IOException();
        int programheadersize = read_half();
        if (programheadersize != ProgramHeader.getSize()) throw new IOException();
        int n_programheaders = read_half();
        int sectionheadersize = read_half();
        if (sectionheadersize != Section.getHeaderSize()) throw new IOException();
        int n_sectionheaders = read_half();
        int section_header_string_table_index = read_half();
        
        // read and parse section headers
        this.set_position(e_shoff);
        section_headers = new ArrayList(n_sectionheaders);
        for (int i=0; i<n_sectionheaders; ++i) {
            Section.UnloadedSection us = new Section.UnloadedSection(this);
            section_headers.add(us);
            Section ss = us.parseHeader();
            sections.add(ss);
        }
        
        // read section header string table
        if (section_header_string_table_index != 0) {
            this.section_header_string_table = (Section.StrTabSection)sections.get(section_header_string_table_index);
            Section.UnloadedSection us = (Section.UnloadedSection)section_headers.get(section_header_string_table_index);
            section_headers.set(section_header_string_table_index, null);
            this.section_header_string_table.load(us, this);
        }
    }
    
    Section getSection(int i) {
        if (i == SHN_ABS) return Section.AbsSection.INSTANCE;
        Section s = (Section)sections.get(i);
        Section.UnloadedSection us = (Section.UnloadedSection)section_headers.get(i);
        if (us != null) {
            section_headers.set(i, null);
            try {
                s.load(us, this);
            } catch (IOException x) {
                x.printStackTrace();
            }
        }
        return s;
    }
    
    void write_byte(byte v) throws IOException {
        file.write(v);
    }
    
    void write_bytes(byte[] v) throws IOException {
        file.write(v);
    }
    
    void write_half(int v) throws IOException {
        if (isLittleEndian()) {
            file.write((byte)v);
            file.write((byte)(v>>8));
        } else {
            file.write((byte)(v>>8));
            file.write((byte)v);
        }
    }
    
    void write_word(int v) throws IOException {
        if (isLittleEndian()) {
            file.write((byte)v);
            file.write((byte)(v>>8));
            file.write((byte)(v>>16));
            file.write((byte)(v>>24));
        } else {
            file.write((byte)(v>>24));
            file.write((byte)(v>>16));
            file.write((byte)(v>>8));
            file.write((byte)v);
        }
    }
    
    void write_sword(int v) throws IOException {
        write_word(v);
    }
    
    void write_off(int v) throws IOException {
        write_word(v);
    }
    
    void write_addr(int v) throws IOException {
        write_word(v);
    }
    
    void write_sectionname(String s) throws IOException {
        int value;
        if (section_header_string_table == null)
            value = 0;
        else
            value = section_header_string_table.getStringIndex(s);
        write_word(value);
    }
    
    void set_position(int offset) throws IOException {
        file.seek(offset);
    }
    
    byte read_byte() throws IOException {
        return file.readByte();
    }
    
    void read_bytes(byte[] b) throws IOException {
        file.readFully(b);
    }
    
    int read_half() throws IOException {
        int b1 = file.readByte() & 0xFF;
        int b2 = file.readByte() & 0xFF;
        int r;
        if (isLittleEndian()) {
            r = (b2 << 8) | b1;
        } else {
            r = (b1 << 8) | b2;
        }
        return r;
    }
    
    int read_word() throws IOException {
        int b1 = file.readByte() & 0xFF;
        int b2 = file.readByte() & 0xFF;
        int b3 = file.readByte() & 0xFF;
        int b4 = file.readByte() & 0xFF;
        int r;
        if (isLittleEndian()) {
            r = (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
        } else {
            r = (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
        }
        return r;
    }
    
    int read_sword() throws IOException {
        return read_word();
    }
    
    int read_off() throws IOException {
        return read_word();
    }
    
    int read_addr() throws IOException {
        return read_word();
    }
}

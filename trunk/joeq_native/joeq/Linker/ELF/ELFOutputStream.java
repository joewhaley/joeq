/*
 * ELFOutputStream.java
 *
 * Created on February 6, 2002, 8:00 PM
 */

package Linker.ELF;
import java.io.IOException;
import java.io.OutputStream;

/**
 *
 * @author  John Whaley
 * @version $Id$ 
 */
public class ELFOutputStream extends ELF {
    
    protected OutputStream out;
    public ELFOutputStream(byte data, int type, int machine, int entry, OutputStream out) {
        super(data, type, machine, entry);
        this.out = out;
    }
    
    public OutputStream getOutputStream() { return out; }
    
    void write_byte(byte v) throws IOException {
        out.write(v);
    }
    
    void write_bytes(byte[] v) throws IOException {
        out.write(v);
    }
    
    void write_half(int v) throws IOException {
        if (isLittleEndian()) {
            out.write((byte)v);
            out.write((byte)(v>>8));
        } else {
            out.write((byte)(v>>8));
            out.write((byte)v);
        }
    }
    
    void write_word(int v) throws IOException {
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
    
    void set_position(int offset) throws IOException { throw new IOException(); }
    byte read_byte() throws IOException { throw new IOException(); }
    void read_bytes(byte[] b) throws IOException { throw new IOException(); }
    int read_half() throws IOException { throw new IOException(); }
    int read_word() throws IOException { throw new IOException(); }
    int read_sword() throws IOException { throw new IOException(); }
    int read_off() throws IOException { throw new IOException(); }
    int read_addr() throws IOException { throw new IOException(); }
}

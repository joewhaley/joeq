/*
 * ELFOutput.java
 *
 * Created on February 6, 2002, 8:00 PM
 */

package Linker.ELF;
import java.io.DataOutput;
import java.io.IOException;

/**
 *
 * @author  John Whaley
 * @version $Id$ 
 */
public class ELFOutput extends ELFImpl {
    
    protected DataOutput out;
    public ELFOutput(byte data, int type, int machine, int entry, DataOutput out) {
        super(data, type, machine, entry);
        this.out = out;
    }
    
    public DataOutput getOutput() { return out; }
    
    public void write_byte(byte v) throws IOException {
        out.write(v);
    }
    
    public void write_bytes(byte[] v) throws IOException {
        out.write(v);
    }
    
    public void write_half(int v) throws IOException {
        out.writeShort((short)v);
    }
    
    public void write_word(int v) throws IOException {
        out.writeInt(v);
    }
    
    public void write_sword(int v) throws IOException {
        out.writeInt(v);
    }
    
    public void write_off(int v) throws IOException {
        out.writeInt(v);
    }
    
    public void write_addr(int v) throws IOException {
        out.writeInt(v);
    }
    
    public void write_sectionname(String s) throws IOException {
        int value;
        if (section_header_string_table == null)
            value = 0;
        else
            value = section_header_string_table.getStringIndex(s);
        write_word(value);
    }
    
    public void set_position(int offset) throws IOException { throw new IOException(); }
    public byte read_byte() throws IOException { throw new IOException(); }
    public void read_bytes(byte[] b) throws IOException { throw new IOException(); }
    public int read_half() throws IOException { throw new IOException(); }
    public int read_word() throws IOException { throw new IOException(); }
    public int read_sword() throws IOException { throw new IOException(); }
    public int read_off() throws IOException { throw new IOException(); }
    public int read_addr() throws IOException { throw new IOException(); }
}

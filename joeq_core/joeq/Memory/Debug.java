package Memory;

import Main.jq;
import Run_Time.SystemInterface;

/**
 * @author John Whaley
 */
public abstract class Debug {

    private static byte[] buffer = new byte[16];
    private static int bufferIndex;

    private static void writeDecimalToBuffer(int i) {
        boolean nonzero_found = false;
        bufferIndex = -1;
        if (i < 0) {
            i = -i;
            buffer[++bufferIndex] = (byte)'-';
        }
        for (int j=1000000000; j > 1; j /= 10) {
            int k = i / j;
            i = i % j;
            if (nonzero_found || k != 0) {
                buffer[++bufferIndex] = (byte)(k + '0');
                nonzero_found = true;
            }
        }
        buffer[++bufferIndex] = (byte)(i + '0');
        buffer[++bufferIndex] = (byte)0;
    }

    private static void writeHexToBuffer(int i) {
        bufferIndex = -1;
        buffer[++bufferIndex] = (byte)'0';
        buffer[++bufferIndex] = (byte)'x';
        for (int j=0; j < 8; ++j) {
            int v = (i & 0xF0000000) >>> 28;
            buffer[++bufferIndex] = (v < 0xa)?((byte)(v+'0')):((byte)(v+'a'-0xa));
            i <<= 4;
        }
        buffer[++bufferIndex] = (byte)0;
        jq.Assert(bufferIndex == 10);
    }

    public static void write(String s) {
        SystemInterface.debugwrite(s);
    }
    
    public static void write(int x) {
        writeDecimalToBuffer(x);
        SystemInterface.debugwrite(buffer, bufferIndex);
    }

    public static void writeHex(int x) {
        writeHexToBuffer(x);
        SystemInterface.debugwrite(buffer, bufferIndex);
    }
    
    public static void write(Address x) {
        writeHex(x.to32BitValue());
    }
    
    public static void write(int x, String s) {
        write(x); write(s);
    }

    public static void write(String s, int x) {
        write(s); write(x);
    }
    
    public static void write(String s, Address x) {
        write(s); write(x);
    }
    
    public static void write(String s1, int x, String s2) {
        write(s1); write(x); write(s2);
    }
    
    public static void write(int x1, String s, int x2) {
        write(x1); write(s); write(x2);
    }
    
    public static void writeln() {
        writeln("");
    }
    
    public static void writeln(String s) {
        SystemInterface.debugwriteln(s);
    }
    
    public static void writeln(int x) {
        writeDecimalToBuffer(x);
        SystemInterface.debugwriteln(buffer, bufferIndex);
    }
    
    public static void writelnHex(int x) {
        writeHexToBuffer(x);
        SystemInterface.debugwriteln(buffer, bufferIndex);
    }
    
    public static void writeln(Address x) {
        writelnHex(x.to32BitValue());
    }
    
    public static void writeln(int x, String s) {
        write(x); writeln(s);
    }
    
    public static void writeln(String s, int x) {
        write(s); writeln(x);
    }
    
    public static void writeln(String s, Address x) {
        write(s); writeln(x);
    }
    
    public static void writeln(String s1, int x, String s2) {
        write(s1); write(x); writeln(s2);
    }
}

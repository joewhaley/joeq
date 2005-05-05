// DirectBufferedFileOutputStream.java, created Wed Mar  5  0:26:34 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UTFDataFormatException;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import jwutil.strings.UTFDataFormatError;
import jwutil.strings.Utf8;
import jwutil.util.Assert;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class DirectBufferedFileOutputStream extends OutputStream implements ExtendedDataOutput {

    public static final int DEFAULT_INITIAL_SIZE = 65536;

    private ByteBuffer directByteBuffer;
    private final FileChannel fileChannel;
    private final FileOutputStream out;
    
    /** Creates new DirectBufferedFileOutputStream */
    public DirectBufferedFileOutputStream(FileOutputStream out, int size) {
        this.directByteBuffer = ByteBuffer.allocateDirect(size);
        FileChannel fc = null;
        try {
            fc = MyFileChannelImpl.getFileChannel(out);
        } catch (IOException x) {
            Assert.UNREACHABLE(x.toString());
        }
        this.fileChannel = fc;
        this.out = out;
    }

    /** Creates new DirectBufferedFileOutputStream */
    public DirectBufferedFileOutputStream(FileOutputStream out) {
        this(out, DEFAULT_INITIAL_SIZE);
    }
    
    public final ByteOrder order() {
        return directByteBuffer.order();
    }
    
    public final void order(ByteOrder bo) {
        this.directByteBuffer = directByteBuffer.order(bo);
    }
    
    static String dumpBufferInfo(ByteBuffer p) {
        return "pos="+p.position()+",limit="+p.limit()+",cap="+p.capacity();
    }

    int totalBytesWritten;

    private void dumpBuffer() throws IOException {
        directByteBuffer.flip();
        if (true) {
            while (directByteBuffer.hasRemaining()) {
                int v = fileChannel.write(directByteBuffer);
                totalBytesWritten += v;
            }
            directByteBuffer.clear();
        } else {
            int v = fileChannel.write(directByteBuffer);
            totalBytesWritten += v;
            directByteBuffer.compact();
        }
    }

    public void write(byte[] p1) throws IOException {
        try {
            directByteBuffer.put(p1);
            return;
        } catch (BufferOverflowException x) {
            dumpBuffer();
            out.write(p1);
            totalBytesWritten += p1.length;
        }
    }
    public void write(byte[] p1, int p2, int p3) throws IOException {
        try {
            directByteBuffer.put(p1, p2, p3);
            return;
        } catch (BufferOverflowException x) {
            dumpBuffer();
            out.write(p1, p2, p3);
            totalBytesWritten += p3;
        }
    }
    public void write(int p1) throws IOException {
        this.writeByte(p1);
    }
    public void writeBoolean(boolean p1) throws IOException {
        writeByte(p1?1:0);
    }
    public void writeByte(int p1) throws IOException {
        for (;;) {
            try {
                directByteBuffer.put((byte)p1);
                return;
            } catch (BufferOverflowException x) {
                dumpBuffer();
            }
        }
    }
    public void writeShort(int p1) throws IOException {
        for (;;) {
            try {
                directByteBuffer.putShort((short)p1);
                return;
            } catch (BufferOverflowException x) {
                dumpBuffer();
            }
        }
    }
    public void writeChar(int p1) throws IOException {
        for (;;) {
            try {
                directByteBuffer.putChar((char)p1);
                return;
            } catch (BufferOverflowException x) {
                dumpBuffer();
            }
        }
    }
    public void writeInt(int p1) throws IOException {
        for (;;) {
            try {
                directByteBuffer.putInt(p1);
                return;
            } catch (BufferOverflowException x) {
                dumpBuffer();
            }
        }
    }
    public void writeLong(long p1) throws IOException {
        for (;;) {
            try {
                directByteBuffer.putLong(p1);
                return;
            } catch (BufferOverflowException x) {
                dumpBuffer();
            }
        }
    }
    public void writeFloat(float p1) throws IOException {
        writeInt(Float.floatToRawIntBits(p1));
    }
    public void writeDouble(double p1) throws IOException {
        writeLong(Double.doubleToRawLongBits(p1));
    }
    public void writeBytes(String p1) throws IOException {
        for (int i=0; i<p1.length(); ++i)
            writeByte(p1.charAt(i));
    }
    public void writeChars(java.lang.String p1) throws IOException {
        for (int i=0; i<p1.length(); ++i)
            writeChar(p1.charAt(i));
    }
    public void writeUTF(String p1) throws IOException {
        try {
            byte[] b = Utf8.toUtf8(p1);
            if (b.length > 65535)
                throw new UTFDataFormatException(b.length + " > 65535");
            writeShort(b.length);
            write(b);
        } catch (UTFDataFormatError x) {
            throw new UTFDataFormatException(x.getMessage());
        }
    }
    public void flush() throws IOException {
        dumpBuffer();
        out.flush();
    }
    public void close() throws IOException {
        this.flush();
        out.close();
    }
    
    public void writeUByte(int p1) throws IOException {
        writeByte(p1);
    }
    public void writeUShort(int p1) throws IOException {
        writeChar(p1);
    }
    public void writeUInt(int p1) throws IOException {
        writeInt(p1);
    }
    public void writeULong(long p1) throws IOException {
        writeLong(p1);
    }

    public int size() {
        return totalBytesWritten;
    }

    protected void finalize() throws Throwable {
        super.finalize();
        this.close();
    }

}

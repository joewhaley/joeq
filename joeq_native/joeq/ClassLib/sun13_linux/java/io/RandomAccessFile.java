/*
 * RandomAccessFile.java
 *
 * Created on February 26, 2001, 12:01 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.io;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import jq;

public abstract class RandomAccessFile {

    // question: should this be private?
    public static void open(java.io.RandomAccessFile dis, java.lang.String name, boolean writeable)
    throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int flags = writeable?(SystemInterface._O_RDWR | SystemInterface._O_CREAT | SystemInterface._O_BINARY)
                             :(SystemInterface._O_RDONLY | SystemInterface._O_BINARY);
        int fdnum = SystemInterface.file_open(filename, flags, 0);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        Reflection.putfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd, fdnum);
    }
    public static int read(java.io.RandomAccessFile dis) throws java.io.IOException {
        byte[] b = new byte[1];
        int v = readBytes(dis, b, 0, 1);
        if (v == -1) return -1;
        else if (v != 1) throw new java.io.IOException();
        return b[0]&0xFF;
    }
    private static int readBytes(java.io.RandomAccessFile dis, byte b[], int off, int len) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        // check for index out of bounds/null pointer
        if (len < 0) throw new IndexOutOfBoundsException();
        byte b2 = b[off+len-1];
        // BUG in Sun's implementation, which we mimic here.  off=b.length and len=0 doesn't throw an error (?)
        if (off < 0) throw new IndexOutOfBoundsException();
        if (len == 0) return 0;
        int start = Unsafe.addressOf(b)+off;
        int result = SystemInterface.file_readbytes(fdnum, start, len);
        if (result == 0)
            return -1; // EOF
        if (result == -1)
            throw new java.io.IOException();
        return result;
    }
    public static void write(java.io.RandomAccessFile dis, int b) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        int result = SystemInterface.file_writebyte(fdnum, b);
        if (result != 1)
            throw new java.io.IOException();
    }
    private static void writeBytes(java.io.RandomAccessFile dis, byte b[], int off, int len) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        // check for index out of bounds/null pointer
        if (len < 0) throw new IndexOutOfBoundsException();
        byte b2 = b[off+len-1];
        // BUG in Sun's implementation, which we mimic here.  off=b.length and len=0 doesn't throw an error (?)
        if (off < 0) throw new IndexOutOfBoundsException();
        if (len == 0) return;
        int start = Unsafe.addressOf(b)+off;
        int result = SystemInterface.file_writebytes(fdnum, start, len);
        if (result != len)
            throw new java.io.IOException();
    }
    public static long getFilePointer(java.io.RandomAccessFile dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        long curpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_CUR);
        if (curpos == (long)-1)
            throw new java.io.IOException();
        return curpos;
    }
    public static void seek(java.io.RandomAccessFile dis, long pos) throws java.io.IOException {
        if (pos < 0L)
            throw new java.io.IOException(pos+" < 0");
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        long result = SystemInterface.file_seek(fdnum, pos, SystemInterface.SEEK_SET);
        if (result == (long)-1)
            throw new java.io.IOException();
    }
    public static long length(java.io.RandomAccessFile dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        long curpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_CUR);
        if (curpos == (long)-1)
            throw new java.io.IOException();
        long endpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_END);
        if (endpos == (long)-1)
            throw new java.io.IOException();
        long result = SystemInterface.file_seek(fdnum, curpos, SystemInterface.SEEK_SET);
        if (result == (long)-1)
            throw new java.io.IOException();
        return endpos;
    }
    public static void setLength(java.io.RandomAccessFile dis, long newLength) throws java.io.IOException {
        jq.TODO();
    }
    public static void close(java.io.RandomAccessFile dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        int result = SystemInterface.file_close(fdnum);
        // Sun's "implementation" ignores errors on file close, allowing files to be closed multiple times.
        //if (result != 0)
        //    throw new java.io.IOException();
    }
    private static void initIDs(jq_Class clazz) { }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/RandomAccessFile;");
    public static final jq_InstanceMethod _open = _class.getOrCreateInstanceMethod("open", "(Ljava/lang/String;Z)V");
    public static final jq_InstanceField _fd = _class.getOrCreateInstanceField("fd", "Ljava/io/FileDescriptor;");
}

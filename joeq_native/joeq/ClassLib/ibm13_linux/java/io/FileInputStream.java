/*
 * FileInputStream.java
 *
 * Created on January 29, 2001, 1:37 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.io;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import jq;

abstract class FileInputStream {
    
    private static void open(java.io.FileInputStream dis, String name)
    throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int fdnum = SystemInterface.file_open(filename, SystemInterface._O_RDONLY | SystemInterface._O_BINARY, 0);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        Reflection.putfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd, fdnum);
    }
    public static int read(java.io.FileInputStream dis) throws java.io.IOException {
        byte[] b = new byte[1];
        int v = readBytes(dis, b, 0, 1);
        if (v == -1) return -1;
        else if (v != 1) throw new java.io.IOException();
        return b[0]&0xFF;
    }
    private static int readBytes(java.io.FileInputStream dis, byte b[], int off, int len) throws java.io.IOException {
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
    public static long skip(java.io.FileInputStream dis, long n) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        long curpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_CUR);
        long result = SystemInterface.file_seek(fdnum, n, SystemInterface.SEEK_CUR);
        if (result == (long)-1)
            throw new java.io.IOException();
        return result-curpos;
    }
    public static int available(java.io.FileInputStream dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        if (fdnum == 0) { // stdin
            int result = SystemInterface.console_available();
            if (result == -1) throw new java.io.IOException();
            return result;
        } else {
            long curpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_CUR);
            if (curpos == (long)-1)
                throw new java.io.IOException();
            long endpos = SystemInterface.file_seek(fdnum, 0, SystemInterface.SEEK_END);
            if (endpos == (long)-1)
                throw new java.io.IOException();
            long result = SystemInterface.file_seek(fdnum, curpos, SystemInterface.SEEK_SET);
            if (result == (long)-1)
                throw new java.io.IOException();
            if (endpos-curpos > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            else return (int)(endpos-curpos);
        }
    }
    public static void close(java.io.FileInputStream dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        int result = SystemInterface.file_close(fdnum);
        // Sun's "implementation" ignores errors on file close, allowing files to be closed multiple times.
        //if (result != 0)
        //    throw new java.io.IOException();
    }
    private static void initIDs(jq_Class clazz) { }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileInputStream;");
    public static final jq_InstanceField _fd = _class.getOrCreateInstanceField("fd", "Ljava/io/FileDescriptor;");
}

/*
 * FileOutputStream.java
 *
 * Created on January 29, 2001, 2:22 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.io;

import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Bootstrap.PrimordialClassLoader;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import jq;

abstract class FileOutputStream {

    private static void open(java.io.FileOutputStream dis, String name) throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int fdnum = SystemInterface.file_open(filename, SystemInterface._O_WRONLY | SystemInterface._O_BINARY | SystemInterface._O_CREAT | SystemInterface._O_TRUNC, SystemInterface._S_IREAD | SystemInterface._S_IWRITE);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        Reflection.putfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd, fdnum);
    }
    private static void openAppend(java.io.FileOutputStream dis, String name) throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int fdnum = SystemInterface.file_open(filename, SystemInterface._O_WRONLY | SystemInterface._O_BINARY | SystemInterface._O_CREAT | SystemInterface._O_APPEND, SystemInterface._S_IREAD | SystemInterface._S_IWRITE);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        Reflection.putfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd, fdnum);
    }
    public static void write(java.io.FileOutputStream dis, int b) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        int result = SystemInterface.file_writebyte(fdnum, b);
        if (result != 1)
            throw new java.io.IOException();
    }
    private static void writeBytes(java.io.FileOutputStream dis, byte b[], int off, int len) throws java.io.IOException {
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
    public static void close(java.io.FileOutputStream dis) throws java.io.IOException {
        int fdnum = Reflection.getfield_I(Reflection.getfield_A(dis, _fd), FileDescriptor._fd);
        int result = SystemInterface.file_close(fdnum);
        // Sun's "implementation" ignores errors on file close, allowing files to be closed multiple times.
        //if (result != 0)
        //    throw new java.io.IOException();
    }
    private static void initIDs(jq_Class clazz) {}

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileOutputStream;");
    public static final jq_InstanceField _fd = _class.getOrCreateInstanceField("fd", "Ljava/io/FileDescriptor;");
}

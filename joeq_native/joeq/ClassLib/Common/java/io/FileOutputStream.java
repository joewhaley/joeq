/*
 * FileOutputStream.java
 *
 * Created on January 29, 2001, 2:22 PM
 *
 */

package ClassLib.Common.java.io;

import Memory.HeapAddress;
import Run_Time.SystemInterface;

/*
 * @author  John Whaley
 * @version $Id$
 */
abstract class FileOutputStream {

    private FileDescriptor fd;
    
    private void open(String name) throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int fdnum = SystemInterface.file_open(filename, SystemInterface._O_WRONLY | SystemInterface._O_BINARY | SystemInterface._O_CREAT | SystemInterface._O_TRUNC, SystemInterface._S_IREAD | SystemInterface._S_IWRITE);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        this.fd.fd = fdnum;
    }
    private void openAppend(String name) throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
        int fdnum = SystemInterface.file_open(filename, SystemInterface._O_WRONLY | SystemInterface._O_BINARY | SystemInterface._O_CREAT | SystemInterface._O_APPEND, SystemInterface._S_IREAD | SystemInterface._S_IWRITE);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        this.fd.fd = fdnum;
    }
    public void write(int b) throws java.io.IOException {
        int fdnum = this.fd.fd;
        int result = SystemInterface.file_writebyte(fdnum, b);
        if (result != 1)
            throw new java.io.IOException();
    }
    private void writeBytes(byte b[], int off, int len) throws java.io.IOException {
        writeBytes(b, off, len, this.fd);
    }
    // IBM JDK has this extra fd argument here.
    private void writeBytes(byte b[], int off, int len, FileDescriptor fd) throws java.io.IOException {
        int fdnum = fd.fd;
        // check for index out of bounds/null pointer
        if (len < 0) throw new IndexOutOfBoundsException();
        byte b2 = b[off+len-1];
        // BUG in Sun's implementation, which we mimic here.  off=b.length and len=0 doesn't throw an error (?)
        if (off < 0) throw new IndexOutOfBoundsException();
        if (len == 0) return;
        HeapAddress start = (HeapAddress) HeapAddress.addressOf(b).offset(off);
        int result = SystemInterface.file_writebytes(fdnum, start, len);
        if (result != len)
            throw new java.io.IOException();
    }
    public void close() throws java.io.IOException {
        int fdnum = this.fd.fd;
        int result = SystemInterface.file_close(fdnum);
        // Sun's "implementation" ignores errors on file close, allowing files to be closed multiple times.
        //if (result != 0)
        //    throw new java.io.IOException();
    }
    private static void initIDs() {}

}

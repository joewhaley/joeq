/*
 * Win32FileSystem.java
 *
 * Created on January 29, 2001, 2:29 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun14_win32.java.io;

import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import jq;

public abstract class Win32FileSystem {

    public native java.lang.String canonicalize(java.lang.String s) throws java.io.IOException;
    public native int getBooleanAttributes(java.io.File f);
    public native boolean checkAccess(java.io.File f, boolean b);
    public native long getLastModifiedTime(java.io.File f);
    public native long getLength(java.io.File f);
    public native boolean createFileExclusively(java.lang.String s) throws java.io.IOException;
    public native boolean delete(java.io.File f);
    public synchronized native boolean deleteOnExit(java.io.File f);
    public native java.lang.String[] list(java.io.File f);
    public native boolean createDirectory(java.io.File f);
    public native boolean rename(java.io.File f1, java.io.File f2);
    public native boolean setLastModifiedTime(java.io.File f, long t);
    public native boolean setReadOnly(java.io.File f);
    
    // gets the current directory on the named drive.
    String getDriveDirectory(int i) {
        byte[] b = new byte[256];
        int result = SystemInterface.fs_getdcwd(i, b);
        if (result == 0) throw new InternalError();
        String res = SystemInterface.fromCString(Unsafe.addressOf(b));
        // skip "C:"
        if (res.charAt(1) == ':') return res.substring(2);
        else return res;
    }
    
    
}

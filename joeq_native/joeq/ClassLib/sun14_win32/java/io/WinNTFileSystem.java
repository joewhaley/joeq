/*
 * Win32FileSystem.java
 *
 * Created on January 29, 2001, 2:29 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun14_win32.java.io;

public abstract class WinNTFileSystem extends Win32FileSystem {

    public java.lang.String canonicalize(java.lang.String s) throws java.io.IOException { return super.canonicalize(s); }
    public int getBooleanAttributes(java.io.File f) { return super.getBooleanAttributes(f); }
    public boolean checkAccess(java.io.File f, boolean b) { return super.checkAccess(f, b); }
    public long getLastModifiedTime(java.io.File f) { return super.getLastModifiedTime(f); }
    public long getLength(java.io.File f) { return super.getLength(f); }
    public boolean createFileExclusively(java.lang.String s) throws java.io.IOException { return super.createFileExclusively(s); }
    public boolean delete(java.io.File f) { return super.delete(f); }
    public synchronized boolean deleteOnExit(java.io.File f) { return super.deleteOnExit(f); }
    public java.lang.String[] list(java.io.File f) { return super.list(f); }
    public boolean createDirectory(java.io.File f) { return super.createDirectory(f); }
    public boolean rename(java.io.File f1, java.io.File f2) { return super.rename(f1, f2); }
    public boolean setLastModifiedTime(java.io.File f, long t) { return super.setLastModifiedTime(f, t); }
    public boolean setReadOnly(java.io.File f) { return super.setReadOnly(f); }
    String getDriveDirectory(int i) { return super.getDriveDirectory(i); }
    
    private static void initIDs() {}
}

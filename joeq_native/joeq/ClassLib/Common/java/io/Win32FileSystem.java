/*
 * Win32FileSystem.java
 *
 * Created on January 29, 2001, 2:29 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.Common.java.io;

import Bootstrap.PrimordialClassLoader;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Clazz.jq_Class;
import Clazz.jq_Initializer;
import jq;

public abstract class Win32FileSystem {

    // gets the current directory on the named drive.
    private static String getDriveDirectory(int i) {
        byte[] b = new byte[256];
        int result = SystemInterface.fs_getdcwd(i, b);
        if (result == 0) throw new InternalError();
        jq.assert(result == Unsafe.addressOf(b));
        String res = SystemInterface.fromCString(result);
        // skip "C:"
        if (res.charAt(1) == ':') return res.substring(2);
        else return res;
    }

    public String canonicalize(String s) throws java.io.IOException {
        // check for and eliminate wildcards.
        if ((s.indexOf('*')>=0) || (s.indexOf('?')>=0))
            throw new java.io.IOException("wildcards not allowed in file name: "+s);
        byte[] b = new byte[256];
        int r = SystemInterface.fs_fullpath(s, b);
        if (r == 0) throw new java.io.IOException("fullpath returned error on: "+s);
        jq.assert(r == Unsafe.addressOf(b));
        String res = SystemInterface.fromCString(r);
        int strlen = res.length();
        StringBuffer result = new StringBuffer(strlen);
        int curindex = 0;
        if (res.startsWith("\\\\")) {
            // trim trailing "\" on UNC name.
            //if (res.charAt(strlen-1) == '\\') { res = res.substring(0, strlen-1); --strlen; }
            curindex = res.indexOf('\\', 2);
            if (curindex == -1) throw new java.io.IOException("invalid UNC pathname: "+s);
            result.append(res.substring(0, curindex));
        } else if (res.charAt(1) == ':') {
            // change drive letter to upper case.
            result.append(Character.toUpperCase(res.charAt(0)));
            result.append(':');
            curindex = 2;
        }
        while (curindex < strlen) {
            result.append('\\');
            int next_idx = res.indexOf('\\', curindex);
            if (next_idx == -1) {
                result.append(res.substring(curindex));
                return result.toString();
            }
            String sub = res.substring(curindex, next_idx);
            int b3 = SystemInterface.fs_gettruename(sub);
            if (b3 == 0) {
                // bail out and return what we have.
                result.append(res.substring(curindex));
                return result.toString();
            }
            result.append(SystemInterface.fromCString(b3));
            curindex = next_idx+1;
        }
        // path name ended in "\"
        return result.toString();
    }
    
    /* Constants for simple boolean attributes */
    public static final int BA_EXISTS    = 0x01;
    public static final int BA_REGULAR   = 0x02;
    public static final int BA_DIRECTORY = 0x04;
    public static final int BA_HIDDEN    = 0x08;
    public int getBooleanAttributes(java.io.File file) {
        int res = SystemInterface.fs_getfileattributes(file.getPath());
        if (res == -1) return 0;
        return BA_EXISTS |
               (((res & SystemInterface.FILE_ATTRIBUTE_DIRECTORY) != 0)?BA_DIRECTORY:BA_REGULAR) |
               (((res & SystemInterface.FILE_ATTRIBUTE_HIDDEN) != 0)?BA_HIDDEN:0);
    }

    public boolean checkAccess(java.io.File file, boolean flag) {
        int res = SystemInterface.fs_access(file.getPath(), flag?2:4);
        return res == 0;
    }

    public long getLastModifiedTime(java.io.File file) {
        long res = SystemInterface.fs_getfiletime(file.getPath());
        return res;
    }

    public long getLength(java.io.File file) {
        return SystemInterface.fs_stat_size(file.getPath());
    }
    
    public boolean delete(java.io.File file) {
        int res = SystemInterface.fs_remove(file.getPath());
        return res == 0;
    }
    
    public String[] list(java.io.File file) {
        int dir = SystemInterface.fs_opendir(file.getPath());
        if (dir == 0) return null;
        String[] s = new String[16];
        int ptr, i;
        for (i=0; 0!=(ptr=SystemInterface.fs_readdir(dir)); ++i) {
            if (i == s.length) {
                String[] s2 = new String[s.length<<1];
                System.arraycopy(s, 0, s2, 0, s.length);
                s = s2;
            }
            s[i] = SystemInterface.fromCString(ptr);
            if (s[i].equals(".") || s[i].equals("..")) --i;
        }
        SystemInterface.fs_closedir(dir);
        String[] ret = new String[i];
        System.arraycopy(s, 0, ret, 0, i);
        return ret;
    }

    public boolean createDirectory(java.io.File file) {
        int res = SystemInterface.fs_mkdir(file.getPath());
        return res == 0;
    }

    public boolean rename(java.io.File file, java.io.File file1) {
        int res = SystemInterface.fs_rename(file.getPath(), file1.getPath());
        return res == 0;
    }

    public boolean setLastModifiedTime(java.io.File file, long l) {
        int res = SystemInterface.fs_setfiletime(file.getPath(), l);
        return res != 0;
    }

    public boolean setReadOnly(java.io.File file) {
        int res = SystemInterface.fs_chmod(file.getPath(), SystemInterface._S_IREAD);
        return res == 0;
    }

    private static int listRoots0() {
        return SystemInterface.fs_getlogicaldrives();
    }
    
    private static void initIDs() { }
    
    //public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/Win32FileSystem;");
    //public static final jq_Initializer _constructor = (jq_Initializer)_class.getOrCreateInstanceMethod("<init>", "()V");
}

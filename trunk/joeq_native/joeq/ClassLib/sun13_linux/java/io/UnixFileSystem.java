package ClassLib.sun13_linux.java.io;

import Clazz.*;
import Bootstrap.*;
import Run_Time.SystemInterface;

import java.security.AccessController;
import sun.security.action.GetPropertyAction;

public abstract class UnixFileSystem {

    public static long getLength(/*java.io.UnixFileSystem*/Object dis, java.io.File file) {
        return SystemInterface.fs_stat_size(file.getPath());
    }

    public static boolean delete(/*java.io.UnixFileSystem*/Object dis, java.io.File file) {
        int res = SystemInterface.fs_remove(file.getPath());
        return res == 0;
    }

    public static long getLastModifiedTime(/*java.io.UnixFileSystem*/Object dis, java.io.File file) {
        long res = SystemInterface.fs_getfiletime(file.getPath());
        return res;
    }

    public static boolean rename(/*java.io.UnixFileSystem*/Object dis, java.io.File file, java.io.File file1) {
        int res = SystemInterface.fs_rename(file.getPath(), file1.getPath());
        return res == 0;
    }

    public static String[] list(/*java.io.UnixFileSystem*/Object dis, java.io.File file) {
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

    public static boolean createDirectory(/*java.io.UnixFileSystem*/Object dis, java.io.File file) {
        int res = SystemInterface.fs_mkdir(file.getPath());
        return res == 0;
    }

    public static boolean setLastModifiedTime(/*java.io.UnixFileSystem*/Object dis, java.io.File file, long l) {
        int res = SystemInterface.fs_setfiletime(file.getPath(), l);
        return res != 0;
    }

    /*
    public native String canonicalize(String s)
        throws IOException;
    public native int getBooleanAttributes0(File file);
    public native boolean checkAccess(File file, boolean flag);
    public native boolean createFileExclusively(String s)
        throws IOException;
    public synchronized native boolean deleteOnExit(File file);
    public native boolean setReadOnly(File file);
    */
    private static void initIDs(jq_Class clazz) { }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/UnixFileSystem;");
    public static final jq_Initializer _constructor = (jq_Initializer)_class.getOrCreateInstanceMethod("<init>", "()V");
}

// UnixFileSystem.java, created Thu Jul  4  4:50:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.io;

import Memory.Address;
import Run_Time.SystemInterface;

/**
 * UnixFileSystem
 *
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class UnixFileSystem {

    public long getLength(java.io.File file) {
        return SystemInterface.fs_stat_size(file.getPath());
    }

    public boolean delete(java.io.File file) {
        int res = SystemInterface.fs_remove(file.getPath());
        return res == 0;
    }

    public long getLastModifiedTime(java.io.File file) {
        long res = SystemInterface.fs_getfiletime(file.getPath());
        return res;
    }

    public boolean rename(java.io.File file, java.io.File file1) {
        int res = SystemInterface.fs_rename(file.getPath(), file1.getPath());
        return res == 0;
    }

    public String[] list(java.io.File file) {
        int dir = SystemInterface.fs_opendir(file.getPath());
        if (dir == 0) return null;
        String[] s = new String[16];
        int i;
        Address ptr;
        for (i=0; !(ptr=SystemInterface.fs_readdir(dir)).isNull(); ++i) {
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

    public boolean setLastModifiedTime(java.io.File file, long l) {
        int res = SystemInterface.fs_setfiletime(file.getPath(), l);
        return res != 0;
    }

    public String canonicalize(String s) throws java.io.IOException {
        // TODO.
        return s;
    }

    /*
    public native int getBooleanAttributes0(File file);
    public native boolean checkAccess(File file, boolean flag);
    public native boolean createFileExclusively(String s)
        throws IOException;
    public synchronized native boolean deleteOnExit(File file);
    public native boolean setReadOnly(File file);
    */
    private static void initIDs() { }
    
    //public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/UnixFileSystem;");
    //public static final jq_Initializer _constructor = (jq_Initializer)_class.getOrCreateInstanceMethod("<init>", "()V");
}

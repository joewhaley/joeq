package ClassLib.ibm13_linux.java.io;

import Clazz.*;
import Bootstrap.*;

import java.security.AccessController;
import sun.security.action.GetPropertyAction;

public abstract class UnixFileSystem {

    /*
    public native String canonicalize(String s)
        throws IOException;
    public native int getBooleanAttributes0(File file);
    public native boolean checkAccess(File file, boolean flag);
    public native long getLastModifiedTime(File file);
    public native long getLength(File file);
    public native boolean createFileExclusively(String s)
        throws IOException;
    public native boolean delete(File file);
    public synchronized native boolean deleteOnExit(File file);
    public native String[] list(File file);
    public native boolean createDirectory(File file);
    public native boolean rename(File file, File file1);
    public native boolean setLastModifiedTime(File file, long l);
    public native boolean setReadOnly(File file);
    */
    private static void initIDs(jq_Class clazz) { }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/UnixFileSystem;");
    public static final jq_Initializer _constructor = (jq_Initializer)_class.getOrCreateInstanceMethod("<init>", "()V");
}

/*
 * SystemCall.java
 *
 * Created on January 1, 2001, 10:30 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_StaticField;
import Clazz.jq_NameAndDesc;
import UTF.Utf8;
import Run_Time.Unsafe;
import jq;


public abstract class SystemInterface {

    // NOTE: the order of the static fields here must match the bootstrap loader.
    
    public static int/*Address*/ entry;
    public static int/*Address*/ trap_handler;
    public static int/*Address*/ debugmsg;
    public static int/*Address*/ syscalloc;
    public static int/*Address*/ die;
    public static int/*Address*/ currentTimeMillis;
    public static int/*Address*/ memcpy;
    public static int/*Address*/ file_open;
    public static int/*Address*/ file_readbytes;
    public static int/*Address*/ file_writebyte;
    public static int/*Address*/ file_writebytes;
    public static int/*Address*/ file_sync;
    public static int/*Address*/ file_seek;
    public static int/*Address*/ file_close;
    public static int/*Address*/ console_available;
    public static int/*Address*/ main_argc;
    public static int/*Address*/ main_argv_length;
    public static int/*Address*/ main_argv;
    public static int/*Address*/ fs_getdcwd;
    public static int/*Address*/ fs_fullpath;
    public static int/*Address*/ fs_gettruename;
    public static int/*Address*/ fs_getfileattributes;
    public static int/*Address*/ fs_access;
    public static int/*Address*/ fs_getfiletime;
    public static int/*Address*/ fs_stat_size;
    public static int/*Address*/ fs_remove;
    public static int/*Address*/ fs_opendir;
    public static int/*Address*/ fs_readdir;
    public static int/*Address*/ fs_closedir;
    public static int/*Address*/ fs_mkdir;
    public static int/*Address*/ fs_rename;
    public static int/*Address*/ fs_chmod;
    public static int/*Address*/ fs_setfiletime;
    public static int/*Address*/ fs_getlogicaldrives;

    public static final jq_Class _class;
    public static final jq_StaticField _entry;
    public static final jq_StaticField _trap_handler;
    public static final jq_StaticField _debugmsg;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/SystemInterface;");
        _entry = _class.getOrCreateStaticField("entry", "I");
        _trap_handler = _class.getOrCreateStaticField("trap_handler", "I");
        _debugmsg = _class.getOrCreateStaticField("debugmsg", "I");
    }

    public static void debugmsg(String msg) {
        if (jq.Bootstrapping) System.err.println(msg);
        else {
            debugmsg(toCString(msg));
        }
    }
    
    public static void debugmsg(byte[] msg) {
        Unsafe.pushArg(Unsafe.addressOf(msg));
        try { Unsafe.invoke(debugmsg); } catch (Throwable t) {}
    }
    
    public static int syscalloc(int size) {
        Unsafe.pushArg(size);
        try {
            long r = Unsafe.invoke(syscalloc);
            return (int)r;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static void die() {
        try { Unsafe.invoke(die); } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static byte[] toCString(String s) {
        int len = s.length();
        byte[] b = new byte[len+1];
        s.getBytes(0, len, b, 0);
        return b;
    }

    public static String fromCString(int p) {
        int len;
        for (len=0; (byte)Unsafe.peek(p+len)!=(byte)0; ++len) ;
        byte[] b = new byte[len];
        memcpy(Unsafe.addressOf(b), p, len);
        return new String(b);
    }
    
    public static long currentTimeMillis() {
        //if (jq.Bootstrapping)
        //    return System.currentTimeMillis();
        //else
            try { return Unsafe.invoke(currentTimeMillis); } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }

    public static void memcpy(int to, int from, int size) {
        Unsafe.pushArg(size);
        Unsafe.pushArg(from);
        Unsafe.pushArg(to);
        try {
            Unsafe.invoke(memcpy);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }

    // constants from fcntl.h
    public static final int _O_RDONLY = 0x0000;
    public static final int _O_WRONLY = 0x0001;
    public static final int _O_RDWR   = 0x0002;
    public static final int _O_APPEND = 0x0008;
    public static final int _O_CREAT  = 0x0100;
    public static final int _O_TRUNC  = 0x0200;
    public static final int _O_EXCL   = 0x0400;
    public static final int _O_TEXT   = 0x4000;
    public static final int _O_BINARY = 0x8000;
    public static int file_open(byte[] filename, int mode, int smode) {
        Unsafe.pushArg(smode);
        Unsafe.pushArg(mode);
        Unsafe.pushArg(Unsafe.addressOf(filename));
        try {
            return (int)Unsafe.invoke(file_open);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_readbytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_readbytes);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebyte(int fd, int b) {
        Unsafe.pushArg(b);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_writebyte);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_writebytes);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_sync(int fd) {
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_sync);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static final int SEEK_SET = 0; // from stdio.h
    public static final int SEEK_CUR = 1;
    public static final int SEEK_END = 2;
    public static long file_seek(int fd, long offset, int origin) {
        Unsafe.pushArg((int)origin);
        Unsafe.pushArg((int)(offset>>32)); // hi
        Unsafe.pushArg((int)offset);       // lo
        Unsafe.pushArg(fd);
        try {
            return Unsafe.invoke(file_seek);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_close(int fd) {
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_close);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int console_available() {
        try {
            long r = Unsafe.invoke(console_available);
            return (int)r;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int main_argc() {
        try {
            return (int)Unsafe.invoke(main_argc);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int main_argv_length(int i) {
        Unsafe.pushArg(i);
        try {
            return (int)Unsafe.invoke(main_argv_length);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static void main_argv(int i, byte[] b) {
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            Unsafe.invoke(main_argv);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static int fs_getdcwd(int i, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            return (int)Unsafe.invoke(fs_getdcwd);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_fullpath(String s, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        Unsafe.pushArg(Unsafe.addressOf(b));
        try {
            return (int)Unsafe.invoke(fs_fullpath);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_gettruename(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_gettruename);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static final int FILE_ATTRIBUTE_READONLY  = 0x001; // in mapiwin.h
    public static final int FILE_ATTRIBUTE_HIDDEN    = 0x002;
    public static final int FILE_ATTRIBUTE_SYSTEM    = 0x004;
    public static final int FILE_ATTRIBUTE_DIRECTORY = 0x010;
    public static final int FILE_ATTRIBUTE_ARCHIVE   = 0x020;
    public static final int FILE_ATTRIBUTE_NORMAL    = 0x080;
    public static final int FILE_ATTRIBUTE_TEMPORARY = 0x100;
    public static int fs_getfileattributes(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_getfileattributes);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_access(String s, int mode) {
        Unsafe.pushArg(mode);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_access);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static long fs_getfiletime(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return Unsafe.invoke(fs_getfiletime);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static long fs_stat_size(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return Unsafe.invoke(fs_stat_size);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static int fs_remove(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_remove);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_opendir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_opendir);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_readdir(int p) {
        Unsafe.pushArg(p);
        try {
            return (int)Unsafe.invoke(fs_readdir);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_closedir(int p) {
        Unsafe.pushArg(p);
        try {
            return (int)Unsafe.invoke(fs_closedir);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_mkdir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_mkdir);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_rename(String s, String s1) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s1)));
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_rename);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static final int _S_IEXEC  = 0x0000040; // from sys/stat.h
    public static final int _S_IWRITE = 0x0000080;
    public static final int _S_IREAD  = 0x0000100;
    public static int fs_chmod(String s, int mode) {
        Unsafe.pushArg(mode);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_chmod);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_setfiletime(String s, long time) {
        Unsafe.pushArg((int)(time>>32)); // hi
        Unsafe.pushArg((int)time);       // lo
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_setfiletime);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_getlogicaldrives() {
        try {
            return (int)Unsafe.invoke(fs_getlogicaldrives);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }

}

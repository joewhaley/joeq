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
    
    public static int/*CodeAddress*/ entry_0;
    public static int/*CodeAddress*/ trap_handler_8;
    public static int/*CodeAddress*/ debugmsg_4;
    public static int/*CodeAddress*/ syscalloc_4;
    public static int/*CodeAddress*/ die_4;
    public static int/*CodeAddress*/ currentTimeMillis_0;
    public static int/*CodeAddress*/ mem_cpy_12;
    public static int/*CodeAddress*/ file_open_12;
    public static int/*CodeAddress*/ file_readbytes_12;
    public static int/*CodeAddress*/ file_writebyte_8;
    public static int/*CodeAddress*/ file_writebytes_12;
    public static int/*CodeAddress*/ file_sync_4;
    public static int/*CodeAddress*/ file_seek_16;
    public static int/*CodeAddress*/ file_close_4;
    public static int/*CodeAddress*/ console_available_0;
    public static int/*CodeAddress*/ main_argc_0;
    public static int/*CodeAddress*/ main_argv_length_4;
    public static int/*CodeAddress*/ main_argv_8;
    public static int/*CodeAddress*/ fs_getdcwd_12;
    public static int/*CodeAddress*/ fs_fullpath_12;
    public static int/*CodeAddress*/ fs_gettruename_4;
    public static int/*CodeAddress*/ fs_getfileattributes_4;
    public static int/*CodeAddress*/ fs_access_8;
    public static int/*CodeAddress*/ fs_getfiletime_4;
    public static int/*CodeAddress*/ fs_stat_size_4;
    public static int/*CodeAddress*/ fs_remove_4;
    public static int/*CodeAddress*/ fs_opendir_4;
    public static int/*CodeAddress*/ fs_readdir_4;
    public static int/*CodeAddress*/ fs_closedir_4;
    public static int/*CodeAddress*/ fs_mkdir_4;
    public static int/*CodeAddress*/ fs_rename_8;
    public static int/*CodeAddress*/ fs_chmod_8;
    public static int/*CodeAddress*/ fs_setfiletime_12;
    public static int/*CodeAddress*/ fs_getlogicaldrives_0;

    public static final jq_Class _class;
    public static final jq_StaticField _entry;
    public static final jq_StaticField _trap_handler;
    public static final jq_StaticField _debugmsg;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/SystemInterface;");
        _entry = _class.getOrCreateStaticField("entry_0", "I");
        _trap_handler = _class.getOrCreateStaticField("trap_handler_8", "I");
        _debugmsg = _class.getOrCreateStaticField("debugmsg_4", "I");
    }

    public static void debugmsg(String msg) {
        if (jq.Bootstrapping) System.err.println(msg);
        else {
            debugmsg(toCString(msg));
        }
    }
    
    public static void debugmsg(byte[] msg) {
        Unsafe.pushArg(Unsafe.addressOf(msg));
        try { Unsafe.invoke(debugmsg_4); } catch (Throwable t) {}
    }
    
    public static int syscalloc(int size) {
        Unsafe.pushArg(size);
        try {
            long r = Unsafe.invoke(syscalloc_4);
            return (int)r;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static void die(int code) {
        Unsafe.pushArg(code);
        try {
            Unsafe.invoke(die_4);
        } catch (Throwable t) {
            throw new InternalError();
        }
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
        mem_cpy(Unsafe.addressOf(b), p, len);
        return new String(b);
    }
    
    public static long currentTimeMillis() {
        //if (jq.Bootstrapping)
        //    return System.currentTimeMillis();
        //else
            try { return Unsafe.invoke(currentTimeMillis_0); } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }

    public static void mem_cpy(int to, int from, int size) {
        Unsafe.pushArg(size);
        Unsafe.pushArg(from);
        Unsafe.pushArg(to);
        try {
            Unsafe.invoke(mem_cpy_12);
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
            return (int)Unsafe.invoke(file_open_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_readbytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_readbytes_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebyte(int fd, int b) {
        Unsafe.pushArg(b);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_writebyte_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_writebytes_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_sync(int fd) {
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_sync_4);
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
            return Unsafe.invoke(file_seek_16);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_close(int fd) {
        Unsafe.pushArg(fd);
        try {
            return (int)Unsafe.invoke(file_close_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int console_available() {
        try {
            long r = Unsafe.invoke(console_available_0);
            return (int)r;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int main_argc() {
        try {
            return (int)Unsafe.invoke(main_argc_0);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int main_argv_length(int i) {
        Unsafe.pushArg(i);
        try {
            return (int)Unsafe.invoke(main_argv_length_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static void main_argv(int i, byte[] b) {
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            Unsafe.invoke(main_argv_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static int fs_getdcwd(int i, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            return (int)Unsafe.invoke(fs_getdcwd_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_fullpath(String s, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        Unsafe.pushArg(Unsafe.addressOf(b));
        try {
            return (int)Unsafe.invoke(fs_fullpath_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_gettruename(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_gettruename_4);
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
            return (int)Unsafe.invoke(fs_getfileattributes_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_access(String s, int mode) {
        Unsafe.pushArg(mode);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_access_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static long fs_getfiletime(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return Unsafe.invoke(fs_getfiletime_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static long fs_stat_size(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return Unsafe.invoke(fs_stat_size_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static int fs_remove(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_remove_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_opendir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_opendir_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_readdir(int p) {
        Unsafe.pushArg(p);
        try {
            return (int)Unsafe.invoke(fs_readdir_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_closedir(int p) {
        Unsafe.pushArg(p);
        try {
            return (int)Unsafe.invoke(fs_closedir_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_mkdir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_mkdir_4);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_rename(String s, String s1) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s1)));
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_rename_8);
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
            return (int)Unsafe.invoke(fs_chmod_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_setfiletime(String s, long time) {
        Unsafe.pushArg((int)(time>>32)); // hi
        Unsafe.pushArg((int)time);       // lo
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            return (int)Unsafe.invoke(fs_setfiletime_12);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_getlogicaldrives() {
        try {
            return (int)Unsafe.invoke(fs_getlogicaldrives_0);
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }

}

/*
 * SystemInterface.java
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
import Clazz.jq_InstanceField;
import Clazz.jq_NameAndDesc;
import UTF.Utf8;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;
import Scheduler.jq_NativeThread;
import Scheduler.jq_RegisterState;
import jq;

public abstract class SystemInterface {

    public static int/*CodeAddress*/ debugmsg_4;
    public static int/*CodeAddress*/ debugwmsg_8;
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
    public static int/*CodeAddress*/ yield_0;
    public static int/*CodeAddress*/ msleep_4;
    public static int/*CodeAddress*/ create_thread_8;
    public static int/*CodeAddress*/ init_thread_0;
    public static int/*CodeAddress*/ resume_thread_4;
    public static int/*CodeAddress*/ suspend_thread_4;
    public static int/*CodeAddress*/ allocate_stack_4;
    public static int/*CodeAddress*/ get_current_thread_handle_0;
    public static int/*CodeAddress*/ get_thread_context_8;
    public static int/*CodeAddress*/ set_thread_context_8;
    public static int/*CodeAddress*/ set_current_context_8;
    public static int/*CodeAddress*/ set_interval_timer_8;
    public static int/*CodeAddress*/ init_semaphore_0;
    public static int/*CodeAddress*/ wait_for_single_object_8;
    public static int/*CodeAddress*/ release_semaphore_8;

    public static final jq_Class _class;
    public static final jq_StaticField _debugmsg;
    public static final jq_InstanceField _string_value;
    public static final jq_InstanceField _string_offset;
    public static final jq_InstanceField _string_count;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/SystemInterface;");
        _debugmsg = _class.getOrCreateStaticField("debugmsg_4", "I");
        _string_value = PrimordialClassLoader.getJavaLangString().getOrCreateInstanceField("value", "[C");
        _string_offset = PrimordialClassLoader.getJavaLangString().getOrCreateInstanceField("offset", "I");
        _string_count = PrimordialClassLoader.getJavaLangString().getOrCreateInstanceField("count", "I");
    }

    public static void debugmsg(String msg) {
        if (jq.Bootstrapping) {
            System.err.println(msg);
            return;
        }
        int/*HeapAddress*/ value = Unsafe.peek(Unsafe.addressOf(msg)+_string_value.getOffset());
        int offset = Unsafe.peek(Unsafe.addressOf(msg)+_string_offset.getOffset());
        int count = Unsafe.peek(Unsafe.addressOf(msg)+_string_count.getOffset());
        Unsafe.pushArg(count);
        Unsafe.pushArg(value + offset*2);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(debugwmsg_8);
            Unsafe.getThreadBlock().enableThreadSwitch();
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static void debugmsg(byte[] msg) {
        Unsafe.pushArg(Unsafe.addressOf(msg));
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
	    Unsafe.invoke(debugmsg_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	} catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static int syscalloc(int size) {
        Unsafe.pushArg(size);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(syscalloc_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
            return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static void die(int code) {
        Unsafe.pushArg(code);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(die_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
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
            try {
		Unsafe.getThreadBlock().disableThreadSwitch();
		long v = Unsafe.invoke(currentTimeMillis_0);
		Unsafe.getThreadBlock().enableThreadSwitch();
		return v;
	    } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }

    public static void mem_cpy(int to, int from, int size) {
        Unsafe.pushArg(size);
        Unsafe.pushArg(from);
        Unsafe.pushArg(to);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(mem_cpy_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
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
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_open_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_readbytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_readbytes_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebyte(int fd, int b) {
        Unsafe.pushArg(b);
        Unsafe.pushArg(fd);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_writebyte_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_writebytes(int fd, int/*Address*/ startAddress, int length) {
        Unsafe.pushArg(length);
        Unsafe.pushArg(startAddress);
        Unsafe.pushArg(fd);
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_writebytes_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_sync(int fd) {
        Unsafe.pushArg(fd);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_sync_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
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
            Unsafe.getThreadBlock().disableThreadSwitch();
            long v = Unsafe.invoke(file_seek_16);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int file_close(int fd) {
        Unsafe.pushArg(fd);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(file_close_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int console_available() {
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(console_available_0);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    
    public static int main_argc() {
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(main_argc_0);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int main_argv_length(int i) {
        Unsafe.pushArg(i);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(main_argv_length_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static void main_argv(int i, byte[] b) {
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(main_argv_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    
    public static int fs_getdcwd(int i, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(b));
        Unsafe.pushArg(i);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_getdcwd_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_fullpath(String s, byte[] b) {
        Unsafe.pushArg(b.length);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        Unsafe.pushArg(Unsafe.addressOf(b));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_fullpath_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_gettruename(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_gettruename_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
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
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_getfileattributes_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_access(String s, int mode) {
        Unsafe.pushArg(mode);
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_access_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static long fs_getfiletime(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            long v = Unsafe.invoke(fs_getfiletime_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static long fs_stat_size(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            long v = Unsafe.invoke(fs_stat_size_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    public static int fs_remove(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_remove_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_opendir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_opendir_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_readdir(int p) {
        Unsafe.pushArg(p);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_readdir_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_closedir(int p) {
        Unsafe.pushArg(p);
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_closedir_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_mkdir(String s) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_mkdir_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_rename(String s, String s1) {
        Unsafe.pushArg(Unsafe.addressOf(toCString(s1)));
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_rename_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
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
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_chmod_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_setfiletime(String s, long time) {
        Unsafe.pushArg((int)(time>>32)); // hi
        Unsafe.pushArg((int)time);       // lo
        Unsafe.pushArg(Unsafe.addressOf(toCString(s)));
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_setfiletime_12);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int fs_getlogicaldrives() {
        try {
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(fs_getlogicaldrives_0);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static void yield() {
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(yield_0);
	    Unsafe.getThreadBlock().enableThreadSwitch();
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static void msleep(int ms) {
        try {
            Unsafe.pushArg(ms);
	    Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(msleep_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static int/*CPointer*/ create_thread(int/*CodeAddress*/ start_address, int/*HeapAddress*/ param) {
        try {
            Unsafe.pushArg(param);
            Unsafe.pushArg(start_address);
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(create_thread_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int init_thread() {
        try {
            int v = (int)Unsafe.invoke(init_thread_0);
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
	return 0;
    }
    public static int resume_thread(int/*CPointer*/ thread_handle) {
        try {
            Unsafe.pushArg(thread_handle);
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            int v = (int)Unsafe.invoke(resume_thread_4);
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int suspend_thread(int/*CPointer*/ thread_handle) {
        try {
            Unsafe.pushArg(thread_handle);
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            int v = (int)Unsafe.invoke(suspend_thread_4);
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int/*StackAddress*/ allocate_stack(int size) {
        try {
            Unsafe.pushArg(size);
            Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(allocate_stack_4);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int/*CPointer*/ get_current_thread_handle() {
        try {
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            int v = (int)Unsafe.invoke(get_current_thread_handle_0);
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static void get_thread_context(int pid, jq_RegisterState context) {
        try {
            Unsafe.pushArg(Unsafe.addressOf(context));
            Unsafe.pushArg(pid);
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            Unsafe.invoke(get_thread_context_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static void set_thread_context(int pid, jq_RegisterState context) {
        try {
            Unsafe.pushArg(Unsafe.addressOf(context));
            Unsafe.pushArg(pid);
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            Unsafe.invoke(set_thread_context_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static void set_current_context(jq_Thread thread, jq_RegisterState context) {
        try {
            Unsafe.pushArg(Unsafe.addressOf(context));
            Unsafe.pushArg(Unsafe.addressOf(thread));
	    jq.assert(!Unsafe.getThreadBlock().isThreadSwitchEnabled());
            Unsafe.invoke(set_current_context_8);
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static final int ITIMER_VIRTUAL = 1;
    public static void set_interval_timer(int type, int ms) {
        try {
            Unsafe.pushArg(ms);
            Unsafe.pushArg(type);
	    Unsafe.getThreadBlock().disableThreadSwitch();
            Unsafe.invoke(set_interval_timer_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
        } catch (Throwable t) { jq.UNREACHABLE(); }
    }
    public static int/*CPointer*/ init_semaphore() {
        try {
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(init_semaphore_0);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static final int INFINITE = -1;
    public static final int WAIT_ABANDONED = 0x00000080;
    public static final int WAIT_OBJECT_0  = 0x00000000;
    public static final int WAIT_TIMEOUT   = 0x00000102;
    public static int wait_for_single_object(int/*CPointer*/ obj, int timeout) {
        try {
            Unsafe.pushArg(timeout);
            Unsafe.pushArg(obj);
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(wait_for_single_object_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
    public static int release_semaphore(int/*CPointer*/ semaphore, int v1) {
        try {
            Unsafe.pushArg(v1);
            Unsafe.pushArg(semaphore);
	    Unsafe.getThreadBlock().disableThreadSwitch();
            int v = (int)Unsafe.invoke(release_semaphore_8);
	    Unsafe.getThreadBlock().enableThreadSwitch();
	    return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0;
    }
}

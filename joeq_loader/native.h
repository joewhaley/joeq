
extern "C" void __stdcall debugmsg(const char* s);
extern "C" void* __stdcall syscalloc(const int size);
extern "C" void __stdcall die(const int code);
extern "C" __int64 __stdcall currentTimeMillis(void);
extern "C" void __stdcall mem_cpy(void* to, const void* from, const int size);
extern "C" int __stdcall file_open(const char* s, const int mode, const int smode);
extern "C" int __stdcall file_readbytes(const int fd, char* b, const int len);
extern "C" int __stdcall file_writebyte(const int fd, const int b);
extern "C" int __stdcall file_writebytes(const int fd, const char* b, const int len);
extern "C" int __stdcall file_sync(const int fd);
extern "C" __int64 __stdcall file_seek(const int fd, const __int64 offset, const int origin);
extern "C" int __stdcall file_close(const int fd);
extern "C" int __stdcall console_available(void);
extern "C" int __stdcall main_argc(void);
extern "C" int __stdcall main_argv_length(const int i);
extern "C" void __stdcall main_argv(const int i, char* buf);
extern int _argc;
extern char** _argv;
#if defined(WIN32)
struct dirent {
	char d_name[MAX_PATH];
};
typedef struct {
    struct dirent dirent;
    char *path;
    HANDLE handle;
    WIN32_FIND_DATA find_data;
} DIR;
extern "C" DIR * __stdcall fs_opendir(const char* s);
extern "C" struct dirent * __stdcall fs_readdir(DIR *dir);
extern "C" int __stdcall fs_closedir(DIR *dir);
#endif
extern "C" int __stdcall fs_getdcwd(const int i, char* buf, const int buflen);
extern "C" int __stdcall fs_fullpath(char* buf, const char* s, const int buflen);
extern "C" int __stdcall fs_getfileattributes(const char* s);
extern "C" char* __stdcall fs_gettruename(char* s);
extern "C" int __stdcall fs_access(const char* s, int mode);
extern "C" __int64 __stdcall fs_getfiletime(const char* s);
extern "C" __int64 __stdcall fs_stat_size(const char* s);
extern "C" int __stdcall fs_remove(const char* s);
extern "C" int __stdcall fs_mkdir(const char* s);
extern "C" int __stdcall fs_rename(const char* s, const char* s1);
extern "C" int __stdcall fs_chmod(const char* s, const int mode);
extern "C" int __stdcall fs_setfiletime(const char* s, const __int64 time);
extern "C" int __stdcall fs_getlogicaldrives(void);

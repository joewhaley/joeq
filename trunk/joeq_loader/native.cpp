// native.c : Native method implementations
//

#include "stdafx.h"

#define ARRAY_LENGTH_OFFSET -12

extern "C" void __stdcall debugwmsg(const wchar_t* s)
{
	int* length_loc = (int*)((int)s + ARRAY_LENGTH_OFFSET);
	int length = *length_loc;
	while (--length >= 0) {
		putwchar(*s);
		++s;
	}
	putchar('\n');
	fflush(stdout);
}

extern "C" void __stdcall debugmsg(const char* s)
{
	puts(s);
	fflush(stdout);
}

extern "C" void* __stdcall syscalloc(const int size)
{
	return calloc(1, size);
}

extern "C" void __stdcall die(const int code)
{
	fflush(stdout);
	fflush(stderr);
	exit(code);
}

extern "C" __int64 __stdcall filetimeToJavaTime(const FILETIME* fileTime)
{
	LARGE_INTEGER time;
	time.LowPart = fileTime->dwLowDateTime; time.HighPart = fileTime->dwHighDateTime;
	return (time.QuadPart / 10000L) - 11644473600000L;
}

extern "C" void __stdcall javaTimeToFiletime(const __int64 javaTime, FILETIME* fileTime)
{
	LARGE_INTEGER time;
	time.QuadPart = (javaTime + 11644473600000L) * 10000L;
	fileTime->dwLowDateTime = time.LowPart;
	fileTime->dwHighDateTime = time.HighPart;
}

extern "C" __int64 __stdcall currentTimeMillis(void)
{
	FILETIME fileTime;
	GetSystemTimeAsFileTime(&fileTime);
	return filetimeToJavaTime(&fileTime);
}

extern "C" void __stdcall mem_cpy(void* to, const void* from, const int size)
{
	memcpy(to, from, size);
}

extern "C" int __stdcall file_open(const char* s, const int mode, const int smode)
{
	return _open(s, mode, smode);
}
extern "C" int __stdcall file_readbytes(const int fd, char* b, const int len)
{
	return _read(fd, b, len);
}
extern "C" int __stdcall file_writebyte(const int fd, const int b)
{
	return _write(fd, &b, 1);
}
extern "C" int __stdcall file_writebytes(const int fd, const char* b, const int len)
{
	return _write(fd, b, len);
}
extern "C" int __stdcall file_sync(const int fd)
{
	return _commit(fd);
}
extern "C" __int64 __stdcall file_seek(const int fd, const __int64 offset, const int origin)
{
	return _lseeki64(fd, offset, origin);
}
extern "C" int __stdcall file_close(const int fd)
{
	return _close(fd);
}
extern "C" int __stdcall console_available(void)
{
	HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
	unsigned long count;
	if (!GetNumberOfConsoleInputEvents(in, &count)) return -1;
	else return (int)count;
}
extern "C" int __stdcall main_argc(void) {
	return _argc;
}
extern "C" int __stdcall main_argv_length(const int i) {
	return strlen(_argv[i]);
}
extern "C" void __stdcall main_argv(const int i, char* buf) {
	memcpy(buf, _argv[i], strlen(_argv[i])*sizeof(char));
}
extern "C" int __stdcall fs_getdcwd(const int i, char* buf, const int buflen) {
	return _getdcwd(i, buf, buflen)?1:0;
}
extern "C" int __stdcall fs_fullpath(char* buf, const char* s, const int buflen) {
	return _fullpath(buf, s, buflen)?1:0;
}
extern "C" int __stdcall fs_getfileattributes(const char* s) {
	return GetFileAttributes(s);
}
extern "C" char* __stdcall fs_gettruename(const char* s) {
	WIN32_FIND_DATA fd;
	HANDLE h = FindFirstFile(s, &fd);
	if (h == INVALID_HANDLE_VALUE) return NULL;
	FindClose(h);
	return fd.cFileName;
}
extern "C" int __stdcall fs_access(const char* s, const int mode) {
	return _access(s, mode);
}
extern "C" __int64 __stdcall fs_getfiletime(const char* s) {
	FILETIME fileTime;
	HANDLE file = CreateFile(s, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	int res = GetFileTime(file, NULL, NULL, &fileTime);
	CloseHandle(file);
	if (res == 0) return 0;
	return filetimeToJavaTime(&fileTime);
}
extern "C" __int64 __stdcall fs_stat_size(const char* s) {
	struct _stati64 buf;
	int res = _stati64(s, &buf);
	if (res != 0) return 0;
	return buf.st_size;
}
extern "C" int __stdcall fs_remove(const char* s) {
	return remove(s);
}
extern "C" DIR * __stdcall fs_opendir(const char* s) {
    DIR *dir = (DIR *)malloc(sizeof(DIR));
	if (!dir) return NULL;

    if (s[0] == '\\' && s[1] == 0) {
		char dirname2[4] = { _getdrive()+'A'-1, ':', '\\', 0 };
		s = dirname2;
    }

    dir->path = (char *)malloc(strlen(s)+5);
    if (!dir->path) {
		free(dir); return NULL;
    }
    strcpy(dir->path, s);

    int file_attr = GetFileAttributes(dir->path);
    if ((file_attr == -1) || (file_attr & FILE_ATTRIBUTE_DIRECTORY == 0)) {
		free(dir->path); free(dir); return NULL;
    }

    if (dir->path[1] == ':' && (dir->path[2] == 0 || (dir->path[2] == '\\' && dir->path[3] == 0))) {
		strcat(dir->path, "*.*");
    } else {
		strcat(dir->path, "\\*.*");
    }

    dir->handle = FindFirstFile(dir->path, &dir->find_data);
    if (dir->handle == INVALID_HANDLE_VALUE) {
        if (GetLastError() != ERROR_FILE_NOT_FOUND) {
			free(dir->path); free(dir); return NULL;
		}
    }
    return dir;
}
extern "C" struct dirent * __stdcall fs_readdir(DIR *dir) {
    if (dir->handle == INVALID_HANDLE_VALUE) return NULL;
    strcpy(dir->dirent.d_name, dir->find_data.cFileName);
    if (!FindNextFile(dir->handle, &dir->find_data)) {
		if (GetLastError() == ERROR_INVALID_HANDLE) return NULL;
		FindClose(dir->handle);
		dir->handle = INVALID_HANDLE_VALUE;
    }
    return &dir->dirent;
}
extern "C" int __stdcall fs_closedir(DIR *dir) {
    if (dir->handle != INVALID_HANDLE_VALUE) {
		if (!FindClose(dir->handle)) return -1;
		dir->handle = INVALID_HANDLE_VALUE;
    }
    free(dir->path); free(dir);
    return 0;
}
extern "C" int __stdcall fs_mkdir(const char* s) {
	return _mkdir(s);
}
extern "C" int __stdcall fs_rename(const char* s, const char* s1) {
	return rename(s, s1);
}
extern "C" int __stdcall fs_chmod(const char* s, const int mode) {
	return _chmod(s, mode);
}
extern "C" int __stdcall fs_setfiletime(const char* s, const __int64 time) {
	FILETIME fileTime;
	javaTimeToFiletime(time, &fileTime);
	HANDLE file = CreateFile(s, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	int res = SetFileTime(file, NULL, NULL, &fileTime);
	CloseHandle(file);
	return res;
}
extern "C" int __stdcall fs_getlogicaldrives(void) {
	return GetLogicalDrives();
}

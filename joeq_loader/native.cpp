// native.c : Native method implementations
//

#include "StdAfx.h"

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

#if defined(WIN32)
__int64 __stdcall filetimeToJavaTime(const FILETIME* fileTime)
{
	LARGE_INTEGER time;
	time.LowPart = fileTime->dwLowDateTime; time.HighPart = fileTime->dwHighDateTime;
	return (time.QuadPart / 10000L) - 11644473600000L;
}

void __stdcall javaTimeToFiletime(const __int64 javaTime, FILETIME* fileTime)
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
#else

extern "C" __int64 __stdcall currentTimeMillis(void)
{
	struct timeb t;
	ftime(&t);
	return ((__int64)(t.time))*1000 + t.millitm;
}

#endif

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
#if defined(WIN32)
extern "C" int __stdcall console_available(void)
{
	HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
	unsigned long count;
	if (!GetNumberOfConsoleInputEvents(in, &count)) return -1;
	else return (int)count;
}
#else
extern "C" int __stdcall console_available(void)
{
	return 0; // TODO
}
#endif
extern "C" int __stdcall main_argc(void)
{
	return _argc;
}
extern "C" int __stdcall main_argv_length(const int i)
{
	return strlen(_argv[i]);
}
extern "C" void __stdcall main_argv(const int i, char* buf)
{
	memcpy(buf, _argv[i], strlen(_argv[i])*sizeof(char));
}
#if defined(WIN32)
extern "C" int __stdcall fs_getdcwd(const int i, char* buf, const int buflen)
{
	return _getdcwd(i, buf, buflen)?1:0;
}
extern "C" int __stdcall fs_fullpath(char* buf, const char* s, const int buflen)
{
	return _fullpath(buf, s, buflen)?1:0;
}
extern "C" int __stdcall fs_getfileattributes(const char* s)
{
	return GetFileAttributes(s);
}
extern "C" char* __stdcall fs_gettruename(char* s)
{
	WIN32_FIND_DATA fd;
	HANDLE h = FindFirstFile(s, &fd);
	if (h == INVALID_HANDLE_VALUE) return NULL;
	FindClose(h);
	return fd.cFileName;
}
extern "C" int __stdcall fs_access(const char* s, const int mode)
{
	return _access(s, mode);
}
extern "C" __int64 __stdcall fs_getfiletime(const char* s)
{
	FILETIME fileTime;
	HANDLE file = CreateFile(s, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	int res = GetFileTime(file, NULL, NULL, &fileTime);
	CloseHandle(file);
	if (res == 0) return 0;
	return filetimeToJavaTime(&fileTime);
}
extern "C" __int64 __stdcall fs_stat_size(const char* s)
{
	struct _stati64 buf;
	int res = _stati64(s, &buf);
	if (res != 0) return 0;
	return buf.st_size;
}
#else
extern "C" int __stdcall fs_getdcwd(const int i, char* buf, const int buflen)
{
  // TODO.
  return 0;
}
extern "C" int __stdcall fs_fullpath(char* buf, const char* s, const int buflen)
{
  // TODO.
  return 0;
}
extern "C" int __stdcall fs_getfileattributes(const char* s)
{
  // TODO.
  return 0;
}
extern "C" char* __stdcall fs_gettruename(char* s)
{
  // TODO.
  return s;
}
extern "C" int __stdcall fs_access(const char* s, const int mode)
{
  return access(s, mode);
}
extern "C" __int64 __stdcall fs_getfiletime(const char* s)
{
  struct stat buf;
  int res = stat(s, &buf);
  if (res != 0) return 0;
  return ((__int64)buf.st_mtime) * 1000L;
}
extern "C" __int64 __stdcall fs_stat_size(const char* s)
{
  struct stat buf;
  int res = stat(s, &buf);
  if (res != 0) return 0;
  return buf.st_size;
}
#endif
extern "C" int __stdcall fs_remove(const char* s)
{
	return remove(s);
}
#if defined(WIN32)
extern "C" DIR * __stdcall fs_opendir(const char* s)
{
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
extern "C" struct dirent * __stdcall fs_readdir(DIR *dir)
{
	if (dir->handle == INVALID_HANDLE_VALUE) return NULL;
	strcpy(dir->dirent.d_name, dir->find_data.cFileName);
	if (!FindNextFile(dir->handle, &dir->find_data)) {
		if (GetLastError() == ERROR_INVALID_HANDLE) return NULL;
		FindClose(dir->handle);
		dir->handle = INVALID_HANDLE_VALUE;
	}
	return &dir->dirent;
}
extern "C" int __stdcall fs_closedir(DIR *dir)
{
	if (dir->handle != INVALID_HANDLE_VALUE) {
		if (!FindClose(dir->handle)) return -1;
		dir->handle = INVALID_HANDLE_VALUE;
	}
	free(dir->path); free(dir);
	return 0;
}
extern "C" int __stdcall fs_setfiletime(const char* s, const __int64 time)
{
	FILETIME fileTime;
	javaTimeToFiletime(time, &fileTime);
	HANDLE file = CreateFile(s, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	int res = SetFileTime(file, NULL, NULL, &fileTime);
	CloseHandle(file);
	return res;
}
extern "C" int __stdcall fs_getlogicaldrives(void)
{
	return GetLogicalDrives();
}
#else
extern "C" DIR * __stdcall fs_opendir(const char* s)
{
	return opendir(s);
}
extern "C" struct dirent * __stdcall fs_readdir(DIR *dir)
{
	return readdir(dir);
}
extern "C" int __stdcall fs_closedir(DIR *dir)
{
	return closedir(dir);
}
extern "C" int __stdcall fs_setfiletime(const char* s, const __int64 time)
{
  // TODO.
  return 0;
}
extern "C" int __stdcall fs_getlogicaldrives(void)
{
  // TODO
  return 0;
}
#endif
extern "C" int __stdcall fs_mkdir(const char* s)
{
	return _mkdir(s);
}
extern "C" int __stdcall fs_rename(const char* s, const char* s1)
{
	return rename(s, s1);
}
extern "C" int __stdcall fs_chmod(const char* s, const int mode)
{
	return _chmod(s, mode);
}
extern "C" void __stdcall yield(void)
{
	Sleep(0);
}
extern "C" void __stdcall msleep(int ms)
{
	Sleep(ms);
}
#if defined(WIN32)
extern "C" HANDLE __stdcall create_thread(LPTHREAD_START_ROUTINE start, LPVOID arg)
{
	DWORD tid;
	return CreateThread(NULL, 0, start, arg, CREATE_SUSPENDED, &tid);
}
extern "C" int __stdcall resume_thread(const HANDLE handle)
{
	return ResumeThread(handle);
}
extern "C" int __stdcall suspend_thread(const HANDLE handle)
{
	return SuspendThread(handle);
}

extern "C" void* __stdcall allocate_stack(const int size)
{
	LPVOID lpvAddr;
	DWORD dwPageSize;
	DWORD dwFinalSize;
	DWORD oldProtect;
	SYSTEM_INFO sSysInfo;
	GetSystemInfo(&sSysInfo);     // populate the system information structure
	
	dwPageSize = sSysInfo.dwPageSize;
	dwFinalSize = ((size / dwPageSize) + 2) * dwPageSize;
	lpvAddr = VirtualAlloc(NULL, dwFinalSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	if (lpvAddr == NULL) {
		fprintf(stderr, "PANIC! Cannot allocate stack!");
		return NULL;
	}
	if (!VirtualProtect(lpvAddr, dwPageSize, PAGE_GUARD | PAGE_READWRITE, &oldProtect)) {
		fprintf(stderr, "PANIC! Cannot create stack guard page!");
	}
	return (void*)((int)lpvAddr+dwFinalSize);
}

extern "C" void __stdcall get_thread_context(HANDLE t, CONTEXT* c)
{
	GetThreadContext(t, c);
}

extern "C" void __stdcall set_thread_context(HANDLE t, CONTEXT* c)
{
	SetThreadContext(t, c);
}

extern "C" HANDLE __stdcall get_current_thread_handle(void)
{
	HANDLE currentProcess = GetCurrentProcess();
	HANDLE targetHandle;
	DuplicateHandle(currentProcess, GetCurrentThread(), currentProcess, &targetHandle, 0, TRUE, DUPLICATE_SAME_ACCESS);
	return targetHandle;
}

typedef struct _Thread {
	CONTEXT registers;
	int thread_switch_enabled;
} Thread;

typedef struct _NativeThread {
	HANDLE thread_handle;
	Thread* currentThread;
} NativeThread;

CRITICAL_SECTION semaphore_init;
CRITICAL_SECTION* p_semaphore_init;

void initSemaphoreLock(void)
{
	InitializeCriticalSection(&semaphore_init);
	p_semaphore_init = &semaphore_init;
}

extern "C" HANDLE __stdcall init_semaphore(void)
{
	HANDLE sema;
	EnterCriticalSection(p_semaphore_init);
	sema = CreateSemaphore(0, 0, 1, 0);
	LeaveCriticalSection(p_semaphore_init);
	return sema;
}

extern "C" int __stdcall wait_for_single_object(HANDLE handle, int time)
{
	return WaitForSingleObject(handle, time);
}

extern "C" int __stdcall release_semaphore(HANDLE semaphore, int a)
{
	return ReleaseSemaphore(semaphore, a, NULL);
}

extern "C" void __stdcall set_current_context(Thread* jthread, const CONTEXT* context)
{
	__asm {
		// set thread block
		mov EDX, jthread
		mov FS:14h, EDX
		// load context into ECX
		mov ECX, context
		// set stack pointer
		mov ESP, [ECX+196]
		// set fp regs
		frstor [ECX+28]
		// push return address
		push [ECX+184]
		// change stack pointer to include return address
		mov [ECX+196], ESP
		// push all GPRs
		push [ECX+176] // eax
		push [ECX+172] // ecx
		push [ECX+168] // edx
		push [ECX+164] // ebx
		push [ECX+196] // esp
		push [ECX+180] // ebp
		push [ECX+160] // esi
		push [ECX+156] // edi
		// push eflags
		push [ECX+192]
		// reenable interrupts
		dec dword ptr [EDX+04h]

		// from this point on, the thread can be preempted again.
		// but all GPRs and EIP are on the thread's stack, so it is safe.

		// restore eflags
		popfd
		// restore all GPRs
		popad
		// return to eip
		ret
	}
}
#else
extern "C" int __stdcall suspend_thread(const pthread_t handle)
{
  printf("Suspending thread %d.\n", handle);
  kill( handle, SIGSTOP );
  return 0;
}
extern "C" pthread_t __stdcall create_thread(void * (*start)(void *), void* arg)
{
  int code;
  pthread_t p;
  code = pthread_create(&p, 0, start, arg); // note: NOT SUSPENDED.
  if (!code) {
    // error occurred while creating the thread.
  }
  printf("Created thread %d.\n", p);
  suspend_thread(p);
  return p;
}
extern "C" void __stdcall init_thread(const pthread_t handle)
{
  printf("Thread %d (%d) finished initialization.\n", handle, pthread_self());
}
extern "C" int __stdcall resume_thread(const pthread_t handle)
{
  printf("Resuming thread %d (%d).\n", handle, pthread_self());
  ptrace( PTRACE_CONT, handle, (caddr_t)1, SIGSTOP );
  return 0;
}

extern "C" void* __stdcall allocate_stack(const int size)
{
  // TODO
  return calloc(size, 1);
}

static inline int get_debug_reg( int pid, int num, DWORD *data )
{
    int res = ptrace( PTRACE_PEEKUSER, pid, DR_OFFSET(num), 0 );
    if ((res == -1) && errno)
    {
      // TODO
        return -1;
    }
    *data = res;
    return 0;
}

extern "C" void __stdcall get_thread_context(pthread_t pid, CONTEXT* context)
{
  int flags = context->ContextFlags;
  if (flags & CONTEXT_FULL) {
    struct kernel_user_regs_struct regs;
    if (ptrace( PTRACE_GETREGS, pid, 0, &regs ) == -1) goto error;
    if (flags & CONTEXT_INTEGER) {
      context->Eax = regs.eax;
      context->Ebx = regs.ebx;
      context->Ecx = regs.ecx;
      context->Edx = regs.edx;
      context->Esi = regs.esi;
      context->Edi = regs.edi;
    }
    if (flags & CONTEXT_CONTROL) {
      context->Ebp    = regs.ebp;
      context->Esp    = regs.esp;
      context->Eip    = regs.eip;
      context->SegCs  = regs.cs;
      context->SegSs  = regs.ss;
      context->EFlags = regs.eflags;
    }
    if (flags & CONTEXT_SEGMENTS) {
      context->SegDs = regs.ds;
      context->SegEs = regs.es;
      context->SegFs = regs.fs;
      context->SegGs = regs.gs;
    }
  }
  if (flags & CONTEXT_DEBUG_REGISTERS) {
    if (get_debug_reg( pid, 0, &context->Dr0 ) == -1) goto error;
    if (get_debug_reg( pid, 1, &context->Dr1 ) == -1) goto error;
    if (get_debug_reg( pid, 2, &context->Dr2 ) == -1) goto error;
    if (get_debug_reg( pid, 3, &context->Dr3 ) == -1) goto error;
    if (get_debug_reg( pid, 6, &context->Dr6 ) == -1) goto error;
    if (get_debug_reg( pid, 7, &context->Dr7 ) == -1) goto error;
  }
  if (flags & CONTEXT_FLOATING_POINT) {
    /* we can use context->FloatSave directly as it is using the */
    /* correct structure (the same as fsave/frstor) */
    if (ptrace( PTRACE_GETFPREGS, pid, 0, &context->FloatSave ) == -1) goto error;
    context->FloatSave.Cr0NpxState = 0;  /* FIXME */
  }
  return;
 error:
  // TODO: error condition
  return;
}

extern "C" void __stdcall set_thread_context(pthread_t pid, CONTEXT* context)
{
  int flags = context->ContextFlags;
  if (flags & CONTEXT_FULL) {
    struct kernel_user_regs_struct regs;
    
    /* need to preserve some registers (at a minimum orig_eax must always be preserved) */
    if (ptrace( PTRACE_GETREGS, pid, 0, &regs ) == -1) goto error;
    
    if (flags & CONTEXT_INTEGER) {
      regs.eax = context->Eax;
      regs.ebx = context->Ebx;
      regs.ecx = context->Ecx;
      regs.edx = context->Edx;
      regs.esi = context->Esi;
      regs.edi = context->Edi;
    }
    if (flags & CONTEXT_CONTROL) {
      regs.ebp = context->Ebp;
      regs.esp = context->Esp;
      regs.eip = context->Eip;
      regs.cs  = context->SegCs;
      regs.ss  = context->SegSs;
      regs.eflags = context->EFlags;
    }
    if (flags & CONTEXT_SEGMENTS) {
      regs.ds = context->SegDs;
      regs.es = context->SegEs;
      regs.fs = context->SegFs;
      regs.gs = context->SegGs;
    }
    if (ptrace( PTRACE_SETREGS, pid, 0, &regs ) == -1) goto error;
  }
  if (flags & CONTEXT_DEBUG_REGISTERS) {
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(0), context->Dr0 ) == -1) goto error;
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(1), context->Dr1 ) == -1) goto error;
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(2), context->Dr2 ) == -1) goto error;
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(3), context->Dr3 ) == -1) goto error;
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(6), context->Dr6 ) == -1) goto error;
    if (ptrace( PTRACE_POKEUSER, pid, DR_OFFSET(7), context->Dr7 ) == -1) goto error;
  }
  if (flags & CONTEXT_FLOATING_POINT) {
    /* we can use context->FloatSave directly as it is using the */
    /* correct structure (the same as fsave/frstor) */
    if (ptrace( PTRACE_SETFPREGS, pid, 0, &context->FloatSave ) == -1) goto error;
  }
  return;
 error:
  // TODO: error
  return;
}

extern "C" pthread_t __stdcall get_current_thread_handle(void)
{
  return pthread_self();
}

typedef struct _Thread {
	CONTEXT registers;
	int thread_switch_enabled;
} Thread;

typedef struct _NativeThread {
	pthread_t thread_handle;
	Thread* currentThread;
} NativeThread;

//CRITICAL_SECTION semaphore_init;
//CRITICAL_SECTION* p_semaphore_init;

void initSemaphoreLock(void)
{
  // TODO.
}

extern "C" int __stdcall init_semaphore(void)
{
  // TODO.
  return 0;
}

extern "C" int __stdcall wait_for_single_object(int handle, int time)
{
  // TODO.
  return 0;
}

extern "C" int __stdcall release_semaphore(int semaphore, int a)
{
  // TODO.
  return 0;
}

extern "C" void __stdcall set_current_context(Thread* jthread, const CONTEXT* context)
{
  __asm ("
		movl %0, %%edx
		movl %%edx, %%fs:20
		movl %1, %%ecx
		movl 196(%%ecx), %%esp
		frstor 28(%%ecx)
		pushl 184(%%ecx)
		movl %%esp, 196(%%ecx)
		pushl 176(%%ecx)
		pushl 172(%%ecx)
		pushl 168(%%ecx)
		pushl 164(%%ecx)
		pushl 196(%%ecx)
		pushl 180(%%ecx)
		pushl 160(%%ecx)
		pushl 156(%%ecx)
		pushl 192(%%ecx)
		decl 4(%%edx)
		popf
		popa
		ret"
                :
	        :"r"(jthread), "r"(context)
		);
}
#endif

// native.c : Native method implementations
//

#include "StdAfx.h"

#define ARRAY_LENGTH_OFFSET -12

extern "C" void __stdcall debugwmsg(const unsigned short* s)
{
	int* length_loc = (int*)((int)s + ARRAY_LENGTH_OFFSET);
	int length = *length_loc;
#if defined(WIN32)
	unsigned short* temp = (unsigned short*)malloc((length+1)*sizeof(unsigned short));
	memcpy(temp, s, length*sizeof(unsigned short));
	temp[length] = 0;
	_putws(temp);
	free(temp);
#else
    // TODO: actually write wide characters
	while (--length >= 0) {
	  unsigned short c = *s;
	  putchar((char)c);
	  ++s;
	}
	putchar('\n');
#endif
	fflush(stdout);
}

extern "C" void __stdcall debugmsg(const char* s)
{
	puts(s);
	fflush(stdout);
}

extern "C" void* __stdcall syscalloc(const int size)
{
	void* p = calloc(1, size);
	//printf("Allocated %d bytes at 0x%08x\n", size, p);
	return p;
}

extern "C" void __stdcall die(const int code)
{
	fflush(stdout);
	fflush(stderr);
	exit(code);
}

#if defined(WIN32)
// Windows time functions
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
// Generic unix time functions.
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
#elif defined(linux)
extern "C" int __stdcall console_available(void)
{
	return 0; // TODO
}
#else
#error System type not supported.
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
#elif defined(linux)
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
#else
#error System type not supported.
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
#elif defined(linux)
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
#else
#error System type not supported.
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
extern "C" int __stdcall init_thread(void)
{
	// nothing to do here.
	return GetCurrentThreadId();
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

extern "C" void __stdcall timer_tick(LPVOID arg, DWORD lo, DWORD hi)
{
  // get current Java thread
  Thread* java_thread;
	__asm {
		// set thread block
		mov EDX, FS:14h
		mov java_thread, EDX
	}
  // check if thread switch is ok
  if (java_thread->thread_switch_enabled != 0) {
    return;
  }
  
  NativeThread* native_thread = java_thread->native_thread;

  // disable thread switch.
  ++java_thread->thread_switch_enabled;

  // call the threadSwitch method.
  threadSwitch(native_thread);
}

extern "C" void __stdcall set_interval_timer(int type, int ms)
{
    HANDLE hTimer = NULL;
    LARGE_INTEGER liDueTime;

    liDueTime.QuadPart=-10000*ms;

    // Create a waitable timer.
    hTimer = CreateWaitableTimer(NULL, TRUE, "WaitableTimer");
    if (!hTimer)
    {
        printf("CreateWaitableTimer failed (%d)\n", GetLastError());
        return;
    }

    // Set a timer to wait.
    if (!SetWaitableTimer(hTimer, &liDueTime, ms, timer_tick, NULL, 0))
    {
        printf("SetWaitableTimer failed (%d)\n", GetLastError());
        return;
    }
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
#elif defined(linux)
extern "C" int __stdcall suspend_thread(const int pid)
{
  //printf("Suspending thread %d.\n", pid);
#if defined(USE_CLONE)
  kill(pid, SIGSTOP);
#else
  pthread_kill(pid, SIGSTOP);
#endif
  return 0;
}

int thread_start_trampoline(void *t)
{
#if defined(USE_CLONE)
  const int pid = getpid();
#else
  const int pid = pthread_self();
#endif
  //printf("Thread %d started, suspending self.\n", pid);
  void* arg = *((void**)t);
  int (*start)(void *) = (int (*)(void*)) ((void**)t)[1];
  // write our pid in temp[0]
  *((int*)t) = getpid();
  // write '0' in temp[1], signaling parent that we have finished
  // initialization.
  ((void**)t)[1] = 0;
  // wait until parent receives notification and notifies us back.
  while (!((void**)t)[1]) sched_yield();
  // free temp array
  free(t);

#if 0
  int foo;
  sigset_t set; sigemptyset(&set); sigaddset(&set, SIGCONT); sigaddset(&set, SIGKILL);
  sigwait(&set, &foo);
  if (foo == SIGKILL) {
    printf("Thread %d killed while waiting.\n", pid);
    return 0;
  }
  //printf("Thread %d resumed (signal=%d)\n", pid, foo);
#endif
  //printf("Thread %d calling function at 0x%08x, arg 0x%08x.\n", pid, start, arg);
  return start(arg);
}

extern "C" int __stdcall create_thread(int (*start)(void *), void* arg)
{
  unsigned long int pid;
  void** temp = (void**)malloc(sizeof(start) + sizeof(arg));
  temp[0] = arg;
  temp[1] = (void*)start;
#if defined(USE_CLONE)
  void* child_stack = calloc(1, 65536);
  pid = clone(thread_start_trampoline, child_stack, CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND, temp);
#else
  pthread_create(&pid, 0, (void* (*)(void *))thread_start_trampoline, temp);
#endif
  //printf("Created thread %d.\n", pid);

  // wait for child to start.
  while (temp[1]) sched_yield();

  // child has started, suspend it.
#if defined(USE_CLONE)
  kill(pid, SIGSTOP);
#else
  pthread_kill(pid, SIGSTOP);
#endif

  // mark child to continue, once it is resumed.
  temp[1] = (void*)start;

  return pid;
}

/* We don't want to include the kernel header.  So duplicate the
   information.  */

/* Structure passed on `modify_ldt' call.  */
struct modify_ldt_ldt_s
{
  unsigned int entry_number;
  unsigned long int base_addr;
  unsigned int limit;
  unsigned int seg_32bit:1;
  unsigned int contents:2;
  unsigned int read_exec_only:1;
  unsigned int limit_in_pages:1;
  unsigned int seg_not_present:1;
  unsigned int useable:1;
  unsigned int empty:25;
};

_syscall3( int, modify_ldt, int, func, void *, ptr, unsigned long, bytecount )

/* Initialize the thread-unique value.  */
#define INIT_THREAD_SELF(descr, descrsize, nr) \
{                                                                             \
  struct modify_ldt_ldt_s ldt_entry =                                         \
    { nr, (unsigned long int) descr, descrsize, 1, 0, 0, 0, 0, 1, 0 };        \
  if (modify_ldt (1, &ldt_entry, sizeof (ldt_entry)) != 0)                    \
    abort ();                                                                 \
  __asm__ __volatile__ ("movw %w0, %%fs" : : "q" (nr * 8 + 7));               \
}

static int current_id = 16;

extern "C" int __stdcall init_thread()
{
  int my_id;
  __asm__ __volatile__
    (
     "nop
uphere:
      movl %2, %%eax
      movl %%eax, %%ebx
      inc %%ebx
      lock cmpxchgl %%ebx, %1
      jne uphere
      movl %%ebx, %0
      " :"=r"(my_id), "=m"(current_id)
        :"m"(current_id)
        :"%eax"
    );
  void* descr = calloc(1, 1024);
  INIT_THREAD_SELF(descr, 1*1024, my_id);
  //printf("Thread %d finished initialization, pid=%d, id=%d.\n", pthread_self(), getpid(), my_id);
  return getpid();
}
extern "C" int __stdcall resume_thread(const int pid)
{
  //printf("Resuming thread %d.\n", pid);
  //ptrace( PTRACE_CONT, pid, (caddr_t)1, SIGSTOP );
#if defined(USE_CLONE)
  kill(pid, SIGCONT);
#else
  pthread_kill(pid, SIGCONT);
#endif
  return 0;
}

extern "C" void* __stdcall allocate_stack(const int size)
{
  // TODO
  void* p = calloc(sizeof(char), size);
  //printf("Allocating stack at 0x%08x of size %d.\n", p, size);
  return (char*)p+size;
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

extern "C" void __stdcall get_thread_context(const int pid, CONTEXT* context)
{
  //printf("Getting thread context for pid %d.\n", pid);
  if (ptrace(PTRACE_ATTACH, pid, 0, 0) == -1) {
    printf("Attempt to attach to pid %d failed! %s errno %d\n", pid, strerror(errno), errno);
    return;
  }
  int status;
  waitpid(pid, &status, WUNTRACED);
  if (WIFSTOPPED(status)) {
    //printf("Child is stopped, signal=%d.\n", WSTOPSIG(status));
  }

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
  goto cleanup;
 error:
  // TODO: error condition
  printf("Error occurred while getting context of thread %d: %s errno %d.\n", pid, strerror(errno), errno);
  
 cleanup:
  if (ptrace(PTRACE_DETACH, pid, 0, SIGSTOP) == -1) {
    printf("Attempt to detach from pid %d failed! %s errno %d\n", pid, strerror(errno), errno);
    return;
  }
  return;
}

extern "C" void __stdcall set_thread_context(int pid, CONTEXT* context)
{
  //printf("Setting thread context for pid %d, ip=0x%08x, sp=0x%08x\n", pid, context->Eip, context->Esp);
  if (ptrace(PTRACE_ATTACH, pid, 0, 0) == -1) {
    printf("Attempt to attach to pid %d failed! %s errno %d\n", pid, strerror(errno), errno);
    return;
  }
  int status;
  waitpid(pid, &status, WUNTRACED);
  if (WIFSTOPPED(status)) {
    //printf("Child is stopped, signal=%d.\n", WSTOPSIG(status));
  }

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
  goto cleanup;
 error:
  // TODO: error condition
  printf("Error occurred while setting context of thread %d: %s errno %d.\n", pid, strerror(errno), errno);
  
 cleanup:
  if (ptrace(PTRACE_DETACH, pid, 0, SIGSTOP) == -1) {
    printf("Attempt to detach from pid %d failed! %s errno %d\n", pid, strerror(errno), errno);
    return;
  }
  return;
}

extern "C" int __stdcall get_current_thread_handle(void)
{
#if defined(USE_CLONE)
  return getpid();
#else
  return pthread_self();
#endif
}

sem_t semaphore_init;
sem_t *p_semaphore_init;

void initSemaphoreLock(void)
{
    sem_init(&semaphore_init, 0, 1);
    p_semaphore_init = &semaphore_init;
}

extern "C" int __stdcall init_semaphore(void)
{
    sem_t* n_sem = (sem_t*)malloc(sizeof(sem_t));
    sem_wait(p_semaphore_init);
    sem_init(n_sem, 0, 0);
    sem_post(p_semaphore_init);
    //printf("Initialized new semaphore %x.\n", (int)n_sem);
    return (int)n_sem;
}

#if !defined(timersub)
# define timersub(a, b, result)                                               \
  do {                                                                        \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                             \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                          \
    if ((result)->tv_usec < 0) {                                              \
      --(result)->tv_sec;                                                     \
      (result)->tv_usec += 1000000;                                           \
    }                                                                         \
  } while (0)
#endif

#define WAIT_TIMEOUT 0x00000102
extern "C" int __stdcall wait_for_single_object(int handle, int time)
{
    //printf("Thread %d: Waiting on semaphore %x for %d ms.\n", pthread_self(), (int)handle, time);
    if (sem_trywait((sem_t*)handle) == 0) return 0;
    //printf("Thread %d: Waiting on semaphore %x initially failed, looping.\n", pthread_self(), (int)handle);
    fflush(stdout);
    struct timeval tv1;
    gettimeofday(&tv1, 0);
    for (;;) {
	struct timeval tv2;
	long diff;
	sched_yield();
	if (sem_trywait((sem_t*)handle) == 0) {
	    //printf("Thread %d: Waiting on semaphore %x succeeded.\n", pthread_self(), (int)handle);
	  fflush(stdout);
	  return 0;
	}
	gettimeofday(&tv2, 0);
	timersub(&tv2, &tv1, &tv2);
	diff = tv2.tv_sec*1000000+tv2.tv_usec*1000;
	//printf("Thread %d: Waiting on semaphore %x failed again, time passed %d us.\n", pthread_self(), (int)handle, diff);
	fflush(stdout);
	if (diff > time*1000) return WAIT_TIMEOUT;
    }
}

extern "C" int __stdcall release_semaphore(int semaphore, int a)
{
    //printf("Thread %d: Releasing semaphore %x %d times.\n", pthread_self(), (int)semaphore, a);
    int v = 0;
    while (--a >= 0)
	v = sem_post((sem_t*)semaphore);
    return v;
}

extern "C" void __stdcall set_interval_timer(int type, int ms)
{
  //printf("Thread %d: Setting interval timer type %d, %d ms.\n", pthread_self(), (int)type, ms);
  struct itimerval v;
  v.it_interval.tv_sec = v.it_value.tv_sec = ms / 1000;
  v.it_interval.tv_usec = v.it_value.tv_usec = (ms % 1000) * 1000;
  setitimer(type, &v, 0);
}

extern "C" void __stdcall set_current_context(Thread* jthread, const CONTEXT* context)
{
#if defined(USE_CLONE)
    int pid = getpid();
#else
    int pid = pthread_self();
#endif
    //printf("Thread %d: switching to jthread 0x%08x, ip=0x%08x, sp=0x%08x\n", pid, jthread, context->Eip, context->Esp);
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
#else
#error System type not supported.
#endif

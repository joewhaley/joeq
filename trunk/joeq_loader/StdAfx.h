// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#if !defined(AFX_STDAFX_H__B24BD5A2_E719_11D4_882B_00008632F0B0__INCLUDED_)
#define AFX_STDAFX_H__B24BD5A2_E719_11D4_882B_00008632F0B0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(WIN32)

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#define _WIN32_WINNT 0x0400		// Include SetWaitableTimer

#include "windows.h"
#include <io.h>
#include <direct.h>

#if defined(__BORLANDC__)
#include <dos.h>
#endif

#if !defined(_umask)
#define _umask umask
#endif

#else

#if !defined(__stdcall)
#define __stdcall __attribute__((stdcall))
#endif
#define _umask umask
#define __int64 int64_t
#define _open open
#define _read read
#define _write write
#define _commit fsync
#define _lseeki64 lseek
#define _close close
#define _mkdir(s) mkdir((s),0777)
#define _chmod chmod
#define Sleep(ms) usleep(1000*(ms))

#include <wchar.h>
#include <sys/timeb.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

#endif

#if defined(linux)
#include <linux/unistd.h>
#include "context.h"
#include <sys/ptrace.h>
#include <pthread.h>
#include <signal.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <semaphore.h>
#endif

#include "native.h"
#include "handler.h"

typedef struct _Thread {
	CONTEXT* registers;
	int thread_switch_enabled;
        struct _NativeThread* native_thread;
} Thread;

typedef struct _NativeThread {
	int thread_handle;
	Thread* currentThread;
        int pid;
} NativeThread;

extern "C" void __stdcall trap_handler(void*, int);
extern "C" void __stdcall ctrl_break_handler();
extern "C" void __stdcall threadSwitch(void*);

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_STDAFX_H__B24BD5A2_E719_11D4_882B_00008632F0B0__INCLUDED_)

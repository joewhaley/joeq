// joeq.cpp : Defines the entry point for the console application.
//

#include "StdAfx.h"

int _argc;
char **_argv;

extern "C" void __stdcall entry();
extern "C" void __stdcall ctrl_break_handler();
extern void initSemaphoreLock(void);

#if defined(WIN32)
BOOL WINAPI windows_break_handler(DWORD dwCtrlType)
{
	if (dwCtrlType == CTRL_BREAK_EVENT) {
		ctrl_break_handler();
		return 1;
	}
	return 0;
}
#endif

#if defined(linux)
char FS_AREA[4096]; // used as storage off of the FS register.
_syscall3( int, modify_ldt, int, func, void *, ptr, unsigned long, bytecount )
  //int sys_modify_ldt(int func, void *ptr, unsigned long bytecount);
#endif

int main(int argc, char* argv[])
{
	// clear umask
	_umask(0);

	// initialize argc and argv
	_argc = argc-1;
	_argv = argv+1;

#if defined(WIN32)
	// install hardware exception handler.
	HandlerRegistrationRecord er, *erp = &er;
	er.previous = NULL;
	er.handler = hardwareExceptionHandler;
	_asm mov eax,[erp]
	_asm mov fs:[0],eax // point first word of thread control block to exception handler registration chain

	// install ctrl-break handler
	SetConsoleCtrlHandler(windows_break_handler, TRUE);

	initSemaphoreLock();
#endif

#if 0
	{
	  modify_ldt_ldt_s ldt_entry;
	  printf("Setting local descriptor table to allow use of FS register...\n");
	  ldt_entry.entry_number = 17;
	  ldt_entry.base_addr = (long int)FS_AREA;
	  ldt_entry.limit = sizeof(FS_AREA);
	  ldt_entry.seg_32bit = 1;
	  ldt_entry.contents = MODIFY_LDT_CONTENTS_DATA;
	  ldt_entry.read_exec_only = 0;
	  ldt_entry.limit_in_pages = 0;
	  ldt_entry.seg_not_present = 0;
	  ldt_entry.useable = 1;
	  
	  int r = modify_ldt(1, &ldt_entry, sizeof(ldt_entry));
	  printf("Result = %d\n", r);

	  printf("Setting up FS segment...");
#define LDT_SEL(idx)  ((idx) << 3 | 1 << 2 | 3)
	  __asm__ __volatile__(
			       "movl %0,%%eax; movw %%ax, %%fs" : : "i" LDT_SEL(17)
			       );
	  printf("done.\n");
	}
#endif
	Setup_LDT_Keeper();
	int initialized = 1;
	__asm (" movl %%fs:4, %0"
	       :
	       :"r"(initialized)
	       );

	printf("branching to entrypoint at location 0x%08x\n", entry);
	fflush(stdout);

#if defined(WIN32)
	__asm {
		// set it up so FP = 0, so we know the stack top.
		push EBP
		xor EBP, EBP

		// jump into joeq
		call entry

		// restore FP, so chkesp doesn't complain
		pop EBP
	}
#else
	__asm (
	       "push %%ebp
		xor %%ebp, %%ebp
		call entry
		pop %%ebp"
	       :
	       :
	       :"%ebp"
	       );
#endif

	// unreachable.
	return 0;
}

// joeq.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

int _argc;
char **_argv;

extern "C" void __stdcall entry();
extern "C" void __stdcall ctrl_break_handler();
extern void initSemaphoreLock(void);

BOOL WINAPI windows_break_handler(DWORD dwCtrlType)
{
	if (dwCtrlType == CTRL_BREAK_EVENT) {
		ctrl_break_handler();
		return 1;
	}
	return 0;
}


int main(int argc, char* argv[])
{
	// clear umask
	_umask(0);

	// initialize argc and argv
	_argc = argc-1;
	_argv = argv+1;

	// install hardware exception handler.
	HandlerRegistrationRecord er, *erp = &er;
	er.previous = NULL;
	er.handler = hardwareExceptionHandler;
	_asm mov eax,[erp]
	_asm mov fs:[0],eax // point first word of thread control block to exception handler registration chain

	// install ctrl-break handler
	SetConsoleCtrlHandler(windows_break_handler, TRUE);

	initSemaphoreLock();

	printf("branching to entrypoint at location 0x%08x\n", entry);
	fflush(stdout);

	__asm {
		// set it up so FP = 0, so we know the stack top.
		xor EBP, EBP
		// go there!
		call entry
	}

	// unreachable.
	return 0;
}

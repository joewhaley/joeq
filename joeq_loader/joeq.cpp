// joeq.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

int _argc;
char **_argv;

extern "C" void __stdcall entry();

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

	printf("branching to entrypoint at location 0x%08x\n", entry);
	fflush(stdout);

	entry();

	fflush(stdout);
	fflush(stderr);
	return 0;
}


#include "StdAfx.h"

#if defined(WIN32)
extern "C" void __stdcall trap_handler(void*, int);
//extern "C" void* trap_handler;

EXCEPTION_DISPOSITION hardwareExceptionHandler(EXCEPTION_RECORD *exceptionRecord,
											   void *establisherFrame,
											   CONTEXT *contextRecord,
											   void *dispatcherContext)
{
	void *eip = (void *)contextRecord->Eip;
	int ex_code = exceptionRecord->ExceptionCode;
	int java_ex_code;
	switch (ex_code) {
	case EXCEPTION_ACCESS_VIOLATION: // null pointer exception
		// int 5 seems to create an access violation, for some reason.
		if (*((int*)(((int)eip)-2)) == 0x05cd0272) java_ex_code = 1;
		else java_ex_code = 0;
		break;
	case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: // array bounds exception
		java_ex_code = 1;
		break;
	case EXCEPTION_INT_DIVIDE_BY_ZERO: // divide by zero exception
		java_ex_code = 2;
		break;
	case EXCEPTION_STACK_OVERFLOW:
		java_ex_code = 3;
		break;
	default:
		java_ex_code = -1;
		break;
	}

	// push arguments
	int *esp = (int *)contextRecord->Esp;
	*--esp = java_ex_code;
	*--esp = (int)eip;
	contextRecord->Esp = (int)esp;

	// resume execution at java trap handler
	contextRecord->Eip = (int)trap_handler;
	return ExceptionContinueExecution;
}
#endif

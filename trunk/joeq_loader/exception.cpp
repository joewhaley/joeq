
#include "StdAfx.h"

extern "C" void __stdcall trap_handler(void*, int);
extern "C" void __stdcall ctrl_break_handler();
extern "C" void __stdcall threadSwitch(void*);

#if defined(WIN32)

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

BOOL WINAPI windows_break_handler(DWORD dwCtrlType)
{
	if (dwCtrlType == CTRL_BREAK_EVENT) {
		ctrl_break_handler();
		return 1;
	}
	return 0;
}

void installSignalHandler(void)
{
	// install hardware exception handler.
	HandlerRegistrationRecord er, *erp = &er;
	er.previous = NULL;
	er.handler = hardwareExceptionHandler;
	_asm mov eax,[erp]
	_asm mov fs:[0],eax // point first word of thread control block to exception handler registration chain

	// install ctrl-break handler
	SetConsoleCtrlHandler(windows_break_handler, TRUE);
}

#else

void hardwareExceptionHandler(int signo, siginfo_t *si, void *context)
{
  // magic to get sigcontext!
  sigcontext *sc = (sigcontext *)((char *)context+5*4);
  void *eip = (void *)sc->eip;
  int ex_code = signo;
  int java_ex_code;
  switch (ex_code) {
  case SIGSEGV: // null pointer exception
    // int 5 seems to create an access violation, for some reason.
    // on linux, eip points AFTER the instruction, rather than before.
    if (*((int*)(((int)eip)-4)) == 0x05cd0272) java_ex_code = 1;
    else java_ex_code = 0;
    break;
  case SIGFPE: // divide by zero exception
    java_ex_code = 2;
    break;
  case SIGTRAP: // stack overflow
    java_ex_code = 3;
    break;
  default:
    java_ex_code = -1;
    break;
  }

  // push arguments
  int *esp = (int *)sc->esp;
  *--esp = java_ex_code;
  *--esp = (int)eip;
  sc->esp = (int)esp;

  // resume execution at java trap handler
  sc->eip = (int)trap_handler;
  return;
}

void copyFromSigcontext(CONTEXT* c, sigcontext* sc)
{
  c->Eax = sc->eax;
  c->Ebx = sc->ebx;
  c->Ecx = sc->ecx;
  c->Edx = sc->edx;
  c->Esi = sc->esi;
  c->Edi = sc->edi;
  c->Ebp = sc->ebp;
  c->Esp = sc->esp;
  c->Eip = sc->eip;
  c->SegCs = sc->cs;
  c->SegSs = sc->ss;
  c->EFlags = sc->eflags;
  memcpy(&c->FloatSave, sc->fpstate, sizeof(FLOATING_SAVE_AREA));
  //sc->fpstate->magic = 0xffff; // regular FPU data only
}

void softwareSignalHandler(int signo, siginfo_t *si, void *context)
{
  //printf("PID %d received tick.\n", getpid());

  // get current Java thread
  Thread* java_thread;
  __asm ("movl %%fs:20, %0":"=r"(java_thread));
  // check if thread switch is ok
  if (java_thread->thread_switch_enabled != 0) {
    //printf("Java thread 0x%08x: thread switch not enabled (%d)\n", java_thread, java_thread->thread_switch_enabled);
    return;
  }
  
  NativeThread* native_thread = java_thread->native_thread;

  // magic to get sigcontext!
  sigcontext *sc = (sigcontext *)((char *)context+5*4);

  //printf("Java thread 0x%08x: thread switch enabled (%d) eip=0x%08x esp=0x%08x\n", java_thread, java_thread->thread_switch_enabled, sc->eip, sc->esp);

  // simulate a call to the threadSwitch method.
  int *esp = (int *)sc->esp;
  *--esp = (int)native_thread;
  *--esp = (int)sc->eip;
  sc->esp = (int)esp;
  sc->eip = (int)threadSwitch;
  
  copyFromSigcontext(java_thread->registers, sc);

  // disable thread switch.
  ++java_thread->thread_switch_enabled;

  //printf("Java thread 0x%08x: calling threadSwitch...\n");

  return;
}

void installSignalHandler(void)
{
  // install a stack for the hardware trap handler
  stack_t stack;
  memset(&stack, 0, sizeof stack);
  stack.ss_sp = malloc(SIGSTKSZ);
  stack.ss_size = SIGSTKSZ;
  //printf("Installing hardware trap signal handler stack.\n");
  if (sigaltstack(&stack, 0)) {
    // TODO: error.
    printf("Error installing hardware trap signal handler stack %d.\n", errno);
    return;
  }
  
  // install hardware trap signal handler
  struct sigaction action;
  memset(&action, 0, sizeof action);
  action.sa_sigaction = &hardwareExceptionHandler;

  // mask all signals from reaching the signal handler while the signal
  // handler is running
  //printf("Filling hardware trap signal handler set.\n");
  if (sigfillset(&(action.sa_mask))) {
    // TODO: error.
    printf("Error filling hardware trap signal handler set %d.\n", errno);
    return;
  }
#if 0
  // ignore the SIGSTOP/SIGCONT signals; they are used to stop and restart
  // native threads.
  if (sigdelset(&(action.sa_mask), SIGSTOP)) {
    // TODO: error.
    printf("Error deleting from hardware trap signal handler set %d\n", errno);
    return;
  }
  if (sigdelset(&(action.sa_mask), SIGCONT)) {
    // TODO: error.
    printf("Error deleting from hardware trap signal handler set %d\n", errno);
    return;
  }
#endif
  action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
  //printf("Setting hardware trap signal handler.\n");
  if (sigaction(SIGSEGV, &action, 0)) {
    // TODO: error.
    printf("Error setting hardware trap signal handler %d\n", errno);
    return;
  }
  //printf("Setting hardware trap signal handler.\n");
  if (sigaction(SIGFPE, &action, 0)) {
    // TODO: error.
    printf("Error setting hardware trap signal handler %d\n", errno);
    return;
  }
  //printf("Setting hardware trap signal handler.\n");
  if (sigaction(SIGTRAP, &action, 0)) {
    // TODO: error.
    printf("Error setting hardware trap signal handler %d\n", errno);
    return;
  }

  // install software signal handler
  action.sa_sigaction = &softwareSignalHandler;
  //printf("Setting software signal handler.\n");
  if (sigaction(SIGVTALRM, &action, 0)) {
    // TODO: error.
    printf("Error setting software signal handler %d\n", errno);
    return;
  }

}

#endif

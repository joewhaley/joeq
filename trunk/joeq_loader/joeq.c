// joeq.c : Defines the entry point for the console application.
//

#include "StdAfx.h"

int joeq_argc;
char **joeq_argv;

void __stdcall entry(void);
extern void installSignalHandler(void);
extern void initSemaphoreLock(void);

extern void* joeq_code_startaddress;
extern void* joeq_data_startaddress;

int main(int argc, char* argv[])
{
    // clear umask
    _umask(0);

    // initialize argc and argv
    joeq_argc = argc-1;
    joeq_argv = argv+1;

#if defined(WIN32)
    // install hardware exception handler.
    // NOTE that this must be on the stack and have a lower address than any previous handler.
    // Therefore, it must be done in main().
    {
        HandlerRegistrationRecord er, *erp = &er;
        er.previous = NULL;
        er.handler = hardwareExceptionHandler;
        _asm mov eax,[erp]
        _asm mov fs:[0],eax // point first word of thread control block to exception handler registration chain
    }
#endif

    installSignalHandler();
    initSemaphoreLock();

    printf("Code segment at 0x%p, Data segment at 0x%p\n", &joeq_code_startaddress, &joeq_data_startaddress);
    printf("branching to entrypoint at location 0x%p\n", entry);
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
        "pushl %%ebp
         xor %%ebp, %%ebp
         call entry
         popl %%ebp
            "
            :
            :
            :"%eax","%edx","%ecx","%ebx","%edi","%esi"
            );
#endif

    // unreachable.
    return 0;
}

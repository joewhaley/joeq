
// Win32 hardware exception handler signature (missing from <windows.h>).
//
typedef EXCEPTION_DISPOSITION (*Handler)(EXCEPTION_RECORD *exceptionRecord, void *establisherFrame, CONTEXT *contextRecord, void *dispatcherContext);

// Win32 exception handler registration record (missing from <windows.h>).
//
struct HandlerRegistrationRecord {
	HandlerRegistrationRecord *previous;
	Handler                    handler;
};

extern EXCEPTION_DISPOSITION hardwareExceptionHandler(EXCEPTION_RECORD *exceptionRecord,
													  void *establisherFrame,
													  CONTEXT *contextRecord,
													  void *dispatcherContext);
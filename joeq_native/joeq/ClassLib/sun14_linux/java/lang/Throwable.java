/*
 * Throwable.java
 *
 * Created on January 29, 2000, 10:16 AM
 *
 */

package ClassLib.sun14_linux.java.lang;

import Allocator.CodeAllocator;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Run_Time.ExceptionDeliverer;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Throwable {
    
    private java.lang.Object backtrace;
    
    // native method implementations
    private int getStackTraceDepth() {
        ExceptionDeliverer.StackFrame backtrace = (ExceptionDeliverer.StackFrame)this.backtrace;
        int i=-1;
        while (backtrace != null) { backtrace = backtrace.getNext(); ++i; }
        return i;
    }
    
    private StackTraceElement getStackTraceElement(int i) {
        ExceptionDeliverer.StackFrame backtrace = (ExceptionDeliverer.StackFrame)this.backtrace;
        while (--i >= 0) { backtrace = backtrace.getNext(); }
        java.lang.String declaringClass = "";
        java.lang.String methodName = "";
        java.lang.String fileName = null;
        int lineNumber = -2;
        int/*CodeAddress*/ ip = backtrace.getIP();
        jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
        if (cc != null) {
            jq_Method m = cc.getMethod();
            if (m != null) {
                declaringClass = m.getDeclaringClass().getJDKName();
                methodName = m.getName().toString();
                int code_offset = ip - cc.getEntrypoint();
                if (m != null) {
                    fileName = m.getDeclaringClass().getSourceFile().toString();
                    int bc_index = cc.getBytecodeIndex(ip);
                    lineNumber = m.getLineNumber(bc_index);
                }
            }
        }
        return new StackTraceElement(declaringClass, methodName, fileName, lineNumber);
    }

    public java.lang.Throwable fillInStackTrace() {
        this.backtrace = ExceptionDeliverer.getStackTrace();
        java.lang.Object o = this;
        return (java.lang.Throwable)o;
    }

    public java.lang.Object getBacktraceObject() { return this.backtrace; }
}

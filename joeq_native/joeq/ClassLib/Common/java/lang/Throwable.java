/*
 * Throwable.java
 *
 * Created on January 29, 2000, 10:16 AM
 *
 */

package ClassLib.Common.java.lang;

import Run_Time.ExceptionDeliverer;
import Main.jq;

/*
 * @author  jwhaley
 * @version 
 */
public abstract class Throwable {
    
    private java.lang.Object backtrace;
    
    // native method implementations
    private void printStackTrace0(java.lang.Object s) {
        java.lang.Object backtrace = this.backtrace;
        if (s instanceof java.io.PrintWriter)
            ExceptionDeliverer.printStackTrace(backtrace, (java.io.PrintWriter)s);
        else if (s instanceof java.io.PrintStream)
            ExceptionDeliverer.printStackTrace(backtrace, (java.io.PrintStream)s);
        else
            jq.UNREACHABLE();
    }
    
    public java.lang.Throwable fillInStackTrace() {
        this.backtrace = ExceptionDeliverer.getStackTrace();
        java.lang.Object o = this;
        return (java.lang.Throwable)o;
    }

    public java.lang.Object getBacktraceObject() { return this.backtrace; }
}

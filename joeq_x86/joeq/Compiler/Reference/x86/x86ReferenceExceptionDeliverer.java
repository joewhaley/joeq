/*
 * x86ReferenceExceptionDeliverer.java
 *
 * Created on January 12, 2001, 8:44 AM
 *
 * @author  jwhaley
 * @version 
 */

package Compil3r.Reference.x86;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import jq;

public class x86ReferenceExceptionDeliverer extends ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static final x86ReferenceExceptionDeliverer INSTANCE =
    new x86ReferenceExceptionDeliverer();

    private x86ReferenceExceptionDeliverer() {}

    public final void deliverToStackFrame(jq_CompiledCode cc, Throwable x, int ip, int fp) {
        jq_Method m = cc.getMethod();
        // find new top of stack
        int n_paramwords = m.getParamWords();
        int n_localwords = m.getMaxLocals();
        int sp = fp + ((n_paramwords-n_localwords)<<2) - 4;
        if (TRACE) SystemInterface.debugmsg("poking exception object "+jq.hex8(Unsafe.addressOf(x))+" into location "+jq.hex8(sp));
        // push exception object there
        Unsafe.poke4(sp, Unsafe.addressOf(x));
        // branch!
        Unsafe.switchRegisterState(ip, fp, sp, 0);
    }

}

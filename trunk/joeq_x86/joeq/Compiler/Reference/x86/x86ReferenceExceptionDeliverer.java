/*
 * x86ReferenceExceptionDeliverer.java
 *
 * Created on January 12, 2001, 8:44 AM
 *
 */

package Compil3r.Reference.x86;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Main.jq;

/**
 * @author  John Whaley
 * @version $Id$
 */
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
        Unsafe.longJump(ip, fp, sp, 0);
    }
    
    public final Object getThisPointer(jq_CompiledCode cc, int ip, int fp) {
        jq_Method m = cc.getMethod();
        int n_paramwords = m.getParamWords();
        jq.Assert(n_paramwords >= 1);
        return Unsafe.asObject(Unsafe.peek(fp + ((n_paramwords+1)<<2)));
    }

}

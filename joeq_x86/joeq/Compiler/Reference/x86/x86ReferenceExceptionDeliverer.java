/*
 * x86ReferenceExceptionDeliverer.java
 *
 * Created on January 12, 2001, 8:44 AM
 *
 */

package Compil3r.Reference.x86;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Main.jq;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class x86ReferenceExceptionDeliverer extends ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static final x86ReferenceExceptionDeliverer INSTANCE =
    new x86ReferenceExceptionDeliverer();

    private x86ReferenceExceptionDeliverer() {}

    public final void deliverToStackFrame(jq_CompiledCode cc, Throwable x, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        // find new top of stack
        int n_paramwords = m.getParamWords();
        int n_localwords = m.getMaxLocals();
        StackAddress sp = (StackAddress)fp.offset(((n_paramwords-n_localwords)<<2) - 4);
        if (TRACE) SystemInterface.debugmsg("poking exception object "+HeapAddress.addressOf(x).stringRep()+" into location "+sp.stringRep());
        // push exception object there
        sp.poke(HeapAddress.addressOf(x));
        // branch!
        Unsafe.longJump(ip, fp, sp, 0);
    }
    
    public final Object getThisPointer(jq_CompiledCode cc, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        int n_paramwords = m.getParamWords();
        jq.Assert(n_paramwords >= 1);
        return ((HeapAddress)fp.offset((n_paramwords+1)<<2).peek()).asObject();
    }

}

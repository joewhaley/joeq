// x86ReferenceExceptionDeliverer.java, created Mon Feb  5 23:23:21 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Reference.x86;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Util.Assert;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class x86ReferenceExceptionDeliverer extends ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static final x86ReferenceExceptionDeliverer INSTANCE =
    new x86ReferenceExceptionDeliverer();

    private x86ReferenceExceptionDeliverer() {}

    public final void deliverToStackFrame(jq_CompiledCode cc, Throwable x, jq_TryCatch tc, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        // find new top of stack
        int n_paramwords = m.getParamWords();
        int n_localwords = m.getMaxLocals();
        StackAddress sp = (StackAddress)fp.offset(((n_paramwords-n_localwords)<<2) - 4);
        if (TRACE) SystemInterface.debugwriteln("poking exception object "+HeapAddress.addressOf(x).stringRep()+" into location "+sp.stringRep());
        // push exception object there
        sp.poke(HeapAddress.addressOf(x));
        // branch!
        Unsafe.longJump(ip, fp, sp, 0);
    }
    
    public final Object getThisPointer(jq_CompiledCode cc, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        int n_paramwords = m.getParamWords();
        Assert._assert(n_paramwords >= 1);
        return ((HeapAddress)fp.offset((n_paramwords+1)<<2).peek()).asObject();
    }

}

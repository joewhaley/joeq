// x86QuadExceptionDeliverer.java, created Thu Mar  6  0:42:31 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad.x86;

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
public class x86QuadExceptionDeliverer extends ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static final x86QuadExceptionDeliverer INSTANCE =
    new x86QuadExceptionDeliverer();

    private x86QuadExceptionDeliverer() {}

    public final void deliverToStackFrame(jq_CompiledCode cc, Throwable x, jq_TryCatch tc, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        
        Assert._assert(tc.getExceptionOffset() != 0);
        StackAddress sp = (StackAddress) fp.offset(tc.getExceptionOffset());
        if (TRACE) SystemInterface.debugwriteln("poking exception object "+HeapAddress.addressOf(x).stringRep()+" into location "+sp.stringRep());
        // push exception object there
        sp.poke(HeapAddress.addressOf(x));
        
        sp = (StackAddress) fp.offset(-cc.getStackFrameSize());
        
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

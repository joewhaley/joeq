/*
 * x86QuadExceptionDeliverer.java
 *
 * Created on January 12, 2001, 8:44 AM
 *
 */

package Compil3r.Quad.x86;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Util.Assert;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class x86QuadExceptionDeliverer extends ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static final x86QuadExceptionDeliverer INSTANCE =
    new x86QuadExceptionDeliverer();

    private x86QuadExceptionDeliverer() {}

    public final void deliverToStackFrame(jq_CompiledCode cc, Throwable x, jq_TryCatch tc, CodeAddress ip, StackAddress fp) {
        jq_Method m = cc.getMethod();
        ControlFlowGraph cfg = CodeCache.getCode(m);
        
        StackAddress sp = (StackAddress) fp.offset(tc.getExceptionOffset());
        if (TRACE) SystemInterface.debugwriteln("poking exception object "+HeapAddress.addressOf(x).stringRep()+" into location "+sp.stringRep());
        // push exception object there
        sp.poke(HeapAddress.addressOf(x));
        
        sp = (StackAddress) fp.offset(cc.getStackFrameSize());
        
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

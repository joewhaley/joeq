/*
 * jq_TryCatch.java
 *
 * Created on January 2, 2001, 4:23 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Run_Time.TypeCheck;
import Run_Time.SystemInterface;
import jq;

public class jq_TryCatch {

    public static final boolean DEBUG = false;
    
    // NOTE: startPC is exclusive, endPC is inclusive (opposite of jq_TryCatchBC)
    // this is because the IP that we check against is IMMEDIATELY AFTER where the exception actually occurred.
    // these are OFFSETS.
    private int startPC, endPC, handlerPC;
    private jq_Class exType;

    public jq_TryCatch(int startPC, int endPC, int handlerPC, jq_Class exType) {
        this.startPC = startPC;
        this.endPC = endPC;
        this.handlerPC = handlerPC;
        this.exType = exType;
    }

    // note: offset is the offset of the instruction after the one which threw the exception.
    public boolean catches(int offset, jq_Class t) {
        if (DEBUG) SystemInterface.debugmsg(this+": checking "+jq.hex(offset)+" "+t);
        if (offset <= startPC) return false;
        if (offset > endPC) return false;
        if (exType != null) {
            exType.load(); exType.verify(); exType.prepare();
            if (!TypeCheck.isAssignable(t, exType)) return false;
        }
        return true;
    }
    
    public int getStart() { return startPC; }
    public int getEnd() { return endPC; }
    public int getHandlerEntry() { return handlerPC; }
    public jq_Class getExceptionType() { return exType; }

    public String toString() {
        return "(start="+jq.hex(startPC)+",end="+jq.hex(endPC)+",handler="+jq.hex(handlerPC)+",type="+exType+")";
    }
    
    public static final jq_InstanceField _startPC;
    public static final jq_InstanceField _endPC;
    public static final jq_InstanceField _handlerPC;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_TryCatch;");
        _startPC = k.getOrCreateInstanceField("startPC", "I");
        _endPC = k.getOrCreateInstanceField("endPC", "I");
        _handlerPC = k.getOrCreateInstanceField("handlerPC", "I");
    }
}

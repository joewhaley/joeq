/*
 * jq_TryCatch.java
 *
 * Created on January 2, 2001, 4:23 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import Clazz.jq_Class;
import Run_Time.TypeCheck;
import Run_Time.SystemInterface;
import jq;

public class jq_TryCatch {

    public static final boolean DEBUG = false;
    
    // NOTE: startPC is exclusive, endPC is inclusive (opposite of jq_TryCatchBC)
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
    
    public int/*Address*/ getStart() { return startPC; }
    public int/*Address*/ getEnd() { return endPC; }
    public int/*Address*/ getHandlerEntry() { return handlerPC; }
    public jq_Class getExceptionType() { return exType; }

    public String toString() {
        return "(start="+jq.hex(startPC)+",end="+jq.hex(endPC)+",handler="+jq.hex(handlerPC)+",type="+exType+")";
    }
}

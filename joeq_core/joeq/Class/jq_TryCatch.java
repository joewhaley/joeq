/*
 * jq_TryCatch.java
 *
 * Created on January 2, 2001, 4:23 PM
 *
 */

package Clazz;

import Run_Time.Debug;
import Run_Time.TypeCheck;
import Util.Strings;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class jq_TryCatch {

    public static final boolean DEBUG = false;
    
    // NOTE: startPC is exclusive, endPC is inclusive (opposite of jq_TryCatchBC)
    // this is because the IP that we check against is IMMEDIATELY AFTER where the exception actually occurred.
    // these are CODE OFFSETS.
    private int startPC, endPC, handlerPC;
    private jq_Class exType;
    // this is the offset from the frame pointer where to put the exception.
    private int exceptionOffset;

    public jq_TryCatch(int startPC, int endPC, int handlerPC, jq_Class exType, int exceptionOffset) {
        this.startPC = startPC;
        this.endPC = endPC;
        this.handlerPC = handlerPC;
        this.exType = exType;
        this.exceptionOffset = exceptionOffset;
    }

    // note: offset is the offset of the instruction after the one which threw the exception.
    public boolean catches(int offset, jq_Class t) {
        if (DEBUG) Debug.writeln(this+": checking "+Strings.hex(offset)+" "+t);
        if (offset <= startPC) return false;
        if (offset > endPC) return false;
        if (exType != null) {
            exType.prepare();
            if (!TypeCheck.isAssignable(t, exType)) return false;
        }
        return true;
    }
    
    public int getStart() { return startPC; }
    public int getEnd() { return endPC; }
    public int getHandlerEntry() { return handlerPC; }
    public jq_Class getExceptionType() { return exType; }
    public int getExceptionOffset() { return exceptionOffset; }

    public String toString() {
        return "(start="+Strings.hex(startPC)+",end="+Strings.hex(endPC)+",handler="+Strings.hex(handlerPC)+",type="+exType+",offset="+Strings.shex(exceptionOffset)+")";
    }
    
}

/*
 * jq_TryCatch.java
 *
 * Created on January 2, 2001, 4:23 PM
 *
 */

package Clazz;

import Bootstrap.PrimordialClassLoader;
import Run_Time.DebugInterface;
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

    public jq_TryCatch(int startPC, int endPC, int handlerPC, jq_Class exType) {
        this.startPC = startPC;
        this.endPC = endPC;
        this.handlerPC = handlerPC;
        this.exType = exType;
    }

    // note: offset is the offset of the instruction after the one which threw the exception.
    public boolean catches(int offset, jq_Class t) {
        if (DEBUG) DebugInterface.debugwriteln(this+": checking "+Strings.hex(offset)+" "+t);
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

    public String toString() {
        return "(start="+Strings.hex(startPC)+",end="+Strings.hex(endPC)+",handler="+Strings.hex(handlerPC)+",type="+exType+")";
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

/*
 * jq_TryCatchBC.java
 *
 * Created on January 2, 2001, 4:23 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import Clazz.jq_Class;
import Run_Time.TypeCheck;
import jq;

public class jq_TryCatchBC {

    // NOTE: startPC is inclusive, endPC is exclusive (opposite of jq_TryCatch)
    private char startPC, endPC, handlerPC;
    private jq_Class exType;

    public jq_TryCatchBC(char startPC, char endPC, char handlerPC, jq_Class exType) {
        this.startPC = startPC;
        this.endPC = endPC;
        this.handlerPC = handlerPC;
        this.exType = exType;
    }

    public boolean catches(int pc, jq_Class t) {
        t.load(); t.verify(); t.prepare();
        if (pc < startPC) return false;
        if (pc >= endPC) return false;
        if (exType != null && !TypeCheck.isAssignable(t, exType)) return false;
        return true;
    }
    
    public char getStartPC() { return startPC; }
    public char getEndPC() { return endPC; }
    public char getHandlerPC() { return handlerPC; }
    public jq_Class getExceptionType() { return exType; }
    
    public String toString() {
        return "(start="+(int)startPC+",end="+(int)endPC+",handler="+(int)handlerPC+",type="+exType+")";
    }
}

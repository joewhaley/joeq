/*
 * ExceptionHandler.java
 *
 * Created on May 18, 2001, 10:20 AM
 *
 */

package Compil3r.BytecodeAnalysis;

import Clazz.jq_Class;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class ExceptionHandler {

    final jq_Class exceptionType;
    final BasicBlock[] handledBlocks;
    final BasicBlock entry;
    
    ExceptionHandler(jq_Class exceptionType,
                        int numOfHandledBlocks,
                        BasicBlock entry) {
        this.exceptionType = exceptionType;
        this.handledBlocks = new BasicBlock[numOfHandledBlocks];
        this.entry = entry;
    }
    
    public jq_Class getExceptionType() { return exceptionType; }
    public BasicBlock getEntry() { return entry; }
    public BasicBlock[] getHandledBlocks() { return handledBlocks; }
}

/*
 * ExceptionHandlerBasicBlock.java
 *
 * Created on January 9, 2002, 5:25 PM
 *
 * @author  Administrator
 * @version 
 */

package Compil3r.Quad;

import Clazz.jq_Class;
import java.util.ArrayList;
import java.util.List;

public class ExceptionHandler {

    private jq_Class exception_type;
    private List/*<BasicBlock>*/ handled_blocks;
    private BasicBlock entry;
    
    /** Creates new ExceptionHandler */
    public ExceptionHandler(jq_Class ex_type, int numOfHandledBlocks, BasicBlock entry) {
        this.exception_type = ex_type;
        this.handled_blocks = new ArrayList(numOfHandledBlocks);
        this.entry = entry;
    }

    public jq_Class getExceptionType() { return exception_type; }
    public BasicBlockIterator getHandledBasicBlocks() { return new BasicBlockIterator(handled_blocks); }
    public BasicBlock getEntry() { return entry; }

    void addHandledBasicBlock(BasicBlock bb) { handled_blocks.add(bb); }
}

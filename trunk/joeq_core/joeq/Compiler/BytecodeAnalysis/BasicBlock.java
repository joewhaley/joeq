/*
 * BasicBlock.java
 *
 * Created on April 22, 2001, 1:11 PM
 *
 */

package Compil3r.BytecodeAnalysis;

import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class BasicBlock {

    public final int id;
    final int start;
    int end;
    BasicBlock[] predecessors;
    BasicBlock[] successors;
    ExceptionHandlerSet exception_handler_set;

    boolean isSubroutineRet;
    
    int startingStackDepth;
    
    BasicBlock(int id, int start) {
        this.id = id; this.start = start;
    }
    
    public int getStart() { return start; }
    public int getEnd() { return end; }
    
    public int getNumberOfPredecessors() { return predecessors.length; }
    public int getNumberOfSuccessors() { return successors.length; }
    public BasicBlock getPredecessor(int i) { return predecessors[i]; }
    public BasicBlock getSuccessor(int i) { return successors[i]; }
    public boolean isSubroutineRet() { return isSubroutineRet; }
    void setSubroutineRet(ControlFlowGraph cfg, BasicBlock jsub_bb) {
        isSubroutineRet = true;
        Assert._assert(this.successors.length == 0);
        this.successors = new BasicBlock[jsub_bb.predecessors.length];
        for (int i=0; i<this.successors.length; ++i) {
            int ret_target_index = jsub_bb.predecessors[i].id + 1;
            Assert._assert(ret_target_index < cfg.getNumberOfBasicBlocks());
            BasicBlock ret_target = cfg.getBasicBlock(ret_target_index);
            this.successors[i] = ret_target;
            BasicBlock[] new_pred = new BasicBlock[ret_target.predecessors.length+1];
            if (ret_target.predecessors.length != 0) {
                System.arraycopy(ret_target.predecessors, 0, new_pred, 0, ret_target.predecessors.length);
            }
            new_pred[ret_target.predecessors.length] = this;
            ret_target.predecessors = new_pred;
        }
    }
    
    public ExceptionHandlerIterator getExceptionHandlers() {
        if (exception_handler_set == null) return ExceptionHandlerIterator.nullIterator();
        return exception_handler_set.iterator();
    }
    
    void addExceptionHandler_first(ExceptionHandlerSet eh) {
        Assert._assert(eh.parent == null);
        eh.parent = this.exception_handler_set;
        this.exception_handler_set = eh;
    }
    ExceptionHandlerSet addExceptionHandler(ExceptionHandlerSet eh) {
        if (eh.parent == this.exception_handler_set)
            return this.exception_handler_set = eh;
        else
            return this.exception_handler_set = new ExceptionHandlerSet(eh.getHandler(), this.exception_handler_set);
    }

    public String toString() { return "BB"+id+" ("+start+"-"+end+")"; }
}

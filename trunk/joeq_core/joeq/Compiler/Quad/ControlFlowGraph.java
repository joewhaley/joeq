/*
 * ControlFlowGraph.java
 *
 * Created on April 21, 2001, 11:25 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Iterator;
import Util.FilterIterator;

public class ControlFlowGraph {

    private final BasicBlock start_node;
    private final BasicBlock end_node;
    private final List/*<ExceptionHandler>*/ exception_handlers;
    
    private int bb_counter;
    
    /** Creates new ControlFlowGraph */
    public ControlFlowGraph(int numOfExits, int numOfExceptionHandlers) {
        start_node = BasicBlock.createStartNode();
        end_node = BasicBlock.createEndNode(numOfExits);
        exception_handlers = new ArrayList(numOfExceptionHandlers);
        bb_counter = 1;
    }

    public BasicBlock entry() { return start_node; }
    public BasicBlock exit() { return end_node; }

    public BasicBlock createBasicBlock(int numOfPredecessors, int numOfSuccessors, int numOfInstructions,
                                       ExceptionHandlerSet ehs) {
        return BasicBlock.createBasicBlock(++bb_counter, numOfPredecessors, numOfSuccessors, numOfInstructions, ehs);
    }
    void updateBBcounter(int value) { bb_counter = value-1; }
    public int getNumberOfBasicBlocks() { return bb_counter+1; }

    public BasicBlockIterator reversePostOrderIterator() {
	return reversePostOrderIterator(start_node);
    }
    
    public BasicBlockIterator reverseReversePostOrderIterator() {
        return new BasicBlockIterator(reverseReversePostOrder(end_node));
    }
    
    public BasicBlockIterator reversePostOrderIterator(BasicBlock start_bb) {
        return new BasicBlockIterator(reversePostOrder(start_bb));
    }

    public void visitBasicBlocks(BasicBlockVisitor bbv) {
        for (BasicBlockIterator i=reversePostOrderIterator(); i.hasNext(); ) {
            BasicBlock bb = i.nextBB();
            bbv.visitBasicBlock(bb);
        }
    }
    
    public List reversePostOrder(BasicBlock start_bb) {
	LinkedList result = new LinkedList();
	boolean[] visited = new boolean[bb_counter+1];
	reversePostOrder_helper(start_bb, visited, result, true);
	return result;
    }

    public List reverseReversePostOrder(BasicBlock start_bb) {
	LinkedList result = new LinkedList();
	boolean[] visited = new boolean[bb_counter+1];
	reversePostOrder_helper(start_bb, visited, result, false);
	return result;
    }
    
    private void reversePostOrder_helper(BasicBlock b, boolean[] visited, LinkedList result, boolean direction) {
	if (visited[b.getID()]) return;
	visited[b.getID()] = true;
	BasicBlockIterator bbi = direction ? b.getSuccessors() : b.getPredecessors();
	while (bbi.hasNext()) {
	    BasicBlock b2 = bbi.nextBB();
	    reversePostOrder_helper(b2, visited, result, direction);
	}
        if (direction) {
            ExceptionHandlerIterator ehi = b.getExceptionHandlers();
            while (ehi.hasNext()) {
                ExceptionHandler eh = ehi.nextEH();
                BasicBlock b2 = eh.getEntry();
                reversePostOrder_helper(b2, visited, result, direction);
            }
        } else {
            if (b.isExceptionHandlerEntry()) {
                Iterator i = getExceptionHandlersMatchingEntry(b);
                while (i.hasNext()) {
                    BasicBlock b2 = (BasicBlock)i.next();
                    reversePostOrder_helper(b2, visited, result, direction);
                }
            }
        }
	result.addFirst(b);
    }

    public Iterator getExceptionHandlersMatchingEntry(BasicBlock b) {
        final BasicBlock bb = b;
        return new FilterIterator(exception_handlers.iterator(),
            new FilterIterator.Filter() {
                public boolean isElement(Object o) {
                    ExceptionHandler eh = (ExceptionHandler)o;
                    return eh.getEntry() == bb;
                }
        });
    }
    
    public String fullDump() {
	StringBuffer sb = new StringBuffer();
	BasicBlockIterator i = reversePostOrderIterator();
	while (i.hasNext()) {
	    BasicBlock bb = i.nextBB();
	    sb.append(bb.fullDump());
	}
	return sb.toString();
    }

}

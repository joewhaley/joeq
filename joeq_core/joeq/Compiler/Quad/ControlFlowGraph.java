/*
 * ControlFlowGraph.java
 *
 * Created on April 21, 2001, 11:25 PM
 *
 */

package Compil3r.Quad;
import Util.FilterIterator;
import Util.Templates.List;
import Util.Templates.ListIterator;
import Util.Templates.UnmodifiableList;

/**
 * Control flow graph for the Quad format.
 * The control flow graph is a fundamental part of the quad intermediate representation.
 * The control flow graph organizes the basic blocks for a method.
 * 
 * Control flow graphs always include an entry basic block and an exit basic block.
 * These basic blocks are always empty and have id numbers 0 and 1, respectively.
 * 
 * A control flow graph includes references to the entry and exit nodes, and the
 * set of exception handlers for the method.
 * 
 * @author  John Whaley
 * @version $Id$
 */

public class ControlFlowGraph {

    /* Reference to the start node of this control flow graph. */
    private final BasicBlock start_node;
    /* Reference to the end node of this control flow graph. */
    private final BasicBlock end_node;
    /* List of exception handlers for this control flow graph. */
    private final java.util.List/*<ExceptionHandler>*/ exception_handlers;
    
    /* Current number of basic blocks, used to generate unique id's. */
    private int bb_counter;
    /* Current number of quads, used to generate unique id's. */
    private int quad_counter;
    
    /** Creates a new ControlFlowGraph.
     * @param numOfExits  the expected number of branches to the exit node.
     * @param numOfExceptionHandlers  the expected number of exception handlers. */
    public ControlFlowGraph(int numOfExits, int numOfExceptionHandlers) {
        start_node = BasicBlock.createStartNode();
        end_node = BasicBlock.createEndNode(numOfExits);
        exception_handlers = new java.util.ArrayList(numOfExceptionHandlers);
        bb_counter = 1; quad_counter = 0;
    }

    /** Returns the entry node.
     * @return  the entry node. */
    public BasicBlock entry() { return start_node; }
    /** Returns the exit node.
     * @return  the exit node. */
    public BasicBlock exit() { return end_node; }

    /** Create a new basic block in this control flow graph.  The new basic block
     * is given a new, unique id number.
     * @param numOfPredecessors  number of predecessor basic blocks that this
                                 basic block is expected to have.
     * @param numOfSuccessors  number of successor basic blocks that this
                               basic block is expected to have.
     * @param numOfInstructions  number of instructions that this basic block
                                 is expected to have.
     * @param ehs  set of exception handlers for this basic block.
     * @return  the newly created basic block. */
    public BasicBlock createBasicBlock(int numOfPredecessors, int numOfSuccessors, int numOfInstructions,
                                       ExceptionHandlerList ehs) {
        return BasicBlock.createBasicBlock(++bb_counter, numOfPredecessors, numOfSuccessors, numOfInstructions, ehs);
    }
    /** Use with care after renumbering basic blocks. */
    void updateBBcounter(int value) { bb_counter = value-1; }
    /** Returns a maximum on the number of basic blocks in this control flow graph.
     * @return  a maximum on the number of basic blocks in this control flow graph. */
    public int getNumberOfBasicBlocks() { return bb_counter+1; }

    /** Returns a new id number for a quad. */
    int getNewQuadID() { return ++quad_counter; }
    
    /** Returns an iteration of the basic blocks in this graph in reverse post order.
     * @see  BasicBlockIterator
     * @return  an iteration of the basic blocks in this graph in reverse post order. */
    public ListIterator.BasicBlock reversePostOrderIterator() {
	return reversePostOrderIterator(start_node);
    }
    
    /** Returns an iteration of the basic blocks in the reversed graph in reverse post order.
     * The reversed graph is the graph where all edges are reversed.
     * @see  BasicBlockIterator
     * @return  an iteration of the basic blocks in the reversed graph in reverse post order. */
    public ListIterator.BasicBlock reverseReversePostOrderIterator() {
        return reverseReversePostOrder(end_node).basicBlockIterator();
    }
    
    /** Returns an iteration of the basic blocks in this graph reachable from the given
     * basic block in reverse post order, starting from the given basic block.
     * @param start_bb  basic block to start reverse post order from.
     * @return  an iteration of the basic blocks in this graph reachable from the given basic block in reverse post order. */
    public ListIterator.BasicBlock reversePostOrderIterator(BasicBlock start_bb) {
        return reversePostOrder(start_bb).basicBlockIterator();
    }

    /** Visits all of the basic blocks in this graph with the given visitor.
     * @param bbv  visitor to visit each basic block with. */
    public void visitBasicBlocks(BasicBlockVisitor bbv) {
        for (ListIterator.BasicBlock i=reversePostOrderIterator(); i.hasNext(); ) {
            BasicBlock bb = i.nextBasicBlock();
            bbv.visitBasicBlock(bb);
        }
    }
    
    /** Returns a list of basic blocks in reverse post order, starting at the given basic block.
     * @param start_bb  basic block to start from.
     * @return  a list of basic blocks in reverse post order, starting at the given basic block. */
    public List.BasicBlock reversePostOrder(BasicBlock start_bb) {
	java.util.LinkedList/*<BasicBlock>*/ result = new java.util.LinkedList();
	boolean[] visited = new boolean[bb_counter+1];
	reversePostOrder_helper(start_bb, visited, result, true);
        BasicBlock[] bb = new BasicBlock[result.size()];
        bb = (BasicBlock[])result.toArray(bb);
	return new UnmodifiableList.BasicBlock(bb);
    }

    /** Returns a list of basic blocks of the reversed graph in reverse post order, starting at the given basic block.
     * @param start_bb  basic block to start from.
     * @return  a list of basic blocks of the reversed graph in reverse post order, starting at the given basic block. */
    public List.BasicBlock reverseReversePostOrder(BasicBlock start_bb) {
	java.util.LinkedList/*<BasicBlock>*/ result = new java.util.LinkedList();
	boolean[] visited = new boolean[bb_counter+1];
	reversePostOrder_helper(start_bb, visited, result, false);
        BasicBlock[] bb = new BasicBlock[result.size()];
        bb = (BasicBlock[])result.toArray(bb);
	return new UnmodifiableList.BasicBlock(bb);
    }
    
    /** Helper function to compute reverse post order. */
    private void reversePostOrder_helper(BasicBlock b, boolean[] visited, java.util.LinkedList result, boolean direction) {
	if (visited[b.getID()]) return;
	visited[b.getID()] = true;
	List.BasicBlock bbs = direction ? b.getSuccessors() : b.getPredecessors();
        ListIterator.BasicBlock bbi = bbs.basicBlockIterator();
	while (bbi.hasNext()) {
	    BasicBlock b2 = bbi.nextBasicBlock();
	    reversePostOrder_helper(b2, visited, result, direction);
	}
        if (direction) {
            ListIterator.ExceptionHandler ehi = b.getExceptionHandlers().exceptionHandlerIterator();
            while (ehi.hasNext()) {
                ExceptionHandler eh = ehi.nextExceptionHandler();
                BasicBlock b2 = eh.getEntry();
                reversePostOrder_helper(b2, visited, result, direction);
            }
        } else {
            if (b.isExceptionHandlerEntry()) {
                java.util.Iterator i = getExceptionHandlersMatchingEntry(b);
                while (i.hasNext()) {
                    BasicBlock b2 = (BasicBlock)i.next();
                    reversePostOrder_helper(b2, visited, result, direction);
                }
            }
        }
	result.addFirst(b);
    }

    /** Return an iterator of the exception handlers with the given entry point.
     * @param b  basic block to check exception handlers against.
     * @return  an iterator of the exception handlers with the given entry point. */
    public java.util.Iterator getExceptionHandlersMatchingEntry(BasicBlock b) {
        final BasicBlock bb = b;
        return new FilterIterator(exception_handlers.iterator(),
            new FilterIterator.Filter() {
                public boolean isElement(Object o) {
                    ExceptionHandler eh = (ExceptionHandler)o;
                    return eh.getEntry() == bb;
                }
        });
    }
    
    /** Returns a verbose string of every basic block in this control flow graph.
     * @return  a verbose string of every basic block in this control flow graph. */
    public String fullDump() {
	StringBuffer sb = new StringBuffer();
	ListIterator.BasicBlock i = reversePostOrderIterator();
	while (i.hasNext()) {
	    BasicBlock bb = i.nextBasicBlock();
	    sb.append(bb.fullDump());
	}
	return sb.toString();
    }

}

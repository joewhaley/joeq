/*
 * BasicBlock.java
 *
 * Created on April 21, 2001, 11:04 PM
 *
 */

package Compil3r.Quad;
import jq;
import java.util.*;

/**
 * Represents a basic block in the quad intermediate representation.
 * Basic blocks are single-entry regions, but not necessarily single-exit regions
 * due to the fact that control flow may exit a basic block early due to a
 * run time exception.  That is to say, a potential exception point does not
 * end a basic block.
 *
 * Each basic block contains a list of quads, a list of predecessors, a list of
 * successors, and a set of exception handlers.  It also has an id number that
 * is unique within its control flow graph, and some flags.
 *
 * You should never create a basic block directly.  You should create one via a
 * ControlFlowGraph so that the id number is correct.
 *
 * @author  John Whaley
 * @see  Quad
 * @see  ControlFlowGraph
 * @see  ExceptionHandlerSet
 * @version  $Id$
 */

public class BasicBlock {

    /** Unique id number for this basic block. */
    private int id_number;
    /** List of instructions. */
    private final List/*<Quad>*/ instructions;
    /** List of successor basic blocks. */
    private final List/*<BasicBlock>*/ successors;
    /** List of predecessor basic blocks. */
    private final List/*<BasicBlock>*/ predecessors;
    
    /** Set of exception handlers for this basic block. */
    private ExceptionHandlerSet exception_handler_set;
    
    /** Flags for this basic block. */
    private int flags;
    
    /** Exception handler entry point. */
    private static final int EXCEPTION_HANDLER_ENTRY = 0x1;
    /** JSR subroutine entry point. */
    private static final int JSR_ENTRY = 0x2;
    /** This basic block ends in a 'ret'. */
    private static final int ENDS_IN_RET = 0x4;
    
    /** Creates new entry node. Only to be called by ControlFlowGraph. */
    static BasicBlock createStartNode() {
        return new BasicBlock();
    }
    /** Private constructor for the entry node. */
    private BasicBlock() {
        this.id_number = 0;
        this.instructions = null;
        this.predecessors = null;
        this.successors = new ArrayList(1);
        this.exception_handler_set = null;
    }
    /** Creates new exit node */
    static BasicBlock createEndNode(int numOfPredecessors) {
        return new BasicBlock(numOfPredecessors);
    }
    /** Private constructor for the exit node. */
    private BasicBlock(int numOfPredecessors) {
        this.id_number = 1;
        this.instructions = null;
        this.successors = null;
        this.predecessors = new ArrayList(numOfPredecessors);
        this.exception_handler_set = null;
    }
    /** Create new basic block with no exception handlers.
     * Only to be called by ControlFlowGraph. */
    static BasicBlock createBasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions) {
        return new BasicBlock(id, numOfPredecessors, numOfSuccessors, numOfInstructions, null);
    }
    /** Create new basic block with the given exception handlers.
     * Only to be called by ControlFlowGraph. */
    static BasicBlock createBasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions, ExceptionHandlerSet ehs) {
        return new BasicBlock(id, numOfPredecessors, numOfSuccessors, numOfInstructions, ehs);
    }
    /** Private constructor for internal nodes. */
    private BasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions,
                       ExceptionHandlerSet ehs) {
        this.id_number = id;
        this.predecessors = new ArrayList(numOfPredecessors);
        this.successors = new ArrayList(numOfSuccessors);
        this.instructions = new ArrayList(numOfInstructions);
        this.exception_handler_set = ehs;
    }

    /** Returns true if this is the entry basic block.
     * @return  true if this is the entry basic block. */
    public boolean isEntry() { return predecessors == null; }
    /** Returns true if this is the exit basic block.
     * @return  true if this is the exit basic block. */
    public boolean isExit() { return successors == null; }
    
    /** Returns an iterator over the quads in this basic block in forward order.
     * @see  QuadIterator
     * @return  an iterator over the quads in this basic block in forward order. */
    public QuadIterator iterator() {
	if (instructions == null) return QuadIterator.getEmptyIterator();
        return new QuadIterator(instructions);
    }
    
    /** Returns an iterator over the quads in this basic block in backward order.
     * @see  QuadIterator
     * @return  an iterator over the quads in this basic block in backward order. */
    public QuadIterator backwardIterator() {
	if (instructions == null) return QuadIterator.getEmptyIterator();
        return new BackwardQuadIterator(instructions);
    }

    /** Visit all of the quads in this basic block in forward order
     * with the given quad visitor.
     * @see  QuadVisitor
     * @param qv  QuadVisitor to visit the quads with. */
    public void visitQuads(QuadVisitor qv) {
        for (QuadIterator i = iterator(); i.hasNext(); ) {
            Quad q = i.nextQuad();
            q.accept(qv);
        }
    }
    
    /** Visit all of the quads in this basic block in backward order
     * with the given quad visitor.
     * @see  QuadVisitor
     * @param qv  QuadVisitor to visit the quads with. */
    public void backwardVisitQuads(QuadVisitor qv) {
        for (QuadIterator i = backwardIterator(); i.hasNext(); ) {
            Quad q = i.nextQuad();
            q.accept(qv);
        }
    }
    
    /** Returns the number of quads in this basic block.
     * @return  the number of quads in this basic block. */
    public int size() {
        if (instructions == null) return 0; // entry or exit block
        return instructions.size();
    }

    /** Add a quad to this basic block at the given location.
     * Cannot add quads to the entry or exit basic blocks.
     * @param index  the index to add the quad
     * @param q  quad to add */
    public void addQuad(int index, Quad q) {
	jq.assert(instructions != null, "Cannot add instructions to entry/exit basic block");
        instructions.add(index, q);
    }
    /** Append a quad to the end of this basic block.
     * Cannot add quads to the entry or exit basic blocks.
     * @param q  quad to add */
    public void appendQuad(Quad q) {
	jq.assert(instructions != null, "Cannot add instructions to entry/exit basic block");
        instructions.add(q);
    }
    
    /** Add a predecessor basic block to this basic block.
     * Cannot add predecessors to the entry basic block.
     * @param b  basic block to add as a predecessor */
    public void addPredecessor(BasicBlock b) {
	jq.assert(predecessors != null, "Cannot add predecessor to entry basic block");
	predecessors.add(b);
    }
    /** Add a successor basic block to this basic block.
     * Cannot add successors to the exit basic block.
     * @param b  basic block to add as a successor */
    public void addSuccessor(BasicBlock b) {
	jq.assert(successors != null, "Cannot add successor to exit basic block");
	successors.add(b);
    }
    
    /** Returns the fallthrough successor to this basic block, if it exists.
     * If there is none, returns null.
     * @return  the fallthrough successor, or null if there is none. */
    public BasicBlock getFallthroughSuccessor() {
	if (successors == null) return null;
        return (BasicBlock)successors.get(0);
    }

    /** Returns the fallthrough predecessor to this basic block, if it exists.
     * If there is none, returns null.
     * @return  the fallthrough predecessor, or null if there is none. */
    public BasicBlock getFallthroughPredecessor() {
	if (predecessors == null) return null;
        return (BasicBlock)predecessors.get(0);
    }

    /** Returns an iterator of the successors of this basic block.
     * @see BasicBlockIterator
     * @return  an iterator of the successors of this basic block. */
    public BasicBlockIterator getSuccessors() {
	if (successors == null) return BasicBlockIterator.getEmptyIterator();
        return new BasicBlockIterator(successors);
    }
    
    /** Returns an iterator of the predecessors of this basic block.
     * @see BasicBlockIterator
     * @return  an iterator of the predecessors of this basic block. */
    public BasicBlockIterator getPredecessors() {
	if (predecessors == null) return BasicBlockIterator.getEmptyIterator();
        return new BasicBlockIterator(predecessors);
    }
    
    void addExceptionHandler_first(ExceptionHandlerSet eh) {
        eh.getHandler().addHandledBasicBlock(this);
        jq.assert(eh.parent == null);
        eh.parent = this.exception_handler_set;
        this.exception_handler_set = eh;
    }
    ExceptionHandlerSet addExceptionHandler(ExceptionHandlerSet eh) {
        eh.getHandler().addHandledBasicBlock(this);
        if (eh.parent == this.exception_handler_set)
            return this.exception_handler_set = eh;
        else
            return this.exception_handler_set = new ExceptionHandlerSet(eh.getHandler(), this.exception_handler_set);
    }
    
    /** Returns an iterator of the exception handlers that guard this basic block.
     * @see ExceptionHandlerIterator
     * @return  an iterator of the exception handlers that guard this basic block. */
    public ExceptionHandlerIterator getExceptionHandlers() {
        if (exception_handler_set == null) return ExceptionHandlerIterator.getEmptyIterator();
        return exception_handler_set.iterator();
    }
    
    /** Returns the unique id number for this basic block.
     * @return  the unique id number for this basic block. */
    public int getID() { return id_number; }

    /** Returns true if this basic block has been marked as an exception handler
     * entry point.  Returns false otherwise.
     * @return  if this basic block has been marked as an exception handler entry point. */
    public boolean isExceptionHandlerEntry() { return (flags & EXCEPTION_HANDLER_ENTRY) != 0; }
    /** Marks this basic block as an exception handler entry point.
     */
    public void setExceptionHandlerEntry() { flags |= EXCEPTION_HANDLER_ENTRY; }
    
    /** Returns the name of this basic block.
     * @return  the name of this basic block. */
    public String toString() {
        if (isEntry()) {
            jq.assert(getID() == 0);
            return "BB0 (ENTRY)";
        }
        if (isExit()) {
            jq.assert(getID() == 1);
            return "BB1 (EXIT)";
        }
        return "BB"+getID();
    }
    
    /** Returns a String describing the name, predecessor, successor, exception
     * handlers, and quads of this basic block.
     * @return  a verbose string describing this basic block */    
    public String fullDump() {
        StringBuffer sb = new StringBuffer();
        sb.append(toString());
        sb.append("\t(in: ");
        BasicBlockIterator bbi = getPredecessors();
        if (!bbi.hasNext()) sb.append("<none>");
        else {
            sb.append(bbi.nextBB().toString());
            while (bbi.hasNext()) {
                sb.append(", ");
                sb.append(bbi.nextBB().toString());
            }
        }
        sb.append(", out: ");
        bbi = getSuccessors();
        if (!bbi.hasNext()) sb.append("<none>");
        else {
            sb.append(bbi.nextBB().toString());
            while (bbi.hasNext()) {
                sb.append(", ");
                sb.append(bbi.nextBB().toString());
            }
        }
        sb.append(')');
        ExceptionHandlerIterator ehi = getExceptionHandlers();
        if (ehi.hasNext()) {
            sb.append("\n\texception handlers: ");
            sb.append(ehi.nextEH().toString());
            while (ehi.hasNext()) {
                sb.append(", ");
                sb.append(ehi.nextEH().toString());
            }
        }
        sb.append("\n");
        QuadIterator qi = iterator();
        while (qi.hasNext()) {
            sb.append(qi.nextQuad().toString());
            sb.append('\n');
        }
        sb.append('\n');
        return sb.toString();
    }
    
}

/*
 * BasicBlock.java
 *
 * Created on April 21, 2001, 11:04 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import jq;

import java.util.*;

public class BasicBlock {

    private int id_number;
    private final List/*<Quad>*/ instructions;
    private final List/*<BasicBlock>*/ successors;
    private final List/*<BasicBlock>*/ predecessors;
    
    private ExceptionHandlerSet exception_handler_set;
    
    private int flags;

    public static final int EXCEPTION_HANDLER_ENTRY = 0x1;
    public static final int JSR_ENTRY = 0x2;
    public static final int ENDS_IN_RET = 0x4;
    
    /** Creates new entry node */
    static BasicBlock createStartNode() {
        return new BasicBlock();
    }
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
    private BasicBlock(int numOfPredecessors) {
        this.id_number = 1;
        this.instructions = null;
        this.successors = null;
        this.predecessors = new ArrayList(numOfPredecessors);
        this.exception_handler_set = null;
    }
    /** Create new basic block */
    static BasicBlock createBasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions) {
        return new BasicBlock(id, numOfPredecessors, numOfSuccessors, numOfInstructions, null);
    }
    static BasicBlock createBasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions, ExceptionHandlerSet ehs) {
        return new BasicBlock(id, numOfPredecessors, numOfSuccessors, numOfInstructions, ehs);
    }
    protected BasicBlock(int id, int numOfPredecessors, int numOfSuccessors, int numOfInstructions,
                         ExceptionHandlerSet ehs) {
        this.id_number = id;
        this.predecessors = new ArrayList(numOfPredecessors);
        this.successors = new ArrayList(numOfSuccessors);
        this.instructions = new ArrayList(numOfInstructions);
        this.exception_handler_set = ehs;
    }

    public boolean isEntry() { return predecessors == null; }
    public boolean isExit() { return successors == null; }
    
    public QuadIterator iterator() {
	if (instructions == null) return QuadIterator.getEmptyIterator();
        return new QuadIterator(instructions);
    }
    
    public QuadIterator backwardIterator() {
	if (instructions == null) return QuadIterator.getEmptyIterator();
        return new BackwardQuadIterator(instructions);
    }

    public void visitQuads(QuadVisitor qv) {
        for (QuadIterator i = iterator(); i.hasNext(); ) {
            Quad q = i.nextQuad();
            q.accept(qv);
        }
    }
    
    public void backwardVisitQuads(QuadVisitor qv) {
        for (QuadIterator i = backwardIterator(); i.hasNext(); ) {
            Quad q = i.nextQuad();
            q.accept(qv);
        }
    }
    
    public int size() {
        if (instructions == null) return 0; // entry or exit block
        return instructions.size();
    }

    public void addQuad(int index, Quad q) {
	jq.assert(instructions != null, "Cannot add instructions to start/end node");
        instructions.add(index, q);
    }
    public void appendQuad(Quad q) {
	jq.assert(instructions != null, "Cannot append instructions to start/end node");
        instructions.add(q);
    }
    
    public void addPredecessor(BasicBlock b) {
	jq.assert(predecessors != null, "Cannot add predecessor to start node");
	predecessors.add(b);
    }
    public void addSuccessor(BasicBlock b) {
	jq.assert(successors != null, "Cannot add successor to end node");
	successors.add(b);
    }
    
    public BasicBlock getFallthroughSuccessor() {
	if (successors == null) return null;
        return (BasicBlock)successors.get(0);
    }

    public BasicBlock getFallthroughPredecessor() {
	if (predecessors == null) return null;
        return (BasicBlock)predecessors.get(0);
    }

    public BasicBlockIterator getSuccessors() {
	if (successors == null) return BasicBlockIterator.getEmptyIterator();
        return new BasicBlockIterator(successors);
    }
    
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
    
    public ExceptionHandlerIterator getExceptionHandlers() {
        if (exception_handler_set == null) return ExceptionHandlerIterator.nullIterator();
        return exception_handler_set.iterator();
    }
    
    public int getID() { return id_number; }

    public boolean isExceptionHandlerEntry() { return (flags & EXCEPTION_HANDLER_ENTRY) != 0; }
    public void setExceptionHandlerEntry() { flags |= EXCEPTION_HANDLER_ENTRY; }
    
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

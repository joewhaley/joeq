/*
 * ExceptionHandler.java
 *
 * Created on January 9, 2002, 5:25 PM
 *
 */

package Compil3r.Quad;
import Clazz.jq_Class;
import Run_Time.TypeCheck;
import Util.Templates.List;
import Util.Templates.ListWrapper;

/**
 * Exception handler for basic blocks.  Each exception handler handles a type of
 * exception.  When an exception is raised at run time, a routine looks up the list
 * of exception handlers that guard the location where the exception was raised.
 * It checks each of the exception handlers in order.  Control flow branches to the
 * first exception handler whose type matches the type of the raised exception.
 * Note that the type check is a Java "assignable" type check, and therefore
 * inheritance and interface checks may be necessary.
 * 
 * @see  ExceptionHandlerSet
 * @see  Run_Time.TypeCheck
 * @author  John Whaley
 * @version  $Id$
 */

public class ExceptionHandler {

    /** Type of exception that this exception handler catches. */
    private jq_Class exception_type;
    /** List of handled basic blocks. */
    private java.util.List/*<BasicBlock>*/ handled_blocks;
    /** Exception handler entry point. */
    private BasicBlock entry;
    
    /** Creates new ExceptionHandler.
     * @param ex_type  type of exception to catch.
     * @param numOfHandledBlocks  estimated number of handled basic blocks.
     * @param entry  exception handler entry point. */
    public ExceptionHandler(jq_Class ex_type, int numOfHandledBlocks, BasicBlock entry) {
        this.exception_type = ex_type;
        this.handled_blocks = new java.util.ArrayList(numOfHandledBlocks);
        this.entry = entry;
    }

    /** Returns the type of exception that this exception handler catches.
     * @return  the type of exception that this exception handler catches. */
    public jq_Class getExceptionType() { return exception_type; }
    /** Returns an iteration of the handled basic blocks.
     * @return  an iteration of the handled basic blocks. */
    public List.BasicBlock getHandledBasicBlocks() { return new ListWrapper.BasicBlock(handled_blocks); }
    /** Returns the entry point for this exception handler.
     * @return  the entry point for this exception handler. */
    public BasicBlock getEntry() { return entry; }

    public boolean mustCatch(jq_Class exType) {
        return TypeCheck.isAssignable(exType, exception_type);
    }
    public boolean mayCatch(jq_Class exType) {
        return TypeCheck.isAssignable(exType, exception_type) ||
              TypeCheck.isAssignable(exception_type, exType);
    }
    public String toString() { return "Type: "+exception_type+" Entry: "+entry; }
    
    /** Add a handled basic block to the list of handled basic blocks.
     * @param bb  basic block to add. */
    void addHandledBasicBlock(BasicBlock bb) { handled_blocks.add(bb); }
}

/*
 * ExceptionHandlerSet.java
 *
 * Created on April 22, 2001, 12:19 AM
 *
 */

package Compil3r.Quad;
import Util.ListFactory;
import java.util.*;

/**
 * Holds a list of exception handlers that protect a basic block.
 * It includes a reference to a parent exception handler set, to handle nesting
 * of exception handlers.
 * These form a tree structure where each node has a pointer to its parent.
 *
 * @author  John Whaley
 * @see  ExceptionHandler
 * @see  ExceptionHandlerIterator
 * @version  $Id$
 */

public class ExceptionHandlerSet {

    /** The exception handler. */
    private final ExceptionHandler exception_handler;
    /** The parent exception handler set. */
    ExceptionHandlerSet parent;
    
    /** Creates new ExceptionHandlerSet containing the given exception handler and no parent set.
     * @param exception_handler  exception handler to include in the set. */
    public ExceptionHandlerSet(ExceptionHandler exception_handler) {
        this.exception_handler = exception_handler;
        this.parent = null;
    }
    /** Creates new ExceptionHandlerSet containing the given exception handler and parent set.
     * @param exception_handler  exception handler to include in the set.
     * @param parent  the parent set of exception handlers. */
    public ExceptionHandlerSet(ExceptionHandler exception_handler, ExceptionHandlerSet parent) {
        this.exception_handler = exception_handler;
        this.parent = parent;
    }

    /** Return the handler in this set.  Doesn't include parent handlers.
     * @return  the handler in this set, without parent handlers. */    
    public ExceptionHandler getHandler() { return exception_handler; }
    /** Return the parent set of exception handlers, or null if this set doesn't have a parent.
     * @return  the parent set of exception handlers, or null if this set doesn't have a parent. */
    public ExceptionHandlerSet getParent() { return parent; }
    
    /** Return an iteration over the handlers in this set (and the handlers in parent sets).
     * Handlers are returned in the correct order (this set, followed by parent sets.)
     * @return  an iteration over the handlers in this set (and the handlers in parent sets) in correct order. */
    public ExceptionHandlerIterator iterator() {
        return new ExceptionHandlerIterator(ListFactory.singleton(exception_handler), parent);
    }
    
}

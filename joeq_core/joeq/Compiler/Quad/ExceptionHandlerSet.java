/*
 * ExceptionHandlerSet.java
 *
 * Created on April 22, 2001, 12:19 AM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import Util.ListFactory;
import java.util.*;

public class ExceptionHandlerSet {

    private final ExceptionHandler exception_handler;
    ExceptionHandlerSet parent;
    
    /** Creates new ExceptionHandlerSet */
    public ExceptionHandlerSet(ExceptionHandler exception_handler) {
        this.exception_handler = exception_handler;
        this.parent = null;
    }
    public ExceptionHandlerSet(ExceptionHandler exception_handler, ExceptionHandlerSet parent) {
        this.exception_handler = exception_handler;
        this.parent = parent;
    }

    public ExceptionHandler getHandler() { return exception_handler; }
    public ExceptionHandlerSet getParent() { return parent; }
    
    public ExceptionHandlerIterator iterator() {
        return new ExceptionHandlerIterator(ListFactory.singleton(exception_handler), parent);
    }
    
}

/*
 * ExceptionHandler.java
 *
 * Created on May 18, 2001, 10:20 AM
 *
 */

package Compil3r.BytecodeAnalysis;

import Clazz.jq_Class;
import Util.ListFactory;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class ExceptionHandlerSet {

    final ExceptionHandler exception_handler;
    ExceptionHandlerSet parent;
    
    ExceptionHandlerSet(ExceptionHandler exception_handler, ExceptionHandlerSet parent) {
        this.exception_handler = exception_handler;
        this.parent = parent;
    }
    
    public ExceptionHandler getHandler() { return exception_handler; }
    public ExceptionHandlerSet getParent() { return parent; }

    public ExceptionHandlerIterator iterator() {
        return new ExceptionHandlerIterator(ListFactory.singleton(exception_handler), parent);
    }
}

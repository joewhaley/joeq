/*
 * ExceptionHandlerIterator.java
 *
 * Created on April 22, 2001, 12:53 AM
 *
 */

package Compil3r.Quad;
import Util.Templates.ListIterator;
import java.util.NoSuchElementException;

/**
 * Iterator for iterating through exception handlers.  Compatible with ListIterator.
 * @see  Util.Templates.ListIterator
 * @see  ExceptionHandler
 * @author  John Whaley
 * @version  $Id$
 */
public class ExceptionHandlerIterator implements ListIterator.ExceptionHandler {

    private final ExceptionHandlerList root;
    private ExceptionHandlerList current;
    
    /** Creates new ExceptionHandlerIterator.
     * @param exs  list of exception handlers to iterate through. */
    public ExceptionHandlerIterator(ExceptionHandlerList ehs) {
        root = current = ehs;
    }
    
    /** Returns true if this iterator has a next element.
     * @return  true if this iterator has a next element. */
    public boolean hasNext() { return current != null; }
    /** Returns the next element of this iterator.  Use nextExceptionHandler to avoid the cast.
     * @see  #nextExceptionHandler
     * @return  the next element of this iterator. */
    public Object next() { return nextExceptionHandler(); }
    /** Returns the next element of this iterator, avoiding the cast.
     * @see  #next
     * @return  the next element of this iterator. */
    public ExceptionHandler nextExceptionHandler() {
        if (current == null) throw new NoSuchElementException();
        ExceptionHandler x = current.getHandler();
        current = current.getParent();
        return x;
    }
    /** Returns the index of the next element of this iterator.
     * @return  the index of the next element of this iterator. */
    public int nextIndex() {
        int i=0; ExceptionHandlerList p = root;
        while (p != current) {
            ++i; p = p.getParent();
        }
        return i;
    }
    
    /** Returns true if this iterator has a previous element.
     * @return  true if this iterator has a previous element. */
    public boolean hasPrevious() { return root != current; }
    /** Returns the previous element of this iterator.  Use previousExceptionHandler to avoid the cast.
     * @see  #previousExceptionHandler
     * @return  the previous element of this iterator. */
    public Object previous() { return previousExceptionHandler(); }
    /** Returns the previous element of this iterator, avoiding the cast.
     * @see  #previous
     * @return  the previous element of this iterator. */
    public ExceptionHandler previousExceptionHandler() {
        if (root == current) throw new NoSuchElementException();
        ExceptionHandlerList p = root;
        ExceptionHandlerList q = p.getParent();
        while (q != current) {
            p = q;
            q = q.getParent();
        }
        return p.getHandler();
    }
    /** Returns the index of the previous element of this iterator.
     * @return  the index of the previous element of this iterator. */
    public int previousIndex() { return nextIndex()-1; }
    /** Throws UnsupportedOperationException. (Removing is not supported.)
     * @throws UnsupportedOperationException always
     */
    
    public void remove() { throw new UnsupportedOperationException(); }
    /** Throws UnsupportedOperationException. (Setting is not supported.)
     * @throws UnsupportedOperationException always
     */
    public void set(Object o) { throw new UnsupportedOperationException(); }
    /** Throws UnsupportedOperationException. (Adding is not supported.)
     * @throws UnsupportedOperationException always
     */
    public void add(Object o) { throw new UnsupportedOperationException(); }
    
    /** Return an empty, unmodifiable iterator.
     * @return  an empty, unmodifiable iterator */
    public static ExceptionHandlerIterator getEmptyIterator() { return EMPTY; }
    
    /** The empty basic block iterator.  Immutable. */
    public static final ExceptionHandlerIterator EMPTY = new ExceptionHandlerIterator(null);
}

/*
 * ExceptionHandlerIterator.java
 *
 * Created on April 22, 2001, 12:53 AM
 *
 */

package Compil3r.Quad;
import java.util.*;
import Util.AppendListIterator;

/**
 * Iterator for iterating through exception handlers.  Compatible with ListIterator.
 * @see  ListIterator
 * @see  ExceptionHandler
 * @author  John Whaley
 * @version  $Id$
 */

public class ExceptionHandlerIterator implements ListIterator {

    private final ListIterator iterator;
    
    /** Creates new ExceptionHandlerIterator.
     * @param exs  list of exception handlers to iterate through. */
    public ExceptionHandlerIterator(List/*ExceptionHandler*/ exs) {
        iterator = exs.listIterator();
    }
    /** Creates new ExceptionHandlerIterator.
     * @param exs  list of exception handlers to iterate through.
     * @param parent  parent set of exception handlers, to be iterated through after the given set. */
    public ExceptionHandlerIterator(List/*ExceptionHandler*/ exs, ExceptionHandlerSet parent) {
        ListIterator l2 = parent==null?null:parent.iterator();
        iterator = new AppendListIterator(exs.listIterator(), l2);
    }
    
    /** Returns true if this iterator has a previous element.
     * @return  true if this iterator has a previous element. */
    public boolean hasPrevious() { return iterator.hasPrevious(); }
    /** Returns true if this iterator has a next element.
     * @return  true if this iterator has a next element. */
    public boolean hasNext() { return iterator.hasNext(); }
    /** Returns the previous element of this iterator.  Use previousEH to avoid the cast.
     * @see  previousEH
     * @return  the previous element of this iterator. */
    public Object previous() { return iterator.previous(); }
    /** Returns the next element of this iterator.  Use nextEH to avoid the cast.
     * @see  nextEH
     * @return  the next element of this iterator. */
    public Object next() { return iterator.next(); }
    /** Returns the index of the previous element of this iterator.
     * @return  the index of the previous element of this iterator. */
    public int previousIndex() { return iterator.previousIndex(); }
    /** Returns the index of the next element of this iterator.
     * @return  the index of the next element of this iterator. */
    public int nextIndex() { return iterator.nextIndex(); }
    /** Removes from the list the last element returned by this iterator.
     * @throws  IllegalStateException neither next nor previous have
                been called, or remove or add have been called after
                the last call to next or previous.
     * @throws  UnsupportedOperationException if the remove operation
                is not supported by the underlying list.
     */
    public void remove() { iterator.remove(); }
    /** Replaces the last element returned by next or previous with the specified element.
     * @param o  the element with which to replace the last element
                 returned by next or previous
     * @throws  IllegalStateException neither next nor previous have
                been called, or remove or add have been called after
                the last call to next or previous.
     * @throws  UnsupportedOperationException if the set operation
                is not supported by the underlying list.
     */
    public void set(Object o) { iterator.set(o); }
    /** Inserts the specified element into the list.  The element is inserted
     * immediately before the next element that would be returned by next, if
     * any, and after the next element that would be returned by previous, if
     * any.
     * @param o  the element to insert.
     * @throws  UnsupportedOperationException if the add operation
                is not supported by the underlying list.
     */
    public void add(Object o) { iterator.add(o); }
    
    /** Returns the previous element of this iterator, avoiding the cast.
     * @see  next
     * @return  the previous element of this iterator. */
    public ExceptionHandler previousEH() { return (ExceptionHandler)previous(); }
    /** Returns the next element of this iterator, avoiding the cast.
     * @see  next
     * @return  the next element of this iterator. */
    public ExceptionHandler nextEH() { return (ExceptionHandler)next(); }
    
    /** Return an empty, unmodifiable iterator.
     * @return  an empty, unmodifiable iterator */
    public static ExceptionHandlerIterator getEmptyIterator() { return EMPTY; }
    /** The empty basic block iterator.  Immutable. */
    public static final ExceptionHandlerIterator EMPTY = new ExceptionHandlerIterator(Collections.EMPTY_LIST);
}

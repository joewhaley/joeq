/*
 * BasicBlockIterator.java
 *
 * Created on April 22, 2001, 12:48 AM
 *
 */

package Compil3r.Quad;
import java.util.*;

/**
 * Iterator for iterating through basic blocks.  Compatible with ListIterator.
 * @see  ListIterator
 * @see  BasicBlock
 * @author  John Whaley
 * @version  $Id$
 */

public class BasicBlockIterator implements ListIterator {

    /** The ListIterator that this iterator wraps. */
    private final ListIterator bb_iterator;
    
    /** Creates new BasicBlockIterator
     * @param bbs The list of basic blocks to iterate over. */
    public BasicBlockIterator(List/*BasicBlock*/ bbs) {
        bb_iterator = (ListIterator)bbs.listIterator();
    }

    /** Returns true if this iterator has a previous element.
     * @return  true if this iterator has a previous element. */
    public boolean hasPrevious() { return bb_iterator.hasPrevious(); }
    /** Returns true if this iterator has a next element.
     * @return  true if this iterator has a next element. */
    public boolean hasNext() { return bb_iterator.hasNext(); }
    /** Returns the previous element of this iterator.  Use previousBB to avoid the cast.
     * @see  previousBB
     * @return  the previous element of this iterator. */
    public Object previous() { return bb_iterator.previous(); }
    /** Returns the next element of this iterator.  Use nextBB to avoid the cast.
     * @see  nextBB
     * @return  the next element of this iterator. */
    public Object next() { return bb_iterator.next(); }
    /** Returns the index of the previous element of this iterator.
     * @return  the index of the previous element of this iterator. */
    public int previousIndex() { return bb_iterator.previousIndex(); }
    /** Returns the index of the next element of this iterator.
     * @return  the index of the next element of this iterator. */
    public int nextIndex() { return bb_iterator.nextIndex(); }
    /** Removes from the list the last element returned by this iterator.
     * @throws  IllegalStateException neither next nor previous have
                been called, or remove or add have been called after
                the last call to next or previous.
     * @throws  UnsupportedOperationException if the remove operation
                is not supported by the underlying list.
     */
    public void remove() { bb_iterator.remove(); }
    /** Replaces the last element returned by next or previous with the specified element.
     * @param o  the element with which to replace the last element
                 returned by next or previous
     * @throws  IllegalStateException neither next nor previous have
                been called, or remove or add have been called after
                the last call to next or previous.
     * @throws  UnsupportedOperationException if the set operation
                is not supported by the underlying list.
     */
    public void set(Object o) { bb_iterator.set(o); }
    /** Inserts the specified element into the list.  The element is inserted
     * immediately before the next element that would be returned by next, if
     * any, and after the next element that would be returned by previous, if
     * any.
     * @param o  the element to insert.
     * @throws  UnsupportedOperationException if the add operation
                is not supported by the underlying list.
     */
    public void add(Object o) { bb_iterator.add(o); }
    
    /** Returns the next element of this iterator, avoiding the cast.
     * @see next
     * @return  the next element of this iterator. */
    public BasicBlock nextBB() { return (BasicBlock)next(); }
    /** Returns the previous element of this iterator, avoiding the cast.
     * @see previous
     * @return  the previous element of this iterator. */
    public BasicBlock previousBB() { return (BasicBlock)previous(); }

    /** Return an empty, unmodifiable iterator.
     * @return  an empty, unmodifiable iterator */
    public static BasicBlockIterator getEmptyIterator() { return EMPTY; }
    /** The empty basic block iterator.  Immutable. */
    public static final BasicBlockIterator EMPTY = new BasicBlockIterator(Collections.EMPTY_LIST);

}

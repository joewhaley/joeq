/*
 * BackwardQuadIterator.java
 *
 * Created on April 22, 2001, 12:41 AM
 *
 */

package Compil3r.Quad;
import java.util.*;

/**
 * This iterator is used to iterate backwards over a list of quads.
 * It simply reverses all of the next/previous of QuadIterator.
 * @author  John Whaley
 * @see  QuadIterator
 * @version  $Id$
 */

public class BackwardQuadIterator extends QuadIterator {

    /** Creates new BackwardQuadIterator
     * @param quads List of quads to iterate backwards over. */
    public BackwardQuadIterator(List/*Quad*/ quads) {
        super(quads, quads.size()-1);
    }

    /** Returns true if this iterator has a next element.
     * @return   true if this iterator has a next element. */
    public boolean hasNext() { return quad_iterator.hasPrevious(); }
    /** Returns true if this iterator has a previous element.
     * @return   true if this iterator has a previous element. */
    public boolean hasPrevious() { return quad_iterator.hasNext(); }
    /** Returns the next element of this iterator.
     * @return   the next element of this iterator. */
    public Object next() { return quad_iterator.previous(); }
    /** Returns the previous element of this iterator.
     * @return   the previous element of this iterator. */
    public Object previous() { return quad_iterator.next(); }
    /** Returns the index of the next element of this iterator.
     * @return   the index of the next element of this iterator. */
    public int nextIndex() { return quad_iterator.previousIndex(); }
    /** Returns the index of the previous element of this iterator.
     * @return   the index of the previous element of this iterator. */
    public int previousIndex() { return quad_iterator.nextIndex(); }
    
}

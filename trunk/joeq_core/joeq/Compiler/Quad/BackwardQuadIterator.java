/*
 * BackwardQuadIterator.java
 *
 * Created on April 22, 2001, 12:41 AM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import java.util.*;

public class BackwardQuadIterator extends QuadIterator {

    /** Creates new BackwardQuadIterator */
    public BackwardQuadIterator(List quads) {
        super(quads, quads.size()-1);
    }

    public boolean hasNext() { return quad_iterator.hasPrevious(); }
    public boolean hasPrevious() { return quad_iterator.hasNext(); }
    public Object next() { return quad_iterator.previous(); }
    public Object previous() { return quad_iterator.next(); }
    public int nextIndex() { return quad_iterator.previousIndex(); }
    public int previousIndex() { return quad_iterator.nextIndex(); }
    
}

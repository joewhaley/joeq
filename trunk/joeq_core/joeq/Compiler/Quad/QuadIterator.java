/*
 * QuadIterator.java
 *
 * Created on April 22, 2001, 12:28 AM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import java.util.*;

public class QuadIterator implements ListIterator {

    protected final ListIterator quad_iterator;
    
    /** Creates new QuadIterator */
    public QuadIterator(List quads) {
        quad_iterator = (ListIterator)quads.listIterator();
    }
    public QuadIterator(List quads, int index) {
        if (index == -1) index = 0;
        quad_iterator = (ListIterator)quads.listIterator(index);
    }

    public boolean hasPrevious() { return quad_iterator.hasPrevious(); }
    public boolean hasNext() { return quad_iterator.hasNext(); }
    public Object previous() { return quad_iterator.previous(); }
    public Object next() { return quad_iterator.next(); }
    public int previousIndex() { return quad_iterator.previousIndex(); }
    public int nextIndex() { return quad_iterator.nextIndex(); }
    public void remove() { quad_iterator.remove(); }
    public void set(Object o) { quad_iterator.set(o); }
    public void add(Object o) { quad_iterator.add(o); }
    
    public Quad nextQuad() { return (Quad)next(); }
    public Quad previousQuad() { return (Quad)previous(); }

    public static QuadIterator getEmptyIterator() { return EMPTY; }
    public static final QuadIterator EMPTY = new QuadIterator(new LinkedList());
}

/*
 * BasicBlockIterator.java
 *
 * Created on April 22, 2001, 12:48 AM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import java.util.*;

public class BasicBlockIterator implements ListIterator {

    private final ListIterator bb_iterator;
    
    /** Creates new BasicBlockIterator */
    public BasicBlockIterator(List bbs) {
        bb_iterator = (ListIterator)bbs.listIterator();
    }

    public boolean hasPrevious() { return bb_iterator.hasPrevious(); }
    public boolean hasNext() { return bb_iterator.hasNext(); }
    public Object previous() { return bb_iterator.previous(); }
    public Object next() { return bb_iterator.next(); }
    public int previousIndex() { return bb_iterator.previousIndex(); }
    public int nextIndex() { return bb_iterator.nextIndex(); }
    public void remove() { bb_iterator.remove(); }
    public void set(Object o) { bb_iterator.set(o); }
    public void add(Object o) { bb_iterator.add(o); }
    
    public BasicBlock nextBB() { return (BasicBlock)next(); }
    public BasicBlock previousBB() { return (BasicBlock)previous(); }

    public static BasicBlockIterator getEmptyIterator() { return EMPTY; }
    public static final BasicBlockIterator EMPTY = new BasicBlockIterator(new LinkedList());

}

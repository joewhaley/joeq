/*
 * BackwardIterator.java
 *
 * Created on January 17, 2002, 5:42 PM
 */

package Util.Collections;
import java.util.List;
import java.util.ListIterator;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class BackwardIterator implements ListIterator {

    private final ListIterator i;
    public BackwardIterator(ListIterator x) {
        while (x.hasNext()) x.next();
        this.i = x;
    }
    public BackwardIterator(List x) {
        this.i = x.listIterator(x.size());
    }
    
    public int previousIndex() { return i.nextIndex(); }
    public boolean hasNext() { return i.hasPrevious(); }
    public void set(Object obj) { i.set(obj); }
    public Object next() { return i.previous(); }
    public int nextIndex() { return i.previousIndex(); }
    public void remove() { i.remove(); }
    public boolean hasPrevious() { return i.hasNext(); }
    public void add(Object obj) { i.add(obj); i.previous(); }
    public Object previous() { return i.next(); }
    
}

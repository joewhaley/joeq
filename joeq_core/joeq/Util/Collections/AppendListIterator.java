/*
 * AppendListIterator.java
 *
 * Created on April 22, 2001, 11:14 AM
 *
 */

package Util.Collections;

import java.util.ListIterator;
import java.util.NoSuchElementException;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class AppendListIterator implements ListIterator {

    private final ListIterator iterator1;
    private final ListIterator iterator2;
    private boolean which;
    
    /** Creates new AppendListIterator */
    public AppendListIterator(ListIterator iter1, ListIterator iter2) {
        if (iter1 == null) {
            iterator1 = iter2; iterator2 = null;
        } else {
            iterator1 = iter1; iterator2 = iter2;
        }
        which = false;
    }

    public boolean hasPrevious() {
        if (!which || !iterator2.hasPrevious())
            return iterator1.hasPrevious();
        else
            return iterator2.hasPrevious();
    }
    public boolean hasNext() {
        if (which || ((iterator2 != null) && !iterator1.hasNext()))
            return iterator2.hasNext();
        else
            return iterator1.hasNext();
    }
    public Object previous() {
        if (!which) return iterator1.previous();
        else if (iterator2.hasPrevious()) return iterator2.previous();
        else {
            which = false; return iterator1.previous();
        }
    }
    public Object next() {
        if (which) return iterator2.next();
        else if (iterator1.hasNext()) return iterator1.next();
        else if (iterator2 != null) {
            which = true; return iterator2.next();
        } else throw new NoSuchElementException();
    }
    public int previousIndex() {
        if (!which) return iterator1.previousIndex();
        else return iterator1.nextIndex() + iterator2.previousIndex();
    }
    public int nextIndex() {
        if (!which) return iterator1.nextIndex();
        else return iterator1.nextIndex() + iterator2.nextIndex();
    }
    public void remove() {
        if (!which) iterator1.remove();
        else iterator2.remove();
    }
    public void set(Object o) {
        if (!which) iterator1.set(o);
        else iterator2.set(o);
    }
    public void add(Object o) {
        if (!which) iterator1.add(o);
        else iterator2.add(o);
    }

}

/*
 * AppendIterator.java
 *
 * Created on July 10, 2001, 11:14 AM
 * 
 */ 

package Util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class AppendIterator implements Iterator {

    private final Iterator iterator1;
    private final Iterator iterator2;
    private boolean which;
    
    /** Creates new AppendIterator */
    public AppendIterator(Iterator iter1, Iterator iter2) {
        if (iter1 == null) {
            iterator1 = iter2; iterator2 = null;
        } else {
            iterator1 = iter1; iterator2 = iter2;
        }
        which = false;
    }

    public Object next() {
        if (which) {
            return iterator2.next();
        } else if (iterator1.hasNext()) {
            return iterator1.next();
        } else if (iterator2 != null) {
            which = true; return iterator2.next();
        } else throw new NoSuchElementException();
    }
    
    public boolean hasNext() {
        if (which || ((iterator2 != null) && !iterator1.hasNext())) {
            return iterator2.hasNext();
        } else {
            return iterator1.hasNext();
        }
    }
    
    public void remove() {
        if (!which) iterator1.remove();
        else iterator2.remove();
    }
    
}

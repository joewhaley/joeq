/*
 * SingletonIterator.java
 *
 * Created on June 28, 2001, 12:49 PM
 *
 */

package Util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class SingletonIterator extends UnmodifiableIterator implements Iterator {
    Object o; boolean done;
    public SingletonIterator(Object o) { this.o = o; }
    public Object next() { if (done) throw new NoSuchElementException(); done = true; return o; }
    public boolean hasNext() { return !done; }
}

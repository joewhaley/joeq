/*
 * NullIterator.java
 *
 * Created on June 28, 2001, 12:49 PM
 *
 */

package Util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class NullIterator extends UnmodifiableIterator implements Iterator {
    private NullIterator() { }
    public Object next() { throw new NoSuchElementException(); }
    public boolean hasNext() { return false; }
    public static final NullIterator INSTANCE = new NullIterator();
}

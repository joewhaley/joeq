/*
 * AtomicCounter.java
 *
 * Created on April 5, 2001, 12:01 AM
 *
 */

package Util;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class AtomicCounter {

    private int current;
    
    /** Creates new AtomicCounter */
    public AtomicCounter(int initialValue) { current = initialValue-1;}
    public AtomicCounter() { this(0); }

    public synchronized int increment() { return ++current; }
    public synchronized void reset(int v) { current = v-1; }
    
    public int value() { return current+1; }
}

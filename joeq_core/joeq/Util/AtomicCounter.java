/*
 * AtomicCounter.java
 *
 * Created on April 5, 2001, 12:01 AM
 *
 * @author  John Whaley
 * @version 
 */

package Util;

public class AtomicCounter {

    private int current;
    
    /** Creates new AtomicCounter */
    public AtomicCounter(int initialValue) { current = initialValue-1;}
    public AtomicCounter() { this(0); }

    public synchronized int get() { return ++current; }
    
}

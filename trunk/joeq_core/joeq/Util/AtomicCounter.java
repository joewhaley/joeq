// AtomicCounter.java, created Mon Apr  9  1:53:53 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
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
    
    public String toString() { return Integer.toString(value()); }
}

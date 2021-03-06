// TupleIterator.java, created Jan 23, 2005 5:57:51 PM by joewhaley
// Copyright (C) 2005 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.collections;

/**
 * TupleIterator
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class TupleIterator extends UnmodifiableIterator {
    
    public abstract int[] nextTuple(int[] t);
    public abstract int[] nextTuple();
    public Object next() { return nextTuple(); }
    
}

// ArrayIterator.java, created Tue Jun 15 15:59:04 1999 by cananian
// Copyright (C) 1998 C. Scott Ananian <cananian@alumni.princeton.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.

package Util;

import java.util.Iterator;
import java.util.NoSuchElementException;
/**
 * An <code>ArrayIterator</code> iterates over the elements of an array.
 * <p>The <code>remove()</code> method is <b>not</b> implemented.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 */

public class ArrayIterator extends UnmodifiableIterator implements Iterator {
    final Object[] oa;
    int i = 0;

    /** Creates an <code>ArrayEnumerator</code>. */
    public ArrayIterator(Object[] oa) {
        this.oa = oa;
    }
    public boolean hasNext() { return ( i < oa.length ); }
    public Object  next() {
	if (i < oa.length)
	    return oa[i++];
	else
	    throw new NoSuchElementException();
    }
}

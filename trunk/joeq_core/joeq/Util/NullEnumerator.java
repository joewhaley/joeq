// NullEnumerator.java, created Wed Sep 16 15:07:32 1998 by cananian
// Copyright (C) 1998 C. Scott Ananian <cananian@alumni.princeton.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

import java.util.Enumeration;
import java.util.NoSuchElementException;
/**
 * A <code>NullEnumerator</code> enumerates no elements.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 * @deprecated Use harpoon.Util.Default.nullIterator instead.
 */

public class NullEnumerator implements Enumeration {
    /** Creates a <code>NullEnumerator</code>. */
    public NullEnumerator() { }
    
    /** @return <code>false</code> */
    public boolean hasMoreElements() { return false; }
    /** @return <code>null</code> */
    public Object  nextElement() { throw new NoSuchElementException(); }
}

// Pair.java, created Wed Mar  5  0:26:26 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Collections;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Pair extends java.util.AbstractList
    implements java.io.Serializable {
    public Object left, right;
    public Pair(Object left, Object right) {
        this.left = left; this.right = right;
    }
    public int size() { return 2; }
    public Object get(int index) {
        switch(index) {
        case 0: return this.left;
        case 1: return this.right;
        default: throw new IndexOutOfBoundsException();
        }
    }
    public Object set(int index, Object element) {
        Object prev;
        switch(index) {
        case 0: prev=this.left; this.left=element; return prev;
        case 1: prev=this.right; this.right=element; return prev;
        default: throw new IndexOutOfBoundsException();
        }
    }
}

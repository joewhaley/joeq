// Default.java, created Thu Apr  8 02:22:56 1999 by cananian
// Copyright (C) 1999 C. Scott Ananian <cananian@alumni.princeton.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.

package Util;

import java.util.AbstractList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * <code>Default</code> contains one-off or 'standard, no-frills'
 * implementations of simple <code>Iterator</code>s,
 * <code>Enumeration</code>s, and <code>Comparator</code>s.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 */
public abstract class Default  {
    /** A <code>Comparator</code> for objects that implement 
     *   <code>Comparable</code>. */
    public static final Comparator comparator = new SerializableComparator() {
	public int compare(Object o1, Object o2) {
	    if (o1==null && o2==null) return 0;
	    // hack: in JDK1.1 String is not Comparable
	    if (o1 instanceof String && o2 instanceof String)
	       return ((String)o1).compareTo((String)o2);
	    // hack: in JDK1.1 Integer is not Comparable
	    if (o1 instanceof Integer && o2 instanceof Integer)
	       return ((Integer)o1).intValue() - ((Integer)o2).intValue();
	    return (o1==null) ? -((Comparable)o2).compareTo(o1):
	                         ((Comparable)o1).compareTo(o2);
	}
	// this should always be a singleton.
	private Object readResolve() { return Default.comparator; }
    };
    /** An <code>Enumerator</code> over the empty set.
     * @deprecated Use nullIterator. */
    public static final Enumeration nullEnumerator = new Enumeration() {
	public boolean hasMoreElements() { return false; }
	public Object nextElement() { throw new NoSuchElementException(); }
    };
    /** An <code>Iterator</code> over the empty set. */
    public static final Iterator nullIterator = new UnmodifiableIterator() {
	public boolean hasNext() { return false; }
	public Object next() { throw new NoSuchElementException(); }
    };
    /** An <code>Iterator</code> over a singleton set. */
    public static final Iterator singletonIterator(Object o) {
	return Collections.singleton(o).iterator();
    } 
    /** An unmodifiable version of the given iterator. */
    public static final Iterator unmodifiableIterator(final Iterator i) {
	return new UnmodifiableIterator() {
	    public boolean hasNext() { return i.hasNext(); }
	    public Object next() { return i.next(); }
	};
    }
    /** An empty map. Missing from <code>java.util.Collections</code>.*/
    public static final Map EMPTY_MAP = new SerializableMap() {
	public void clear() { }
	public boolean containsKey(Object key) { return false; }
	public boolean containsValue(Object value) { return false; }
	public Set entrySet() { return Collections.EMPTY_SET; }
	public boolean equals(Object o) {
	    if (!(o instanceof Map)) return false;
	    return ((Map)o).size()==0;
	}
	public Object get(Object key) { return null; }
	public int hashCode() { return 0; }
	public boolean isEmpty() { return true; }
	public Set keySet() { return Collections.EMPTY_SET; }
	public Object put(Object key, Object value) {
	    throw new UnsupportedOperationException();
	}
	public void putAll(Map t) {
	    if (t.size()==0) return;
	    throw new UnsupportedOperationException();
	}
	public Object remove(Object key) { return null; }
	public int size() { return 0; }
	public Collection values() { return Collections.EMPTY_SET; }
	public String toString() { return "{}"; }
	// this should always be a singleton.
	private Object readResolve() { return Default.EMPTY_MAP; }
    };
    /** A pair constructor method.  Pairs implement <code>hashCode()</code>
     *  and <code>equals()</code> "properly" so they can be used as keys
     *  in hashtables and etc.  They are implemented as mutable lists of
     *  fixed size 2. */
    public static List pair(final Object left, final Object right) {
	// this can't be an anonymous class because we want to make it
	// serializable.
	return new PairList(left, right);
    }
    private static class PairList extends AbstractList
	implements java.io.Serializable {
	private Object left, right;
	PairList(Object left, Object right) {
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
    /** A serializable comparator. */
    private interface SerializableComparator
	extends Comparator, java.io.Serializable { /* only declare */ }
    /** A serializable map. */
    private interface SerializableMap
	extends Map, java.io.Serializable { /* only declare */ }
}

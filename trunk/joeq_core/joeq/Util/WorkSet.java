// WorkSet.java, created Tue Feb 23 01:18:37 1999 by cananian
// Copyright (C) 1999 C. Scott Ananian <cananian@alumni.princeton.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

import java.util.HashMap;
import java.util.Iterator;

import jq;

/**
 * A <code>WorkSet</code> is a <code>Set</code> offering constant-time
 * access to the first/last element inserted, and an iterator whose speed
 * is not dependent on the total capacity of the underlying hashtable.
 * <p>Conforms to the JDK 1.2 Collections API.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 */
public class WorkSet extends java.util.AbstractSet implements Worklist{
    private /*final*/ HashMap hm;
    private EntryList listhead = EntryList.init(); // header and footer nodes.
    private EntryList listfoot = listhead.next;
    private final static boolean debug=false; // turn on consistency checks.
    
    /** Creates a new, empty <code>WorkSet</code> with a default capacity
     *  and load factor. */
    public WorkSet() {
	hm = new HashMap();
    }
    /** Constructs a new, empty <code>WorkSet</code> with the specified
     *  initial capacity and default load factor. */
    public WorkSet(int initialCapacity) {
	hm = new HashMap(initialCapacity);
    }
    /** Constructs a new, empty <code>WorkSet</code> with the specified
     *  initial capacity and the specified load factor. */
    public WorkSet(int initialCapacity, float loadFactor) {
	hm = new HashMap(initialCapacity, loadFactor);
    }
    /** Constructs a new <code>WorkSet</code> with the contents of the
     *  specified <code>Collection</code>. */
    public WorkSet(java.util.Collection c) {
	// make hash map about twice as big as the collection.
	hm = new HashMap(Math.max(2*c.size(),11));
	addAll(c);
    }

    /** Adds an element to the front of the (ordered) set and returns true,
     *  if the element is not already present in the set.  Makes no change
     *  to the set and returns false if the element is already in the set.
     */
    public boolean addFirst(Object o) {
	if (o==null) throw new NullPointerException();
	if (hm.containsKey(o)) return false;
	EntryList nel = new EntryList(o);
	listhead.add(nel);
	hm.put(o, nel);
	// verify list/set correspondence.
	if (debug) jq.assert(EntryList.equals(listhead, hm.keySet()));
	return true;
    }
    /** Adds an element to the end of the (ordered) set and returns true,
     *  if the element is not already present in the set.  Makes no change
     *  to the set and returns false if the element is already in the set.
     */
    public boolean addLast(Object o) {
	if (o==null) throw new NullPointerException();
	if (hm.containsKey(o)) return false;
	EntryList nel = new EntryList(o);
	listfoot.prev.add(nel);
	hm.put(o, nel);
	// verify list/set correspondence.
	if (debug) jq.assert(EntryList.equals(listhead, hm.keySet()));
	return true;
    }
    /** Returns the first element in the ordered set. */
    public Object getFirst() {
	if (isEmpty()) throw new java.util.NoSuchElementException();
	return listhead.next.o;
    }
    /** Returns the last element in the ordered set. */
    public Object getLast() {
	if (isEmpty()) throw new java.util.NoSuchElementException();
	return listfoot.prev.o;
    }
    /** Removes the first element in the ordered set and returns it. */
    public Object removeFirst() {
	if (isEmpty()) throw new java.util.NoSuchElementException();
	Object o = listhead.next.o;
	hm.remove(o);
	listhead.next.remove();
	return o;
    }
    /** Removes the last element in the ordered set and returns it. */
    public Object removeLast() {
	if (isEmpty()) throw new java.util.NoSuchElementException();
	Object o = listfoot.prev.o;
	hm.remove(o);
	listfoot.prev.remove();
	return o;
    }

    /** Looks at the object as the top of this <code>WorkSet</code>
     *  (treating it as a <code>Stack</code>) without removing it
     *  from the set/stack. */
    public Object peek() { return getLast(); }

    /** Removes some item from this and return it (Worklist adaptor
	method). 
	<BR> <B>modifies:</B> <code>this</code>
	<BR> <B>effects:</B> If there exists an <code>Object</code>,
	                     <code>item</code>, that is an element of
			     <code>this</code>, removes
			     <code>item</code> from <code>this</code>
			     and returns <code>item</code>. Else does
			     nothing.
    */
    public Object pull() { return removeLast(); }

    /** Removes the item at the top of this <code>WorkSet</code>
     *  (treating it as a <code>Stack</code>) and returns that object
     *  as the value of this function. */
    public Object pop() { return removeLast(); }

    /** Pushes item onto the top of this <code>WorkSet</code> (treating
     *  it as a <code>Stack</code>), if it is not already there.
     *  If the <code>item</code> is already in the set/on the stack,
     *  then this method does nothing.
	<BR> <B>modifies:</B> <code>this</code>
	<BR> <B>effects:</B> If <code>item</code> is not already an
	                     element of <code>this</code>, adds
			     <code>item</code> to <code>this</code>.
			     Else does nothing. 
    */
    public void push(Object item) {
	this.add(item);
    }

    /** Adds the object to the set and returns true if the element
     *  is not already present.  Otherwise makes no change to the
     *  set and returns false. */
    public boolean add(Object o) { return addLast(o); }

    /** Removes all elements from the set. */
    public void clear() {
	hm.clear(); listhead.next = listfoot; listfoot.prev = listhead;
    }

    /** Determines if this contains an item.
	<BR> <B>effects:</B> If <code>o</code> is an element of 
	                     <code>this</code>, returns true.
			     Else returns false.
    */
    public boolean contains(Object o) {
	return hm.containsKey(o);
    }

    /** Determines if there are any more items left in this. 
	<BR> <B>effects:</B> If <code>this</code> has any elements,
	                     returns true.  Else returns false.
    */
    public boolean isEmpty() {
	return (listhead.next == listfoot);
    }
    /** Efficient set iterator. */
    public Iterator iterator() {
	return new Iterator() {
	    EntryList elp = listhead;
	    public boolean hasNext() {
		return elp.next.next!=null; /* remember to skip FOOTER node */
	    }
	    public Object next() {
		if (!hasNext())
		    throw new java.util.NoSuchElementException();
		Object o=elp.next.o; elp=elp.next; return o;
	    }
	    public void remove() {
		if (elp==listhead) throw new IllegalStateException();
		hm.remove(elp.o);
		(elp = elp.prev).next.remove();
		// verify list/set correspondence.
		if (debug) jq.assert(EntryList.equals(listhead, hm.keySet()));
	    }
	};
    }
    public boolean remove(Object o) {
	if (!hm.containsKey(o)) return false;
	// remove from hashmap
	EntryList elp = (EntryList) hm.remove(o);
	// remove from linked list.
	elp.remove();
	// verify list/set correspondence.
	if (debug) jq.assert(EntryList.equals(listhead, hm.keySet()));
	return true;
    }
    public int size() { return hm.size(); }

    // INNER CLASS. -------------------------------------------
    private static final class EntryList {
	final Object o;
	EntryList prev=null, next=null;
	EntryList(Object o) { this.o = o; }

	public String toString() {
	    StringBuffer sb = new StringBuffer("[");
	    for (EntryList elp = this; elp!=null; elp=elp.next) {
		sb.append(elp.o);
		if (elp.next!=null)
		    sb.append(", ");
	    }
	    sb.append(']');
	    return sb.toString();
	}
	static boolean equals(EntryList el, java.util.Collection c) {
	    int size=0;
	    // remember that header and footer are skipped!
	    for (EntryList elp=el.next; elp.next!=null; elp=elp.next, size++)
		if (!c.contains(elp.o)) return false;
	    if (size!=c.size()) return false;
	    return true;
	}
	// utility.
	/** Remove this entry from the list. */
	void remove() {
	    // always a predecessor and successor.
	    this.prev.next = this.next;
	    this.next.prev = this.prev;
	    this.next = this.prev = null; // safety.
	}
	/** Link in the supplied entry after this one. */
	void add(EntryList nel) {
	    nel.next = this.next;
	    nel.prev = this;
	    this.next = nel.next.prev = nel;
	}
	// return a list with only a header and footer node.
	static EntryList init() {
	    EntryList header = new EntryList(null);
	    EntryList footer = new EntryList(null);
	    header.next = footer;
	    footer.prev = header;
	    return header;
	}
    }
}

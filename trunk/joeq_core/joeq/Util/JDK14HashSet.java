/*
 * @(#)JDK14HashSet.java	1.25 01/12/03
 *
 * Copyright 2002 Sun Microsystems, Inc. All rights reserved.
 * SUN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package Util;
import java.util.*;

/**
 * This class implements the <tt>Set</tt> interface, backed by a hash table
 * (actually a <tt>HashMap</tt> instance).  It makes no guarantees as to the
 * iteration order of the set; in particular, it does not guarantee that the
 * order will remain constant over time.  This class permits the <tt>null</tt>
 * element.<p>
 *
 * This class offers constant time performance for the basic operations
 * (<tt>add</tt>, <tt>remove</tt>, <tt>contains</tt> and <tt>size</tt>),
 * assuming the hash function disperses the elements properly among the
 * buckets.  Iterating over this set requires time proportional to the sum of
 * the <tt>JDK14HashSet</tt> instance's size (the number of elements) plus the
 * "capacity" of the backing <tt>HashMap</tt> instance (the number of
 * buckets).  Thus, it's very important not to set the initial capacity too
 * high (or the load factor too low) if iteration performance is important.<p>
 *
 * <b>Note that this implementation is not synchronized.</b> If multiple
 * threads access a set concurrently, and at least one of the threads modifies
 * the set, it <i>must</i> be synchronized externally.  This is typically
 * accomplished by synchronizing on some object that naturally encapsulates
 * the set.  If no such object exists, the set should be "wrapped" using the
 * <tt>Collections.synchronizedSet</tt> method.  This is best done at creation
 * time, to prevent accidental unsynchronized access to the <tt>JDK14HashSet</tt>
 * instance:
 * 
 * <pre>
 *     Set s = Collections.synchronizedSet(new JDK14HashSet(...));
 * </pre><p>
 *
 * The iterators returned by this class's <tt>iterator</tt> method are
 * <i>fail-fast</i>: if the set is modified at any time after the iterator is
 * created, in any way except through the iterator's own <tt>remove</tt>
 * method, the Iterator throws a <tt>ConcurrentModificationException</tt>.
 * Thus, in the face of concurrent modification, the iterator fails quickly
 * and cleanly, rather than risking arbitrary, non-deterministic behavior at
 * an undetermined time in the future.
 * 
 * <p>Note that the fail-fast behavior of an iterator cannot be guaranteed
 * as it is, generally speaking, impossible to make any hard guarantees in the
 * presence of unsynchronized concurrent modification.  Fail-fast iterators
 * throw <tt>ConcurrentModificationException</tt> on a best-effort basis. 
 * Therefore, it would be wrong to write a program that depended on this
 * exception for its correctness: <i>the fail-fast behavior of iterators
 * should be used only to detect bugs.</i>
 *
 * @author  Josh Bloch
 * @version 1.25, 12/03/01
 * @see	    Collection
 * @see	    Set
 * @see	    TreeSet
 * @see	    Collections#synchronizedSet(Set)
 * @see	    JDK14HashMap
 * @since   1.2
 */

public class JDK14HashSet extends AbstractSet
		     implements Set, Cloneable, java.io.Serializable
{
    static final long serialVersionUID = -5024744406713321676L;

    private transient JDK14HashMap map;

    // Dummy value to associate with an Object in the backing Map
    private static final Object PRESENT = new Object();

    /**
     * Constructs a new, empty set; the backing <tt>JDK14HashMap</tt> instance has
     * default initial capacity (16) and load factor (0.75).
     */
    public JDK14HashSet() {
	map = new JDK14HashMap();
    }

    /**
     * Constructs a new set containing the elements in the specified
     * collection.  The <tt>JDK14HashMap</tt> is created with default load factor
     * (0.75) and an initial capacity sufficient to contain the elements in
     * the specified collection.
     *
     * @param c the collection whose elements are to be placed into this set.
     * @throws NullPointerException   if the specified collection is null.
     */
    public JDK14HashSet(Collection c) {
	map = new JDK14HashMap(Math.max((int) (c.size()/.75f) + 1, 16));
	addAll(c);
    }

    /**
     * Constructs a new, empty set; the backing <tt>JDK14HashMap</tt> instance has
     * the specified initial capacity and the specified load factor.
     *
     * @param      initialCapacity   the initial capacity of the hash map.
     * @param      loadFactor        the load factor of the hash map.
     * @throws     IllegalArgumentException if the initial capacity is less
     *             than zero, or if the load factor is nonpositive.
     */
    public JDK14HashSet(int initialCapacity, float loadFactor) {
	map = new JDK14HashMap(initialCapacity, loadFactor);
    }

    /**
     * Constructs a new, empty set; the backing <tt>JDK14HashMap</tt> instance has
     * the specified initial capacity and default load factor, which is
     * <tt>0.75</tt>.
     *
     * @param      initialCapacity   the initial capacity of the hash table.
     * @throws     IllegalArgumentException if the initial capacity is less
     *             than zero.
     */
    public JDK14HashSet(int initialCapacity) {
	map = new JDK14HashMap(initialCapacity);
    }

    /**
     * Constructs a new, empty linked hash set.  (This package private
     * constructor is only used by LinkedHashSet.) The backing 
     * JDK14HashMap instance is a LinkedHashMap with the specified initial
     * capacity and the specified load factor.
     *
     * @param      initialCapacity   the initial capacity of the hash map.
     * @param      loadFactor        the load factor of the hash map.
     * @param      dummy             ignored (distinguishes this
     *             constructor from other int, float constructor.)
     * @throws     IllegalArgumentException if the initial capacity is less
     *             than zero, or if the load factor is nonpositive.
     */
    JDK14HashSet(int initialCapacity, float loadFactor, boolean dummy) {
	map = new LinkedHashMap(initialCapacity, loadFactor);
    }

    /**
     * Returns an iterator over the elements in this set.  The elements
     * are returned in no particular order.
     *
     * @return an Iterator over the elements in this set.
     * @see ConcurrentModificationException
     */
    public Iterator iterator() {
	return map.keySet().iterator();
    }

    /**
     * Returns the number of elements in this set (its cardinality).
     *
     * @return the number of elements in this set (its cardinality).
     */
    public int size() {
	return map.size();
    }

    /**
     * Returns <tt>true</tt> if this set contains no elements.
     *
     * @return <tt>true</tt> if this set contains no elements.
     */
    public boolean isEmpty() {
	return map.isEmpty();
    }

    /**
     * Returns <tt>true</tt> if this set contains the specified element.
     *
     * @param o element whose presence in this set is to be tested.
     * @return <tt>true</tt> if this set contains the specified element.
     */
    public boolean contains(Object o) {
	return map.containsKey(o);
    }

    /**
     * Adds the specified element to this set if it is not already
     * present.
     *
     * @param o element to be added to this set.
     * @return <tt>true</tt> if the set did not already contain the specified
     * element.
     */
    public boolean add(Object o) {
	return map.put(o, PRESENT)==null;
    }

    /**
     * Removes the specified element from this set if it is present.
     *
     * @param o object to be removed from this set, if present.
     * @return <tt>true</tt> if the set contained the specified element.
     */
    public boolean remove(Object o) {
	return map.remove(o)==PRESENT;
    }

    /**
     * Removes all of the elements from this set.
     */
    public void clear() {
	map.clear();
    }

    /**
     * Returns a shallow copy of this <tt>JDK14HashSet</tt> instance: the elements
     * themselves are not cloned.
     *
     * @return a shallow copy of this set.
     */
    public Object clone() {
	try { 
	    JDK14HashSet newSet = (JDK14HashSet)super.clone();
	    newSet.map = (JDK14HashMap)map.clone();
	    return newSet;
	} catch (CloneNotSupportedException e) { 
	    throw new InternalError();
	}
    }

    /**
     * Save the state of this <tt>JDK14HashSet</tt> instance to a stream (that is,
     * serialize this set).
     *
     * @serialData The capacity of the backing <tt>JDK14HashMap</tt> instance
     *		   (int), and its load factor (float) are emitted, followed by
     *		   the size of the set (the number of elements it contains)
     *		   (int), followed by all of its elements (each an Object) in
     *             no particular order.
     */
    private synchronized void writeObject(java.io.ObjectOutputStream s)
        throws java.io.IOException {
	// Write out any hidden serialization magic
	s.defaultWriteObject();

        // Write out JDK14HashMap capacity and load factor
        s.writeInt(map.capacity());
        s.writeFloat(map.loadFactor());

        // Write out size
        s.writeInt(map.size());

	// Write out all elements in the proper order.
	for (Iterator i=map.keySet().iterator(); i.hasNext(); )
            s.writeObject(i.next());
    }

    /**
     * Reconstitute the <tt>JDK14HashSet</tt> instance from a stream (that is,
     * deserialize it).
     */
    private synchronized void readObject(java.io.ObjectInputStream s)
        throws java.io.IOException, ClassNotFoundException {
	// Read in any hidden serialization magic
	s.defaultReadObject();

        // Read in JDK14HashMap capacity and load factor and create backing JDK14HashMap
        int capacity = s.readInt();
        float loadFactor = s.readFloat();
        map = (this instanceof LinkedHashSet ?
               new LinkedHashMap(capacity, loadFactor) :
               new JDK14HashMap(capacity, loadFactor));

        // Read in size
        int size = s.readInt();

	// Read in all elements in the proper order.
	for (int i=0; i<size; i++) {
            Object e = s.readObject();
            map.put(e, PRESENT);
        }
    }
}

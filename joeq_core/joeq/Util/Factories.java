// Factories.java, created Tue Oct 19 23:21:25 1999 by pnkfelix
// Copyright (C) 1999 Felix S. Klock II <pnkfelix@mit.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import Main.jq;

/** <code>Factories</code> consists exclusively of static methods that
    operate on or return <code>CollectionFactory</code>s. 
 
    @author  Felix S. Klock II <pnkfelix@mit.edu>
    @version $Id$
 */
public final class Factories {
    
    /** Private ctor so no one will instantiate this class. */
    private Factories() {
        
    }
    
    /** A <code>MapFactory</code> that generates <code>HashMap</code>s. */ 
    public static final MapFactory hashMapFactory = new SerialMapFactory() {
            public java.util.Map makeMap(java.util.Map map) {
                return new java.util.HashMap(map);
            }
    };
    
    /** A <code>SetFactory</code> that generates <code>HashSet</code>s. */
    public static final SetFactory hashSetFactory = new SerialSetFactory() {
            public java.util.Set makeSet(java.util.Collection c) {
                return new java.util.HashSet(c);
            }
            public Set makeSet(int i) {
                return new java.util.HashSet(i);
            }
    };
    
    /** A <code>SetFactory</code> that generates <code>WorkSet</code>s. */
    private static final SetFactory workSetFactory = new SerialSetFactory() {
            public java.util.Set makeSet(java.util.Collection c) {
                return new WorkSet(c);
            }
            public Set makeSet(int i) {
                return new WorkSet(i);
            }
    };
    
    /** A <code>SetFactory</code> that generates
        <code>LinearSet</code>s backed by <code>ArrayList</code>s. */
    public static final SetFactory linearSetFactory = new SerialSetFactory() {
        public java.util.Set makeSet(java.util.Collection c) {
            Set ls;
            if (c instanceof Set) {
                ls = new LinearSet((Set)c);
            } else {
                ls = new LinearSet(c.size());
                ls.addAll(c);
            }
            return ls;
        }
        public Set makeSet(int i) {
            return new LinearSet(i);
        }
    };

    /** A <code>SetFactory</code> that generates <code>TreeSet</code>s. */
    public static final SetFactory treeSetFactory = new SerialSetFactory() {
        public java.util.Set makeSet(java.util.Collection c) {
            return new java.util.TreeSet(c);
        }
    };
    
    /** A <code>ListFactory</code> that generates <code>LinkedList</code>s. */
    public static final ListFactory linkedListFactory=new SerialListFactory() {
            public java.util.List makeList(java.util.Collection c) {
                return new java.util.LinkedList(c);
            }
    };

    /** Returns a <code>ListFactory</code> that generates
        <code>ArrayList</code>s. */
    public static ListFactory arrayListFactory = new SerialListFactory() {
            public java.util.List makeList(java.util.Collection c) {
                return new java.util.ArrayList(c);
            }
        public List makeList(int i) {
            return new java.util.ArrayList(i);
        }

    };

    /** Returns a <code>CollectionFactory</code> that generates
        synchronized (thread-safe) <code>Collection</code>s.  
        The <code>Collection</code>s generated are backed by the 
        <code>Collection</code>s generated by <code>cf</code>. 
        @see Collections#synchronizedCollection
    */
    public static CollectionFactory
        synchronizedCollectionFactory(final CollectionFactory cf) { 
        return new SerialCollectionFactory() {
            public java.util.Collection makeCollection(Collection c) {
                return Collections.synchronizedCollection
                    (cf.makeCollection(c));
            }
        };
    }

    /** Returns a <code>SetFactory</code> that generates synchronized
        (thread-safe) <code>Set</code>s.  The <code>Set</code>s
        generated are backed by the <code>Set</code>s generated by
        <code>sf</code>. 
        @see Collections#synchronizedSet
    */
    public static SetFactory 
        synchronizedSetFactory(final SetFactory sf) {
        return new SerialSetFactory() {
            public java.util.Set makeSet(Collection c) {
                return Collections.synchronizedSet(sf.makeSet(c));
            }
        };
    }

    /** Returns a <code>ListFactory</code> that generates synchronized
        (thread-safe) <code>List</code>s.   The <code>List</code>s
        generated are backed by the <code>List</code>s generated by
        <code>lf</code>. 
        @see Collections#synchronizedList
    */
    public static ListFactory
        synchronizedListFactory(final ListFactory lf) {
        return new SerialListFactory() {
            public java.util.List makeList(Collection c) {
                return Collections.synchronizedList(lf.makeList(c));
            }
        };
    }

    /** Returns a <code>MapFactory</code> that generates synchronized
        (thread-safe) <code>Map</code>s.  The <code>Map</code>s
        generated are backed by the <code>Map</code> generated by
        <code>mf</code>.
        @see Collections#synchronizedMap
    */
    public static MapFactory
        synchronizedMapFactory(final MapFactory mf) {
        return new SerialMapFactory() {
            public java.util.Map makeMap(java.util.Map map) {
                return Collections.synchronizedMap(mf.makeMap(map));
            }
        };
    }

    public static CollectionFactory 
        noNullCollectionFactory(final CollectionFactory cf) {
        return new SerialCollectionFactory() {
            public java.util.Collection makeCollection(final Collection c) {
                jq.Assert(noNull(c));
                final Collection back = cf.makeCollection(c);
                return new CollectionWrapper(back) {
                    public boolean add(Object o) {
                        jq.Assert(o != null);
                        return super.add(o);
                    }
                    public boolean addAll(Collection c2) {
                        jq.Assert(Factories.noNull(c2));
                        return super.addAll(c2);
                    }
                };
            }
        };
    }

    private static boolean noNull(Collection c) {
        Iterator iter = c.iterator();
        while(iter.hasNext()) {
            if(iter.next() == null) return false;
        }
        return true;
    }

    // private classes to add java.io.Serializable to *Factories.
    // if we could make anonymous types w/ multiple inheritance, we wouldn't
    // need these.
    private static abstract class SerialMapFactory
        extends MapFactory implements java.io.Serializable { }
    private static abstract class SerialSetFactory
        extends SetFactory implements java.io.Serializable { }
    private static abstract class SerialListFactory
        extends ListFactory implements java.io.Serializable { }
    private static abstract class SerialCollectionFactory
        extends CollectionFactory implements java.io.Serializable { }
}

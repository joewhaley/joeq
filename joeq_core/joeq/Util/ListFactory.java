// ListFactory.java, created Tue Oct 19 22:39:10 1999 by pnkfelix
// Copyright (C) 1999 Felix S. Klock II <pnkfelix@mit.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.AbstractList;
import java.util.Arrays;

/** <code>ListFactory</code> is a <code>List</code> generator.
    Subclasses should implement constructions of specific types of  
    <code>List</code>s.  <code>ListFactory</code> also has a set of
    static helper methods for building <code>List</code> objects. 

 * 
 * @author  Felix S. Klock II <pnkfelix@mit.edu>
 * @version $Id$
 */
public abstract class ListFactory extends CollectionFactory {
    
    /** Creates a <code>ListFactory</code>. */
    public ListFactory() {
        
    }
    
    public final Collection makeCollection(int initCapacity) {
        return makeList(initCapacity);
    }

    public final Collection makeCollection(Collection c) {
        return makeList(c);
    }

    /** Generates a new, mutable, empty <code>List</code>. */
    public final List makeList() {
        return makeList(Collections.EMPTY_LIST);
    }

    /** Generates a new, mutable, empty <code>List</code>, using 
        <code>initialCapacity</code> as a hint to use for the capacity
        for the produced <code>List</code>. */
    public List makeList(int initialCapacity) {
        return makeList();
    }

    /** Generates a new mutable <code>List</code>, using the elements
        of <code>c</code> as a template for its initial contents. 
    */
    public abstract List makeList(Collection c); 


    /** Creates and returns an unmodifiable <code>List</code> view of
        the list made from connecting <code>lists</code> together in
        order. 
        <BR> <B>requires:</B> <code>lists</code> is a
             <code>List</code> of <code>List</code>s.
        <BR> <B>effects:</B> 
        <pre>let l0 = (List) lists.get(0)
                 l1 = (List) lists.get(1)
                 ...
                 ln = (List) lists.get(n) where n is lists.size()-1
             returns a list view
                 [ l0.get(0) , l0.get(1), ... , l0.get(l0.size()-1), 
                   l1.get(0) , l1.get(1), ... , l1.get(l1.size()-1),
                   ...
                   ln.get(0) , ln.get(1), ... , ln.get(ln.size()-1) ]
        </pre>
        Note that not only changes to the elements of
        <code>lists</code> are reflected in the returned
        <code>List</code>, but even changes to <code>lists</code>
        itself (adding or removing lists) are also reflected.
    */
    public static List concatenate(final List lists) {
        return new AbstractList(){
            public Object get(int index) {
                int origIndex = index;
                int totalSize = 0;
                if (index < 0) 
                    throw new IndexOutOfBoundsException(""+origIndex+" < 0"); 
                int lindex = 0;
                List l = (List) lists.get(lindex);
                totalSize += l.size();
                
                while(true) {
                    if (index < l.size()) {
                        return l.get(index);
                    } else {
                        index -= l.size();
                        lindex++; 
                        if(lindex < lists.size()) {
                            l = (List) lists.get(lindex);
                            totalSize += l.size();
                        } else {
                            throw new IndexOutOfBoundsException
                                (""+origIndex+" > "+totalSize); 
                        }
                    }
                }
            }
            public int size() {
                int sz = 0; 
                for(int i=0; i<lists.size(); i++) {
                    sz += ((List)lists.get(i)).size();
                }
                return sz;
            }
        };
        
    }

    /** Creates and returns an unmodifiable <code>List</code> view of
        the list made from connecting <code>lists</code> together in
        order. 
        <BR> <B>effects:</B> 
        <pre>let l0 = lists[0]
                 l1 = lists[1]
                 ...
                 ln = lists[n] where n is lists.length-1
             returns a list view
                 [ l0.get(0) , l0.get(1), ... , l0.get(l0.size()-1), 
                   l1.get(0) , l1.get(1), ... , l1.get(l1.size()-1),
                   ...
                   ln.get(0) , ln.get(1), ... , ln.get(ln.size()-1) ]
        </pre>
        Note that changes to the elements of <code>lists</code> are
        reflected in <code>this</code>. 

    */
    public static List concatenate(final List[] lists) {
        return concatenate(Arrays.asList(lists));
    }

    /** Creates and returns an immutable <code>List</code> of one element. 
        <BR> <B>effects:</B> returns the list [ o ]
     */
    public static List singleton(final Object o) {
        return new AbstractList() {
            public Object get(int index) {
                if(index==0) return o;
                throw new IndexOutOfBoundsException
                        (""+index+" is out of bounds for list of size 1"); 
            }
            public int size() { return 1; }
        };
    }

}

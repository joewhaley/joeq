// SetFactory.java, created Tue Oct 19 22:22:44 1999 by pnkfelix
// Copyright (C) 1999 Felix S. Klock II <pnkfelix@mit.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

import java.util.Collection;
import java.util.Collections;
import java.util.Set;

/** <code>SetFactory</code> is a <code>Set</code> generator.
    Subclasses should implement constructions of specific types of
    <code>Set</code>s.
 * 
 * @author  Felix S. Klock II <pnkfelix@mit.edu>
 * @version $Id$
 */
public abstract class SetFactory extends CollectionFactory {
    
    /** Creates a <code>SetFactory</code>. */
    public SetFactory() {
        super();
    }
    
    public final Collection makeCollection(Collection c) {
        return makeSet(c);
    }

    public final Collection makeCollection(int initCapacity) {
        return makeSet(initCapacity);
    }

    /** Generates a new, mutable, empty <code>Set</code>. */
    public final java.util.Set makeSet() {
        return makeSet(Collections.EMPTY_SET);
    }

    /** Generates a new, mutable, empty <code>Set</code>, using
        <code>initialCapacity</code> as a hint to use for the capacity
        for the produced <code>Set</code>. */
    public java.util.Set makeSet(int initialCapacity) {
        return makeSet();
    }

    /** Generates a new mutable <code>Set</code>, using the elements
        of <code>c</code> as a template for its initial contents. 
    */ 
    public abstract Set makeSet(Collection c);
    
}
 

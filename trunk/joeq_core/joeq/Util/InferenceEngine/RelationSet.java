// RelationSet.java, created Mar 16, 2004 1:28:55 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.AbstractSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * RelationSet
 * 
 * @author jwhaley
 * @version $Id$
 */
public class RelationSet extends AbstractSet {

    Map nameToRelation;
    Relation.Factory factory;
    
    RelationSet(Relation.Factory f) {
        nameToRelation = new HashMap();
        factory = f;
    }
    
    /**
     * @param relationName
     * @param vars
     * @return
     */
    public Relation getOrCreate(String relationName, List vars) {
        Relation r = (Relation) nameToRelation.get(relationName);
        if (r == null) nameToRelation.put(relationName, r = factory.create(relationName, vars));
        return r;
    }

    /* (non-Javadoc)
     * @see java.util.AbstractCollection#size()
     */
    public int size() {
        return nameToRelation.size();
    }

    /* (non-Javadoc)
     * @see java.util.AbstractCollection#iterator()
     */
    public Iterator iterator() {
        return nameToRelation.values().iterator();
    }
    
}

// Relation.java, created Mar 16, 2004 12:39:48 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.List;

import java.io.IOException;

/**
 * Relation
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Relation {
    
    String name;
    List/*<String>*/ fieldNames;
    List/*<FieldDomain>*/ fieldDomains;
    List/*<String>*/ fieldOptions;
    
    Relation negated;
    
    /**
     * @param name
     * @param fieldNames
     * @param fieldDomains
     */
    public Relation(String name, List fieldNames, List fieldDomains, List fieldOptions) {
        super();
        this.name = name;
        this.fieldNames = fieldNames;
        this.fieldDomains = fieldDomains;
        this.fieldOptions = fieldOptions;
    }
    
    public abstract void initialize();
    public abstract void load() throws IOException;
    public abstract void loadTuples() throws IOException;
    public abstract void save() throws IOException;
    public abstract void saveNegated() throws IOException;
    public abstract void saveTuples() throws IOException;
    public abstract void saveNegatedTuples() throws IOException;
    
    public int size() {
        return (int) dsize();
    }
    public abstract double dsize();
    
    /**
     * Return an iterator over the tuples of this relation.
     * 
     * @return iterator of long[]
     */
    public abstract TupleIterator iterator();
    
    /**
     * Return an iterator over the values in the kth field of the
     * relation.  k is zero-based.
     * 
     * @param k  zero-based field number
     * @return iterator of long[]
     */
    public abstract TupleIterator iterator(int k);
    
    /**
     * Return an iterator over the tuples where the kth field has value j.
     * k is zero-based.
     * 
     * @param k  zero-based field number
     * @param j  value
     * @return iterator of long[]
     */
    public abstract TupleIterator iterator(int k, long j);
    
    /**
     * Get the negated form of this relation, or null if it does not exist.
     * 
     * @return  negated version of this relation, or null
     */
    public Relation getNegated() {
        return negated;
    }
    
    /**
     * Get or create the negated form of this relation.
     * 
     * @param solver  solver
     * @return  negated version of this relation
     */
    public Relation makeNegated(Solver solver) {
        if (negated != null) return negated;
        negated = solver.createRelation("!"+name, fieldNames, fieldDomains, fieldOptions);
        negated.negated = this;
        return negated;
    }
    
    public String toString() {
        return name;
    }

    public abstract boolean contains(int k, long v);
    
}

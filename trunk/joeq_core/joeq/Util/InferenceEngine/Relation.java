// Relation.java, created Mar 16, 2004 12:39:48 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.io.IOException;
import java.util.List;

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
    
    public abstract void load() throws IOException;
    public abstract void loadTuples() throws IOException;
    public abstract void save() throws IOException;
    public abstract void saveNegated() throws IOException;
    public abstract void saveTuples() throws IOException;
    public abstract void saveNegatedTuples() throws IOException;
    
    public String toString() {
        return name;
    }
    
}

// Relation.java, created Mar 16, 2004 12:39:48 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.List;

/**
 * Relation
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Relation {
    
    List/*<String>*/ fieldNames;
    List/*<FieldDomain>*/ fieldDomains;
    
    public abstract static class Factory {
        public abstract Relation create(String name, List fields);
    }
    
}

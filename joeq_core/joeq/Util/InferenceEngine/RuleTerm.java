// RuleTerm.java, created Mar 16, 2004 12:42:16 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Iterator;
import java.util.List;

/**
 * RuleTerm
 * 
 * @author jwhaley
 * @version $Id$
 */
public class RuleTerm {
    
    List/*<Variable>*/ variables;
    Relation relation;
    
    /**
     * @param variables
     * @param relation
     */
    public RuleTerm(List variables, Relation relation) {
        super();
        this.variables = variables;
        this.relation = relation;
    }
    
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("(");
        for (Iterator i = variables.iterator(); i.hasNext(); ) {
            sb.append(i.next());
            if (i.hasNext()) sb.append(",");
        }
        sb.append(") in ");
        sb.append(relation);
        return sb.toString();
    }
    
}

// Variable.java, created Mar 16, 2004 12:43:38 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

/**
 * Variable
 * 
 * @author jwhaley
 * @version $Id$
 */
public class Variable {
    
    String name;
    
    /**
     * @param name
     */
    public Variable(String name) {
        super();
        this.name = name;
    }
    
    /**
     * @return Returns the name.
     */
    public String getName() {
        return name;
    }
    
    /**
     * @param name The name to set.
     */
    public void setName(String name) {
        this.name = name;
    }
    
}

// Constant.java, created Mar 17, 2004 8:30:37 AM by joewhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

/**
 * Constant
 * 
 * @author John Whaley
 * @version $Id$
 */
public class Constant extends Variable {

    long value;
    
    /**
     * 
     */
    public Constant(long value) {
        super(Long.toString(value));
        this.value = value;
    }

}

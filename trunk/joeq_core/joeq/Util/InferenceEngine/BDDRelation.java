// BDDRelation.java, created Mar 16, 2004 12:40:26 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.io.IOException;
import java.util.List;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDFactory;

/**
 * BDDRelation
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BDDRelation extends Relation {
    
    BDDFactory bdd;
    BDD relation;
    
    List/*<BDDDomain>*/ domains;
    
    /**
     * @param bdd
     */
    public BDDRelation(BDDFactory bdd) {
        super();
        this.bdd = bdd;
        this.relation = bdd.zero();
    }
    
    public void load(String filename) throws IOException {
        relation.free();
        relation = bdd.load(filename);
    }
    
}

//PointerAnalysisResults.java, created Mon Sep 22 17:38:25 2003 by joewhaley
//Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
//Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.util.Set;

import Clazz.jq_Method;

/**
 * This interface summarizes all the relevant results of the 
 * external pointer analysis in one convenient place.
 * 
 * @author Vladimir Livshits
 * @author John Whaley
 * @version $Id$
 */
public interface PointerAnalysisResults {
    
    /*
     * This interface summarizes all the relevant results of the 
     * external pointer analysis in one convenient place.
     * 
     * The following things are necessary:
     */
     
    //-------------- 1. Transitive MOD and REF sets for each call       --------------//
    
    /** Returns the set of potentially-modified locations of the
     * given call (and transitively any calls the target may make).  Each
     * location is represented by an SSALocation.
     * 
     * a.f = b;  everything that 'a.f' can point to
     */
    Set/*<SSALocation>*/ mod(ProgramLocation call);
    
    /**
     *  Returns the set of potentially-referenced locations of the
     * given call (and transitively any calls the target may make).  Each
     * location is represented by an SSALocation.
     * 
     * a = b.f;  everything that 'b.f' can point to
     */
    Set/*<SSALocation>*/ ref(ProgramLocation call);
    
    
    //-------------- 2. Aliasing of parameters                          --------------//
    
    /**
     * Returns a set of location/contextset pairs of locations that may be
     * aliased with the given location, along with the set of contexts under
     * which each alias can occur.
     */
    Set/*<ContextSet.ContextLocationPair>*/ getAliases(jq_Method method, SSALocation loc);
    
    /**
     * Returns whether the given location may have aliases in the given set of
     * contexts.
     */
    boolean hasAliases(jq_Method method, SSALocation loc, ContextSet contextSet);
    
    /**
     * Returns whether the given location may have aliases in any context.
     */
    boolean hasAliases(jq_Method method, SSALocation loc);
}


/*
 * Created on Sep 19, 2003
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package Compil3r.Analysis.IPA;

import java.util.Collection;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.Operator.Getfield;
import Compil3r.Quad.Operator.Getstatic;
//import Compil3r.Quad.Operator.Invoke;
import Compil3r.Quad.Operator.Putfield;
import Compil3r.Quad.Operator.Putstatic;

public interface PointerAnalysisResults {
	//private ContextSet _contextSet;	
	//PointerAnalysisResults(ContextSet contextSet){
	//	this._contextSet = contextSet; 
	//}
	
	/*
	 * This class summarizes all the relevant results of the 
	 * external pointer analysis in one convenient place.
	 * 
	 * The following things are necessary:
	*/
	 
	//-------------- 1. Transitive MOD and REF sets for each call       --------------//
	/** Returns the transitively modified locations of the call */
	public Collection/*<SSALocation>*/ mod(ProgramLocation call);
	/** Returns the transitively accessible locations of the call */
	public Collection/*<SSALocation>*/ ref(ProgramLocation call);
			
	//-------------- 2. Sets of affected locations for a LOAD or a STORE --------------//
	/** 
	 * 	Each of the methods below returns a set of locations pointed to by a 
	 * specific load or store operation 
	 * */
	public Collection/*<SSALocation>*/ pointsTo(Getfield op);
	public Collection/*<SSALocation>*/ pointsTo(Getstatic op);
	public Collection/*<SSALocation>*/ pointsTo(Putfield op);
	public Collection/*<SSALocation>*/ pointsTo(Putstatic op);
	
	//-------------- 3. Aliasing of parameters                          --------------//
	/** Returns a list of locations/contextSet pairs that may be aliased with location loc */
	public Collection/*<Pair<SSALocation, ContextSet> >*/ getAliases(jq_Method method, SSALocation loc);
	/** Returns whether location loc may have aliases in the set of contexts contextSet */
	public boolean hasAliases(jq_Method method, SSALocation loc, ContextSet contextSet);
	/** Returns whether location loc may have aliases in any of the contexts */
	public boolean hasAliases(jq_Method method, SSALocation loc);
};

class ContextSet {
	// TODO: fill in the details of the representation
};
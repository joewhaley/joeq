/*
 * Created on Dec 4, 2003
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package Compil3r.Analysis.IPSSA.Utils;

import java.util.HashSet;
import Compil3r.Analysis.IPSSA.SSAIterator;

/**
 * @author livshits
 *
 * To change the template for this generated type comment go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
public class DefinitionSet extends HashSet {
	public SSAIterator.DefinitionIterator getDefinitionIterator(){
		return new SSAIterator.DefinitionIterator(iterator());
	}
}

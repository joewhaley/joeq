/*
 * Created on Dec 4, 2003
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package joeq.Compiler.Analysis.IPSSA.Utils;

import java.util.HashSet;
import joeq.Compiler.Analysis.IPSSA.SSAIterator;

/**
 * @author Vladimir Livshits
 * @version $Id$
 * 
 * Strongly typed definition set.
 */
public class DefinitionSet extends HashSet {
    public DefinitionSet(){
        super();
    }
	public SSAIterator.DefinitionIterator getDefinitionIterator(){
		return new SSAIterator.DefinitionIterator(iterator());
	}
}

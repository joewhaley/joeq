// TransferFunction.java, created Thu Apr 25 16:32:26 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Dataflow;

/**
 * TransferFunction
 * 
 * @author John Whaley
 * @version $Id$
 */
public interface TransferFunction {

    Fact apply(Fact f);
    
}

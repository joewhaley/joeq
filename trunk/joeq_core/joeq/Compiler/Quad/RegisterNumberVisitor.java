// RegisterNumberVisitor.java, created Jun 15, 2003 2:00:45 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import Util.Collections.IndexMap;
import Util.Templates.ListIterator;

/**
 * RegisterNumberVisitor
 * 
 * @author John Whaley
 * @version $Id$
 */
public class RegisterNumberVisitor extends QuadVisitor.EmptyVisitor {

    IndexMap m = new IndexMap("Register numbers");

    public void visitQuad(Quad q) {
        for (ListIterator.RegisterOperand i = q.getDefinedRegisters().registerOperandIterator();
            i.hasNext(); ) {
            m.get(i.nextRegisterOperand().getRegister());
        }
        for (ListIterator.RegisterOperand i = q.getUsedRegisters().registerOperandIterator();
            i.hasNext(); ) {
            m.get(i.nextRegisterOperand().getRegister());
        }
    }

}
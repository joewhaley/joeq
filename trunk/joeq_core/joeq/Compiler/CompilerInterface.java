// Compil3rInterface.java, created Mon Feb  5 23:23:20 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r;

import joeq.Clazz.jq_CompiledCode;
import joeq.Clazz.jq_Method;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public interface Compil3rInterface {
    jq_CompiledCode compile(jq_Method m);
}

// PrintCFG.java, created Thu Jan 16 10:53:32 2003 by mcmartin
// Copyright (C) 2001-3 jwhaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Main;

import Clazz.jq_Class;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class PrintCFG {
    public static void main(String[] args) {
	jq_Class[] c = new jq_Class[args.length];
	for (int i = 0; i < args.length; i++) {
	    c[i] = Helper.load(args[i]);
	}

	Compil3r.Quad.PrintCFG pass = new Compil3r.Quad.PrintCFG();

	for (int i = 0; i < args.length; i++) {
	    Helper.runPass(c[i], pass);
	}
    }
}

// DebugInterface.java, created Sat Dec 14  2:54:05 2002 by mcmartin
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Runtime;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class DebugInterface {

    public static void debugwrite(String msg) {
        System.err.println(msg);
        return;
    }
    
    public static void debugwriteln(String msg) {
        System.err.println(msg);
        return;
    }
    
    public static void die(int code) {
        new InternalError().printStackTrace();
        System.exit(code);
    }
}

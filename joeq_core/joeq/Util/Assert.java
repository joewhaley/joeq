// Assert.java, created Wed Mar  5  0:26:32 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util;

import joeq.Runtime.Debug;

/**
 * Includes methods for an assertion mechanism.  When an assertion fails, it
 * drops into the debugger (in native mode) or just exits (in hosted mode).
 * 
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Assert {
    
    /**
     * Assert that the given predicate is true.  If it is false, we drop into
     * the debugger (in native mode) or just exit (in hosted mode).
     * 
     * @param b predicate to check
     */
    public static void _assert(boolean b) {
        _assert(b, "");
    }

    /**
     * Assert that the given predicate is true.  If it is false, we print
     * the given reason and drop into the debugger (in native mode) or just exit
     * (in hosted mode).
     * 
     * @param b predicate to check
     * @param reason string to print if the assertion fails
     */
    public static void _assert(boolean b, String reason) {
        if (!b) {
            Debug.writeln("Assertion Failure!");
            Debug.writeln(reason);
            Debug.die(-1);
        }
    }

    /**
     * Print a TODO message and drop into the debugger (in native mode) or just
     * exit (in hosted mode).
     * 
     * @param s message to print
     */
    public static void TODO(String s) {
        Debug.writeln("TODO: " + s);
        Debug.die(-1);
    }

    /**
     * Print a TODO message and drop into the debugger (in native mode) or just
     * exit (in hosted mode).
     */
    public static void TODO() {
        Debug.writeln("TODO");
        Debug.die(-1);
    }

    /**
     * Print an UNREACHABLE message and drop into the debugger (in native mode)
     * or just exit (in hosted mode).
     * 
     * @param s message to print
     */
    public static void UNREACHABLE(String s) {
        Debug.writeln("UNREACHABLE: " + s);
        Debug.die(-1);
    }

    /**
     * Print an UNREACHABLE message and drop into the debugger (in native mode)
     * or just exit (in hosted mode).
     */
    public static void UNREACHABLE() {
        Debug.writeln("BUG! unreachable code reached!");
        Debug.die(-1);
    }
    
}

// MessageUtils.java, created Sun Feb 23  2:02:26 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.sun.misc;

/**
 * MessageUtils
 *
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class MessageUtils {
    public static void toStderr(java.lang.String s) {
        System.err.print(s);
    }
    public static void toStdout(java.lang.String s) {
        System.out.print(s);
    }

}

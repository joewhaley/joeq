// AtomicLongCSImpl.java, created Aug 9, 2003 3:50:36 AM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.sun.misc;

/**
 * AtomicLongCSImpl
 * 
 * @author John Whaley
 * @version $Id$
 */
abstract class AtomicLongCSImpl {
    public boolean attemptUpdate(long a, long b) {
        throw new InternalError();
    }
}

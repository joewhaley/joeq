// Shutdown.java, created Thu Jul  4  4:50:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.lang;

import Run_Time.SystemInterface;
import Util.Assert;

/**
 * Shutdown
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
abstract class Shutdown {
    
    static void halt(int status) {
        SystemInterface.die(status);
        Assert.UNREACHABLE();
    }
    private static void runAllFinalizers() {
        try {
            ClassLib.Common.java.lang.ref.Finalizer.runAllFinalizers();
        } catch (java.lang.Throwable x) {
        }
    }
    
}

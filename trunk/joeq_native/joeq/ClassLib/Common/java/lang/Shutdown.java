/*
 * Shutdown.java
 *
 * Created on February 26, 2001, 8:56 PM
 *
 */

package ClassLib.Common.java.lang;

import Run_Time.SystemInterface;
import Util.Assert;

/*
 * @author  John Whaley
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

/*
 * Win32FileSystem.java
 *
 * Created on January 29, 2001, 2:29 PM
 *
 */

package ClassLib.sun13_win32.java.io;

import Run_Time.SystemInterface;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Win32FileSystem {

    // gets the current directory on the named drive.
    private static String getDriveDirectory(int i) {
        byte[] b = new byte[256];
        int result = SystemInterface.fs_getdcwd(i, b);
        if (result == 0) throw new InternalError();
        String res = SystemInterface.fromCString(Unsafe.addressOf(b));
        // skip "C:"
        if (res.charAt(1) == ':') return res.substring(2);
        else return res;
    }

}

/*
 * FileDescriptor.java
 *
 * Created on January 29, 2001, 1:33 PM
 *
 */

package ClassLib.Common.java.io;

import Run_Time.SystemInterface;

/*
 * @author  John Whaley
 * @version $Id$
 */
abstract class FileDescriptor {
    
    int fd;
    
    public native boolean valid();
    
    public void sync() throws java.io.SyncFailedException {
        if (!this.valid()) throw new java.io.SyncFailedException("invalid file descriptor");
        int result = SystemInterface.file_sync(this.fd);
        if (result != 0) throw new java.io.SyncFailedException("flush failed");
    }
    private static void initIDs() { }
    
}

/*
 * FileDescriptor.java
 *
 * Created on January 29, 2001, 1:33 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.io;

import Run_Time.SystemInterface;
import Run_Time.Reflection;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Bootstrap.PrimordialClassLoader;

abstract class FileDescriptor {
    
    public static void sync(java.io.FileDescriptor dis) throws java.io.SyncFailedException {
        if (!dis.valid()) throw new java.io.SyncFailedException("invalid file descriptor");
        int result = SystemInterface.file_sync(Reflection.getfield_I(dis, _fd));
        if (result != 0) throw new java.io.SyncFailedException("flush failed");
    }
    private static void initIDs(jq_Class clazz) { }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileDescriptor;");
    public static final jq_InstanceField _fd = _class.getOrCreateInstanceField("fd", "I");
}

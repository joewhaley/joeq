/*
 * FileSystem.java
 *
 * Created on January 29, 2001, 2:27 PM
 *
 */

package ClassLib.Common.java.io;

import java.lang.reflect.Method;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Main.jq;
import Run_Time.Reflection;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
abstract class FileSystem {

    public static FileSystem getFileSystem() { return (FileSystem) DEFAULT_FS; }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileSystem;");
    //public static final jq_StaticMethod _getFileSystem = _class.getOrCreateStaticMethod("getFileSystem", "()Ljava/io/FileSystem;");
    
    static final Object DEFAULT_FS;
    static {
        if (!jq.RunningNative) {
            Object o;
            try {
                Class klass = Reflection.getJDKType(_class);
                Method m = klass.getMethod("getFileSystem", null);
                m.setAccessible(true);
                o = m.invoke(null, null);
            } catch (Error x) {
                throw x;
            } catch (Throwable x) {
                Assert.UNREACHABLE();
                o = null;
            }
            DEFAULT_FS = o;
        } else {
            Object o = null;
            Assert.UNREACHABLE();
            DEFAULT_FS = o;
        }
    }
}

/*
 * FileSystem.java
 *
 * Created on January 29, 2001, 2:27 PM
 *
 */

package ClassLib.Common.java.io;

import Clazz.jq_Class;
import Run_Time.Reflection;
import Bootstrap.PrimordialClassLoader;
import java.lang.reflect.Method;
import Main.jq;

/*
 * @author  John Whaley
 * @version 
 */
abstract class FileSystem {

    public static Object getFileSystem() { return DEFAULT_FS; }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileSystem;");
    //public static final jq_StaticMethod _getFileSystem = _class.getOrCreateStaticMethod("getFileSystem", "()Ljava/io/FileSystem;");
    
    static final Object DEFAULT_FS;
    static {
        if (jq.Bootstrapping) {
            Object o;
            try {
                Class klass = Reflection.getJDKType(_class);
                Method m = klass.getMethod("getFileSystem", null);
                m.setAccessible(true);
                o = m.invoke(null, null);
            } catch (Error x) {
                throw x;
            } catch (Throwable x) {
                jq.UNREACHABLE();
                o = null;
            }
            DEFAULT_FS = o;
        } else {
	    Object o = null;
	    jq.UNREACHABLE();
	    DEFAULT_FS = o;
        }
    }
}

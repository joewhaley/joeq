/*
 * String.java
 *
 * Created on January 29, 2001, 10:31 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_win32.java.lang;

import UTF.Utf8;
import Clazz.jq_Class;
import Bootstrap.PrimordialClassLoader;

public abstract class String {

    public static java.lang.String intern(java.lang.String dis) {
        // note: this relies on the caching of String objects in Utf8 class
        return Utf8.get(dis).toString();
    }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/String;");
}

/*
 * TimeZone.java
 *
 * Created on March 11, 2001, 2:45 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_linux.java.util;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;

abstract class TimeZone {

    private static String getSystemTimeZoneID(jq_Class clazz, String javaHome, String region) {
        // TODO: correct time zone name.
        return "America/Los_Angeles";
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/TimeZone;");
}

/*
 * StrictMath.java
 *
 * Created on January 29, 2001, 11:04 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang;

import Bootstrap.PrimordialClassLoader;
import com.imsl.math.JMath;
import Clazz.jq_Class;

abstract class StrictMath extends java.lang.Object {

    // native method implementations
    public static double sin(jq_Class clazz, double a) { return JMath.sin(a); }
    public static double cos(jq_Class clazz, double a) { return JMath.cos(a); }
    public static double tan(jq_Class clazz, double a) { return JMath.tan(a); }
    public static double asin(jq_Class clazz, double a) { return JMath.asin(a); }
    public static double acos(jq_Class clazz, double a) { return JMath.acos(a); }
    public static double atan(jq_Class clazz, double a) { return JMath.atan(a); }
    public static double exp(jq_Class clazz, double a) { return JMath.exp(a); }
    public static double log(jq_Class clazz, double a) { return JMath.log(a); }
    public static double sqrt(jq_Class clazz, double a) { return JMath.sqrt(a); }
    public static double IEEEremainder(jq_Class clazz, double f1, double f2) { return JMath.IEEEremainder(f1, f2); }
    public static double ceil(jq_Class clazz, double a) { return JMath.ceil(a); }
    public static double floor(jq_Class clazz, double a) { return JMath.floor(a); }
    public static double rint(jq_Class clazz, double a) { return JMath.rint(a); }
    public static double atan2(jq_Class clazz, double a, double b) { return JMath.atan2(a, b); }
    public static double pow(jq_Class clazz, double a, double b) { return JMath.pow(a, b); }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/StrictMath;");
}

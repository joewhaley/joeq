/*
 * ObjectStreamClass.java
 *
 * Created on July 7, 2002, 11:52 PM
 */

package ClassLib.Common.java.io;

import Clazz.*;
import Run_Time.Reflection;
import Main.jq;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ObjectStreamClass {
    private static void initNative() {}
    private static void getFieldIDs(java.io.ObjectStreamField[] a, long[] b, long[] c) {
        jq.TODO();
    }
    private static boolean hasStaticInitializer(java.lang.Class c) {
        jq_Type t = Reflection.getJQType(c);
        if (t instanceof jq_Class) {
            return ((jq_Class)t).getClassInitializer() != null;
        }
        return false;
    }
}

// ObjectStreamClass.java, created Mon Jul  8  0:41:49 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.io;

import Clazz.jq_Class;
import Clazz.jq_Type;
import Run_Time.Reflection;
import Util.Assert;

/**
 * ObjectStreamClass
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class ObjectStreamClass {
    private static void initNative() {}
    private static void getFieldIDs(java.io.ObjectStreamField[] a, long[] b, long[] c) {
        Assert.TODO();
    }
    private static boolean hasStaticInitializer(java.lang.Class c) {
        jq_Type t = Reflection.getJQType(c);
        if (t instanceof jq_Class) {
            return ((jq_Class)t).getClassInitializer() != null;
        }
        return false;
    }
}

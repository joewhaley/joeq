// Finalizer.java, created Thu Jul  4  4:50:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.lang.ref;

import Clazz.jq_InstanceMethod;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Run_Time.Reflection;
import UTF.Utf8;
import Util.Assert;

/**
 * Finalizer
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Finalizer {

    public static native void runFinalization();
    public static native void runAllFinalizers();
    
    static void invokeFinalizeMethod(Object o) throws Throwable {
        jq_Reference c = jq_Reference.getTypeOf(o);
        jq_InstanceMethod m = c.getVirtualMethod(new jq_NameAndDesc(Utf8.get("finalize"), Utf8.get("()V")));
        Assert._assert(m != null);
        Reflection.invokeinstance_V(m, o);
    }
}

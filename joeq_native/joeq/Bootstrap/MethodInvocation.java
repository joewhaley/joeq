// MethodInvocation.java, created Sun Mar 11  2:21:10 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Bootstrap;

import java.lang.reflect.InvocationTargetException;

import Clazz.jq_Method;
import Run_Time.Reflection;

/**
 * MethodInvocation
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class MethodInvocation {

    jq_Method method;
    Object[] args;
    
    public MethodInvocation(jq_Method m, Object[] a) {
        this.method = m;
        this.args = a;
    }

    public long invoke() throws Throwable {
        try {
            return Reflection.invoke(method, null, args);
        } catch (InvocationTargetException x) {
            throw x.getTargetException();
        }
    }
    
    public String toString() {
        return "method "+method;
    }
}

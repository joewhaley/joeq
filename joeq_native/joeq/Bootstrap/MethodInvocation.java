/*
 * MethodInvocation.java
 *
 * Created on March 10, 2001, 11:49 AM
 *
 */

package Bootstrap;

import java.lang.reflect.InvocationTargetException;

import Clazz.jq_Method;
import Run_Time.Reflection;

/*
 * @author  John Whaley
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

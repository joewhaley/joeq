/*
 * SecurityManager.java
 *
 * Created on Novemeber 7, 2002, 3:00 PM
 *
 */

package ClassLib.Common.java.lang;

import Clazz.jq_CompiledCode;
import Memory.StackAddress;
import Run_Time.Reflection;
import Run_Time.StackCodeWalker;
import Util.Assert;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class SecurityManager {

    protected java.lang.Class[] getClassContext() {
        StackCodeWalker sw = new StackCodeWalker(null, StackAddress.getBasePointer());
        sw.gotoNext();
        int i;
        for (i=0; sw.hasNext(); ++i, sw.gotoNext()) ;
        java.lang.Class[] classes = new java.lang.Class[i];
        sw = new StackCodeWalker(null, StackAddress.getBasePointer());
        sw.gotoNext();
        for (i=0; sw.hasNext(); ++i, sw.gotoNext()) {
            jq_CompiledCode cc = sw.getCode();
            if (cc == null) classes[i] = null;
            else classes[i] = Reflection.getJDKType(cc.getMethod().getDeclaringClass());
        }
        Assert._assert(i == classes.length);
        return classes;
    }

}

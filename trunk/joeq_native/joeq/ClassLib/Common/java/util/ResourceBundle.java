/*
 * ResourceBundle.java
 *
 * Created on January 29, 2001, 3:00 PM
 *
 */

package ClassLib.Common.java.util;

import Clazz.jq_CompiledCode;
import Main.jq;
import Run_Time.Reflection;
import Run_Time.StackWalker;
import Run_Time.Unsafe;

/**
 * @author  John Whaley
 * @version $Id$
 */
abstract class ResourceBundle {
    
    private static Class[] getClassContext() {
        StackWalker sw = new StackWalker(0, Unsafe.EBP());
        sw.gotoNext();
        int i;
        for (i=0; sw.hasNext(); ++i, sw.gotoNext()) ;
        Class[] classes = new Class[i];
        sw = new StackWalker(0, Unsafe.EBP());
        sw.gotoNext();
        for (i=0; sw.hasNext(); ++i, sw.gotoNext()) {
            jq_CompiledCode cc = sw.getCode();
            if (cc == null) classes[i] = null;
            else classes[i] = Reflection.getJDKType(cc.getMethod().getDeclaringClass());
        }
        jq.Assert(i == classes.length);
        return classes;
    }

}

/*
 * Monitor.java
 *
 * Created on January 16, 2001, 9:58 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_StaticMethod;

public abstract class Monitor {

    public static void monitorenter(Object k) {
        // todo
    }
    public static void monitorexit(Object k) {
        // todo
    }
    
    public static final jq_StaticMethod _monitorenter;
    public static final jq_StaticMethod _monitorexit;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Monitor;");
        _monitorenter = k.getOrCreateStaticMethod("monitorenter", "(Ljava/lang/Object;)V");
        _monitorexit = k.getOrCreateStaticMethod("monitorexit", "(Ljava/lang/Object;)V");
    }

}

/*
 * Created on Feb 23, 2004
 */
package ClassLib.sun15_win32.java.util.concurrent.atomic;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;

/**
 * @author jwhaley
 */
public class AtomicLong {
    private static final long valueOffset;
    
    static {
        jq_Class c = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/concurrent/atomic/AtomicLong;");
        c.prepare();
        valueOffset = ((jq_InstanceField) c.getDeclaredMember("value", "J")).getOffset();
    }
}

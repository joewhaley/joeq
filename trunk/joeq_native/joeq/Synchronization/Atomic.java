/*
 * Atomic.java
 *
 * Created on January 25, 2001, 11:50 AM
 *
 */

package Synchronization;

import Clazz.jq_InstanceField;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Main.jq;

import java.lang.reflect.*;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Atomic {
    
    public static final int cas4(Object o, jq_InstanceField f, int before, int after) {
        if (jq.Bootstrapping) {
            Field f2 = (Field)Reflection.getJDKMember(f);
            f2.setAccessible(true);
            jq.Assert((f2.getModifiers() & Modifier.STATIC) == 0);
            try {
                int v = ((Integer)f2.get(o)).intValue();
                if (v == before) {
                    f2.set(o, new Integer(after));
                    return after;
                } else {
                    return v;
                }
            } catch (IllegalAccessException x) {
                jq.UNREACHABLE();
                return 0;
            }
        } else {
            int address = Unsafe.addressOf(o)+f.getOffset();
            return Unsafe.atomicCas4(address, before, after);
        }
    }
    
}

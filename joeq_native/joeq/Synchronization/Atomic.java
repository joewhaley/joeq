/*
 * Atomic.java
 *
 * Created on January 25, 2001, 11:50 AM
 *
 */

package Synchronization;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import Clazz.jq_InstanceField;
import Main.jq;
import Memory.HeapAddress;
import Run_Time.Reflection;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Atomic {
    
    public static final int cas4(Object o, jq_InstanceField f, int before, int after) {
        if (!jq.RunningNative) {
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
            HeapAddress address = (HeapAddress) HeapAddress.addressOf(o).offset(f.getOffset());
            return address.atomicCas4(before, after);
        }
    }
    
}

/*
 * Atomic.java
 *
 * Created on January 25, 2001, 11:50 AM
 *
 * @author  jwhaley
 * @version 
 */

package Synchronization;

import Clazz.jq_InstanceField;
import Run_Time.Unsafe;
import jq;

import java.lang.reflect.*;

public abstract class Atomic {
    
    public static final int cas4(Object o, jq_InstanceField f, int before, int after) {
        if (jq.Bootstrapping) {
            String fieldName = f.getName().toString();
            Class c = o.getClass();
            while (c != null) {
                Field[] fields = c.getDeclaredFields();
                for (int i=0; i<fields.length; ++i) {
                    Field f2 = fields[i];
                    if (f2.getName().equals(fieldName)) {
                        f2.setAccessible(true);
                        jq.assert((f2.getModifiers() & Modifier.STATIC) == 0);
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
                    }
                }
                c = c.getSuperclass();
            }
            jq.UNREACHABLE("Cannot find field "+f+" in class "+o.getClass());
            return 0;
        } else {
            int address = Unsafe.addressOf(o)+f.getOffset();
            return Unsafe.atomicCas4(address, before, after);
        }
    }
    
}

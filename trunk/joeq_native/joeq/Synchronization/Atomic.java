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
    
    public static final boolean cas4(Object o, jq_InstanceField f, int before, int after) {
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
                            if (((Integer)f2.get(o)).intValue() == before) {
                                f2.set(o, new Integer(after));
                                return true;
                            } else {
                                return false;
                            }
                        } catch (IllegalAccessException x) {
                            jq.UNREACHABLE();
                            return false;
                        }
                    }
                }
                c = c.getSuperclass();
            }
            jq.UNREACHABLE("Cannot find field "+f+" in class "+o.getClass());
            return false;
        } else {
            int address = Unsafe.addressOf(o)+f.getOffset();
            return Unsafe.cas4(address, before, after);
        }
    }
    
}

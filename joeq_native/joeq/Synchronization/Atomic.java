// Atomic.java, created Mon Feb  5 23:23:21 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Synchronization;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import Clazz.jq_InstanceField;
import Main.jq;
import Memory.HeapAddress;
import Run_Time.Reflection;
import Util.Assert;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Atomic {
    
    public static final int cas4(Object o, jq_InstanceField f, int before, int after) {
        if (!jq.RunningNative) {
            Field f2 = (Field)Reflection.getJDKMember(f);
            f2.setAccessible(true);
            Assert._assert((f2.getModifiers() & Modifier.STATIC) == 0);
            try {
                int v = ((Integer)f2.get(o)).intValue();
                if (v == before) {
                    f2.set(o, new Integer(after));
                    return after;
                } else {
                    return v;
                }
            } catch (IllegalAccessException x) {
                Assert.UNREACHABLE();
                return 0;
            }
        } else {
            HeapAddress address = (HeapAddress) HeapAddress.addressOf(o).offset(f.getOffset());
            return address.atomicCas4(before, after);
        }
    }
    
}

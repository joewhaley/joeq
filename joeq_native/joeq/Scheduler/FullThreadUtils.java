// FullThreadUtils.java, created Mon Dec 16 18:57:12 2002 by mcmartin
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Scheduler;

import Bootstrap.ObjectTraverser;
import Clazz.jq_Class;
import Main.jq;
import Run_Time.Reflection;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class FullThreadUtils implements ThreadUtils.Delegate {
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
        if (!ObjectTraverser.IsBootstrapping) return null;
        if (!jq.RunningNative) {
            jq_Class k = Bootstrap.PrimordialClassLoader.getJavaLangThread();
            Clazz.jq_InstanceField f = k.getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
            return (Scheduler.jq_Thread)Reflection.getfield_A(t, f);
        }
        return ((ClassLib.Common.InterfaceImpl)ClassLib.ClassLibInterface.DEFAULT).getJQThread(t);
    }    
}

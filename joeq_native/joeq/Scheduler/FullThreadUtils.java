// FullThreadUtils.java, created Mon Dec 16 18:57:12 2002 by mcmartin
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Scheduler;

import joeq.Class.jq_Class;
import joeq.Main.jq;
import joeq.Runtime.Reflection;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class FullThreadUtils implements ThreadUtils.Delegate {
    public joeq.Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
        if (!jq.RunningNative) {
            if (!jq.IsBootstrapping) return null;
            jq_Class k = joeq.Class.PrimordialClassLoader.getJavaLangThread();
            joeq.Class.jq_InstanceField f = k.getOrCreateInstanceField("jq_thread", "Ljoeq/Scheduler/jq_Thread;");
            return (joeq.Scheduler.jq_Thread)Reflection.getfield_A(t, f);
        }
        return ((joeq.ClassLib.Common.InterfaceImpl)joeq.ClassLib.ClassLibInterface.DEFAULT).getJQThread(t);
    }    
}

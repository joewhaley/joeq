// HostedThreadUtils.java, created Mon Dec 16 18:57:13 2002 by mcmartin
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Scheduler;

import Clazz.jq_Class;
import Run_Time.Reflection;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class HostedThreadUtils implements ThreadUtils.Delegate {
    public jq_Thread getJQThread(java.lang.Thread t) {
        jq_Class k = Bootstrap.PrimordialClassLoader.getJavaLangThread();
        Clazz.jq_InstanceField f = k.getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
        return (jq_Thread)Reflection.getfield_A(t, f);
    }    
}

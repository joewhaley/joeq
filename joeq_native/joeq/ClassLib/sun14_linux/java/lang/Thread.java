// Thread.java, created Fri Aug 16 18:11:49 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.sun14_linux.java.lang;

import Scheduler.jq_Thread;

/**
 * Thread
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Thread {

    public final jq_Thread jq_thread;
    
    private void init(java.lang.ThreadGroup g, java.lang.Runnable target, java.lang.String name) {
        this.init(g, target, name, 0L);
    }
    private native void init(java.lang.ThreadGroup g, java.lang.Runnable target, java.lang.String name, long stackSize);
    
    public Thread(java.lang.ThreadGroup group, java.lang.Runnable target, java.lang.String name, long stackSize) {
        java.lang.Object o = this;
        jq_Thread t = new jq_Thread((java.lang.Thread)o);
        this.jq_thread = t;
        this.init(group, target, name, stackSize);
        t.init();
    }
    
}

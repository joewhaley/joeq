/*
 * Thread.java
 *
 * Created on January 29, 2001, 10:21 AM
 *
 */

package ClassLib.sun142_linux.java.lang;

import Scheduler.jq_Thread;

/**
 * @author  John Whaley
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

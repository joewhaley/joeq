/*
 * Thread.java
 *
 * Created on January 29, 2001, 10:21 AM
 *
 */

package ClassLib.apple13_osx.java.lang;
import Scheduler.jq_Thread;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Thread {

    public final jq_Thread jq_thread;

    private void init(java.lang.ThreadGroup g, java.lang.Runnable target, java.lang.String name) {
        this.init(g, target, name, true);
    }
    private native void init(java.lang.ThreadGroup g, java.lang.Runnable target, java.lang.String name, boolean setpriority);
    private static synchronized native int nextThreadNum();

    private Thread(java.lang.ThreadGroup group, java.lang.Runnable target, boolean set_priority) {
        java.lang.Object o = this;
        jq_Thread t = new jq_Thread((java.lang.Thread)o);
        this.jq_thread = t;
	java.lang.String name = "Thread-" + nextThreadNum();
        this.init(group, target, name, false);
        t.init();
    }

    private Thread(java.lang.ThreadGroup group, java.lang.String name, boolean set_priority) {
        java.lang.Object o = this;
        jq_Thread t = new jq_Thread((java.lang.Thread)o);
        this.jq_thread = t;
        this.init(group, null, name, false);
        t.init();
    }
    
}

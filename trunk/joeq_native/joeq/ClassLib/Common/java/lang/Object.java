/*
 * Object.java
 *
 * Created on January 29, 2001, 11:07 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.Common.java.lang;

import Allocator.HeapAllocator;
import Run_Time.Monitor;
import Run_Time.Reflection;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;

public abstract class Object {

    // native method implementations.
    private static void registerNatives() {}
    public final java.lang.Class _getClass() {
        return Reflection.getJDKType(Unsafe.getTypeOf(this));
    }
    public int hashCode() { return java.lang.System.identityHashCode(this); }
    protected java.lang.Object clone() throws CloneNotSupportedException {
        if (this instanceof Cloneable) {
            return HeapAllocator.clone(this);
        } else throw new CloneNotSupportedException(this.getClass().getName());
    }
    public final void _notify() {
        // TODO
    }
    public final void _notifyAll() {
        // TODO
    }
    public final void _wait(long timeout) throws java.lang.InterruptedException {
        if (timeout < 0L)
            throw new IllegalArgumentException(timeout+" < 0");
        // TODO
        int count = Monitor.getLockEntryCount(this);
        int k = count;
        for (;;) {
            Monitor.monitorexit(this);
            if (--k == 0) break;
        }
        jq_Thread t = Unsafe.getThreadBlock();
        t.sleep(timeout);
        for (;;) {
            Monitor.monitorenter(this);
            if (--count == 0) break;
        }
    }
    
}

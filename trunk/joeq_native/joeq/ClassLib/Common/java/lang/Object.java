/*
 * Object.java
 *
 * Created on January 29, 2001, 11:07 AM
 *
 */

package ClassLib.Common.java.lang;

import Allocator.HeapAllocator;
import Run_Time.Monitor;
import Run_Time.Reflection;
import Run_Time.Unsafe;
import Scheduler.jq_Thread;

/*
 * @author  John Whaley
 * @version 
 */
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
            throw new java.lang.IllegalArgumentException(timeout+" < 0");
        // TODO
        int count = Monitor.getLockEntryCount(this);
        int k = count;
        for (;;) {
            Monitor.monitorexit(this);
            if (--k == 0) break;
        }
        jq_Thread t = Unsafe.getThreadBlock();
        java.lang.InterruptedException rethrow;
        try {
            t.sleep(timeout);
            rethrow = null;
        } catch (java.lang.InterruptedException x) {
            rethrow = x;
        }
        for (;;) {
            Monitor.monitorenter(this);
            if (--count == 0) break;
        }
        if (rethrow != null)
            throw rethrow;
    }
    
}

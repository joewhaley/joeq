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
import Run_Time.Reflection;
import Run_Time.Unsafe;

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
        // TODO
    }
    
}

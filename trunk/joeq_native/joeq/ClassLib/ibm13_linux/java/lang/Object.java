/*
 * Object.java
 *
 * Created on January 29, 2001, 11:07 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang;

import Allocator.HeapAllocator;
import Clazz.jq_Class;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Reflection;
import Run_Time.Unsafe;

public abstract class Object {

    // native method implementations.
    private static void registerNatives(jq_Class clazz) {}
    public static final java.lang.Class getClass(java.lang.Object dis) {
        return Reflection.getJDKType(Unsafe.getTypeOf(dis));
    }
    public static int hashCode(java.lang.Object dis) { return java.lang.System.identityHashCode(dis); }
    protected static java.lang.Object clone(java.lang.Object dis) throws CloneNotSupportedException {
        if (dis instanceof Cloneable) {
            return HeapAllocator.clone(dis);
        } else throw new CloneNotSupportedException(dis.getClass().getName());
    }
    public static final void notify(java.lang.Object dis) {
        // TODO
    }
    public static final void notifyAll(java.lang.Object dis) {
        // TODO
    }
    public static final void wait(java.lang.Object dis, long timeout) throws java.lang.InterruptedException {
        // TODO
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Object;");
}

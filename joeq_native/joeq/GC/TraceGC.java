/**
 * TraceGC
 *
 * Created on Sep 23, 2002, 11:04:51 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package GC;

public abstract class TraceGC {

    public static /*final*/ boolean TRACE = false;

    public abstract boolean doWork() {}

    public abstract boolean mark() {}

    public abstract boolean sweep() {}

    public abstract boolean compact() {}
}

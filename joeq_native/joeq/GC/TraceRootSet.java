/**
 * TraceRootSet
 *
 * starting points for tracing algorithm used by Mark-and-Sweep garbage collectors,
 * which include thread stacks, statics, interned strings and JNI references.
 *
 * Created on Sep 23, 2002, 9:52:59 PM
 *
 * @author laudney
 * @version 0.1
 */
package GC;

import java.util.ArrayList;

import Scheduler.jq_RegisterState;

class TraceRootSet {

    static /*final*/ boolean TRACE = false;

    private ArrayList suspendedThreadStates = new ArrayList();

    boolean add(Object o) {
        if (!(o instanceof jq_RegisterState)) {
            throw new IllegalArgumentException();
        }
        return suspendedThreadStates.add(o);
    }
}

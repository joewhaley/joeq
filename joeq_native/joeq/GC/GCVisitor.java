/**
 * GCVisitor
 *
 * Created on Sep 25, 2002, 9:31:57 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package GC;

import Scheduler.jq_NativeThread;
import Scheduler.jq_RegisterState;

public interface GCVisitor {
    public void visit(jq_RegisterState state);
    public void farewell(jq_NativeThread[] nt);
}

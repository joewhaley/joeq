/*
 * jq_SynchThreadQueue.java
 *
 * Created on March 27, 2001, 11:21 PM
 *
 */

package Scheduler;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class jq_SynchThreadQueue extends jq_ThreadQueue {

    //public synchronized boolean isEmpty() { return super.isEmpty(); }
    public synchronized void enqueue(jq_Thread t) { super.enqueue(t); }
    public synchronized jq_Thread dequeue() { return super.dequeue(); }

}

// jq_SynchThreadQueue.java, created Mon Apr  9  1:52:50 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Scheduler;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class jq_SynchThreadQueue extends jq_ThreadQueue {

    //public synchronized boolean isEmpty() { return super.isEmpty(); }
    public synchronized void enqueue(jq_Thread t) { super.enqueue(t); }
    public synchronized void enqueueFront(jq_Thread t) { super.enqueueFront(t); }
    public synchronized jq_Thread dequeue() { return super.dequeue(); }
    public synchronized boolean remove(jq_Thread t2) { return super.remove(t2); }

}

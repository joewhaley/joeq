/*
 * jq_ThreadQueue.java
 *
 * Created on March 26, 2001, 10:27 AM
 *
 * @author  John Whaley
 * @version 
 */

package Scheduler;

import jq;
import java.util.Iterator;

public class jq_ThreadQueue {

    private jq_Thread head, tail;
    
    public boolean isEmpty() {
        return head == null;
    }
    
    public void enqueue(jq_Thread t) {
        jq.assert(t.next == null);
        if (head == null) head = t;
        else tail.next = t;
        tail = t;
    }
    
    public jq_Thread dequeue() {
        jq_Thread t = head;
        if (t == null) return null;
        head = t.next;
        t.next = null;
        if (head == null) tail = null;
        return t;
    }
    
    public Iterator threads() {
        final jq_Thread start = head;
        return new Iterator() {
            jq_Thread t = start;
            public boolean hasNext() {
                return t != null;
            }
            public Object next() {
                jq_Thread t2 = t;
                t = t.next;
                return t2;
            }
            public void remove() { throw new UnsupportedOperationException(); }
        };
    }
    
    public String toString() {
        StringBuffer s = new StringBuffer("{ ");
        Iterator i = threads();
        if (i.hasNext()) {
            s.append(i.next().toString());
            while (i.hasNext()) {
                s.append(", ");
                s.append(i.next().toString());
            }
        }
        s.append(" }");
        return s.toString();
    }
    
}

/*
 * jq_Thread.java
 *
 * Created on January 12, 2001, 1:07 AM
 *
 * @author  jwhaley
 * @version 
 */

package Scheduler;

public class jq_Thread {

    public static final Thread start_thread = new Thread("_start_");
    
    public Throwable exception_object;
    public final Thread thread_object;
    
    public jq_Thread(Thread t) {
        this.thread_object = t;
    }
    
    public Thread getJavaLangThreadObject() { return thread_object; }
    public String toString() { return ""+thread_object; }
    
    public void start() { }
    public void sleep(long millis) { }
    public void yield() { }
    public void setPriority(int newPriority) { }
    public void stop(Object o) { }
    public void suspend() { }
    public void resume() { }
    public void interrupt() { }
    public boolean isInterrupted(boolean clear) { return false; }
    public boolean isAlive() { return true; }
    public int countStackFrames() { return 0; }

}

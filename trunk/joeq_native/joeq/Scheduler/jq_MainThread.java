/*
 * jq_MainThread.java
 *
 * Created on April 1, 2001, 12:22 AM
 *
 * @author  John Whaley
 * @version 
 */

package Scheduler;

import Clazz.jq_StaticMethod;
import Run_Time.Reflection;

public class jq_MainThread extends java.lang.Thread {

    jq_StaticMethod m;
    Object arg;
    
    /** Creates new MainThread */
    public jq_MainThread(jq_StaticMethod m, Object arg) {
        this.m = m; this.arg = arg;
    }
    
    public void run() {
        try {
            Reflection.invokestatic_V(m, arg);
        } catch (Throwable t) {
            System.err.println("Exception occurred! "+t);
            t.printStackTrace(System.err);
            System.exit(-1);
        }
    }

}

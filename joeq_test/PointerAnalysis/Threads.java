// Threads.java, created Oct 29, 2003 8:07:46 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package PointerAnalysis;

/**
 * Threads
 * 
 * @author John Whaley
 * @version $Id$
 */
public class Threads {
    
    public static class ThreadA extends Thread {
        Threads a, b;
        ThreadA(Threads a) {
            this.a = a;
        }
        public void run() {
            b = new Threads();
            for (;;) {
                a.foo();
                b.foo();
            }
        }
    }
    
    public static void main(String[] args) {
        Threads a = new Threads();
        
        ThreadA t1 = new ThreadA(a);
        ThreadA t2 = new ThreadA(a);
        
        t1.start(); t2.start();
    }
    
    int counter;
    synchronized void foo() {
        synchronized(this) {
            ++counter;
            this.hashCode();
        }
    }
}

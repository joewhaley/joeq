// FinallyTest.java, created Oct 13, 2003 11:35:17 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package simple;

/**
 * FinallyTest
 * 
 * @author John Whaley
 * @version $Id$
 */
public class FinallyTest {

    public static void main(String[] args) {
    }

    public void run() {

    }
    
    public Object f() {
        return o;
    }
    
    public void set(Object q) {
        input2 = q;
    }
    
    Object o;
    
    Object input2;

    boolean b;
    
    class Foo {
        
        public void run() {
            Object input;
            
            synchronized (this) {
                input = input2;
            }
    
            try {
                if (f() != null)
                    set(input);
            } finally {
                synchronized (this) {
                    b = false;
                }
            }
        }
    }
}

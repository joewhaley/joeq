// Null.java, created Nov 1, 2003 8:42:36 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package PointerAnalysis;

/**
 * Null
 * 
 * @author John Whaley
 * @version $Id$
 */
public class Null {

    Object f;
    
    public static void main(String[] args) {
        Null n = null;
        n.f = new String("I am not Null");
        n.f = new Null();
        
        n.f.toString();
    }
    
    public String toString() {
        return "I am Null";
    }
    
    static void smethod(Object o, String s, Boolean b) { }
    static void kmethod() { smethod(null, "one", Boolean.FALSE); }

    public static void main2(String[] av) {
        smethod(null, "one", Boolean.TRUE);
        kmethod();
    }
}

// Simple.java, created Jul 29, 2004 2:47:22 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Generics;

/**
 * Simple
 * 
 * @author jwhaley
 * @version $Id$
 */
public class Simple /*<T>*/ {

    Object o;
    
    Object get() { return o; }
    void set(Object o) { this.o = o; }
    
    public static void main(String[] args) {
        Simple/*<Integer>*/ s = new Simple/*<Integer>*/();
        s.set(new Foo());
        Foo f = (Foo) s.get(); // cast can be eliminated.
    }
    
}

class Foo { }

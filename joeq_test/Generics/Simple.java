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
public class Simple /*<T3,T2 extends T3,T1 extends T2,T5,T4 extends T5>*/ {

    /*T2*/Object o;
    
    /*T3*/Object get() { return o; }
    void set(/*T1*/Object o) { this.o = o; }
    /*T5*/Object id(/*T4*/Object a) { return a; }
    
    public static void main(String[] args) {
        Simple/*<Foo,Foo,Foo,Foo,Foo>*/ s = new Simple/*<Foo,Foo,Foo,Foo,Foo>*/();
        s.set(new Foo());
        Foo f = (Foo) s.get(); // cast can be eliminated.
    }
    
}

class Foo { }

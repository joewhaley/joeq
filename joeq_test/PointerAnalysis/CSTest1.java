//CSTest1.java, created Jul 6, 2003 2:53:03 AM by John Whaley
//Copyright (C) 2003 John Whaley
//Licensed under the terms of the GNU LGPL; see COPYING for details.
package PointerAnalysis;

/**
 * @author John Whaley
 */
public class CSTest1 {

    CSTest1() {}

    public static void main(String[] args) {
        CSTest1 x = new CSTest1();
        CSTest1 y = new CSTest2();
        
        CSTest1 a = (CSTest1) id(x);
        CSTest1 b = (CSTest1) id(y);
        
        a.virtual(); // should be to CSTest1.virtual()
    }
    
    public static void main2(String[] args) {
        CSTest1 x = new CSTest1();
        CSTest1 y = new CSTest2();
        
        update(x, x);
        update(x, y);
        
        x.f.virtual(); // flow-insensitive says to CSTest1+CSTest2
    }
        
    static Object id(Object o) { return o; }
    
    CSTest1 f;
    
    static void update(CSTest1 a, CSTest1 b) {
        a.f = b;
    }
    
    public String toString() { return "CSTest1"; }
    
    public int virtual() { return 1; }
    
    public static void main3(String[] args) {
        CSTest1 x = new CSTest1();
        CSTest1 y = new CSTest2();
        
        update(x, x);
        update(x, y);
        update(y, y);
        
        CSTest1 a = (CSTest1) x.recursive(); // should be to CSTest1
        CSTest1 b = (CSTest1) y.recursive(); // should be to CSTest2
        
        a.virtual(); // flow-insensitive says to CSTest1+CSTest2
        b.virtual(); // should be to CSTest2
    }

    public CSTest1 recursive() {
        if (f == null) return this;
        return f.recursive();
    }
    
}

class CSTest2 extends CSTest1 {
    
    CSTest2() {}
    
    public int virtual() { return 2; }
    
    public CSTest1 recursive() {
        if (f == null) return this;
        return f.recursive(); // should be to CSTest2
    }
}

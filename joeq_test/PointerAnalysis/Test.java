package PointerAnalysis;

public class Test {

    Object o1, o2, o3;

    public Test() { }
    
    public Test(Object o) {
        this.o1 = o;
    }

    public Object get1() { return o1; }

    public void virtual() { }

    public static void test0() {
        
    }
    
    public static void test1() {
        Object o = new Integer(1);
        Object p = new Float(2f);
        Test a = new Test(o);
        Test2 b = new Test2(p);
        
        //a.o1.toString();
        a.o1.hashCode();
        
        b.o1.hashCode();
        
    }

}

class Test2 extends Test {
    
    public Test2() { }
    
    public Test2(Object o) { super(o); }
    
    public void virtual() { }
}

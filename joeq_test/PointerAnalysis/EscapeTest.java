package PointerAnalysis;

public class EscapeTest {

    Object f;

    public static void main(String[] args) {

        Object o = new Object();

        Object o2 = new Object();

        EscapeTest a = new EscapeTest();
        
        EscapeTest b = a.foo(o2);

        EscapeTest c = a.bar(b);

    }

    public EscapeTest foo(Object a) {

        EscapeTest o = new EscapeTest();

        o.f = a;

        return o;

    }

    public EscapeTest bar(Object a) {

        EscapeTest o = new EscapeTest();

        foo(a);

        return o;

    }

    public static void main2(String[] args) {

        Object o = new Object();
        Object o2 = new Object();

        EscapeTest a = new EscapeTest();
        
        EscapeTest b = a.foo(o2);

    }


}

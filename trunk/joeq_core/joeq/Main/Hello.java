/*
 * Hello.java
 *
 * Created on January 15, 2001, 6:20 PM
 *
 * @author  jwhaley
 * @version 
 */

package Main;

import java.io.*;
import Run_Time.*;

public abstract class Hello {

    static Throwable t1 = new Throwable();
    static ArrayIndexOutOfBoundsException t2 = new ArrayIndexOutOfBoundsException();
    /**
    * @param args the command line arguments
    */
    public static void main (String args[]) throws Exception {

        Object[] a = new Object[10];
        a[10] = new Object();
        
        foo1(new Exception());
        foo1(t1);
        foo1(t2);
        /*
        File f = new File("\\jdk1.3\\jre\\classes");
        String[] s = f.list();
        for (int i=0; i<s.length; ++i) {
            System.out.println(s[i]);
        }
         */
    }

    public static void foo1(Throwable x) {
        Throwable y = foo2(x);
        y.printStackTrace();
        System.out.println(x == y);
    }
    
    public static Throwable foo2(Throwable t) {
        t.printStackTrace();
        return t.fillInStackTrace();
    }
    
}

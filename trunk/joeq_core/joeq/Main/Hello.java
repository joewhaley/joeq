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

    /**
    * @param args the command line arguments
    */
    public static void main (String args[]) throws Exception {
        
        Class c = Hello.class;
        
        File f = new File("\\jdk1.3\\jre\\classes");
        String[] s = f.list();
        for (int i=0; i<s.length; ++i) {
            System.out.println(s[i]);
        }
    }

}

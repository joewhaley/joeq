/*
 * Hello.java
 *
 * Created on January 15, 2001, 6:20 PM
 *
 */

package Main;

import java.util.Iterator;

import Bootstrap.PrimordialClassLoader;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Hello {

    /**
    * @param args the command line arguments
    */
    public static void main (String args[]) throws Exception {
        jq.initializeForHostJVMExecution();
        
        for (int i=0; i<args.length; ++i) {
            printAllClassesInPackage(args[i]);
        }
    }

    public static void printAllClassesInPackage(String packageName) {
        Iterator i = PrimordialClassLoader.loader.listPackage(packageName);
        while (i.hasNext()) {
            String s = (String)i.next();
            jq.Assert(s.endsWith(".class"));
            s = s.substring(0, s.length()-6);
            System.out.println("L"+s+";");
        }
    }
    
}

/*
 * Hello.java
 *
 * Created on January 15, 2001, 6:20 PM
 *
 */

package Main;

import java.util.Iterator;

import Bootstrap.PrimordialClassLoader;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Hello {

    /**
    * @param args the command line arguments
    */
    public static void main (String args[]) throws Exception {
        HostedVM.initialize();
        
        for (int i=0; i<args.length; ++i) {
            printAllClassesInPackage(args[i]);
        }
    }

    public static void printAllClassesInPackage(String packageName) {
        Iterator i = PrimordialClassLoader.loader.listPackage(packageName);
        while (i.hasNext()) {
            String s = (String)i.next();
            Assert._assert(s.endsWith(".class"));
            s = s.substring(0, s.length()-6);
            System.out.println("L"+s+";");
        }
    }
    
}

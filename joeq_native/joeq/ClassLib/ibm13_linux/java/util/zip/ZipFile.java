/*
 * ZipFile.java
 *
 * Created on July 3, 2002, 3:15 PM
 */

package ClassLib.ibm13_linux.java.util.zip;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ZipFile {

    public static java.util.Vector inflaters;
    
    public static void init_inflaters() {
        inflaters = new java.util.Vector();
    }
    
}

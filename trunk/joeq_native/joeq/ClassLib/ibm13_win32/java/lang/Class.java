/*
 * Class.java
 *
 * Created on July 3, 2002, 1:16 PM
 */

package ClassLib.ibm13_win32.java.lang;
import Bootstrap.PrimordialClassLoader;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class Class {

    private static native Class forName0(java.lang.String name, boolean initialize,
                                         java.lang.ClassLoader loader)
        throws ClassNotFoundException;
    private native java.lang.Object newInstance0()
        throws InstantiationException, IllegalAccessException;

    private static Class forName1(java.lang.String name)
        throws ClassNotFoundException
    {
        // TODO: is this the correct classloader to use?
        java.lang.ClassLoader loader = PrimordialClassLoader.loader;
        return forName0(name, true, loader);
    }
    
    private java.lang.Object newInstance2(java.lang.Class ccls)
        throws InstantiationException, IllegalAccessException
    {
        // TODO: what do we do with the extra ccls argument?
        return newInstance0();
    }
    
}

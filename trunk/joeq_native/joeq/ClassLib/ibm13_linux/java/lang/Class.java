// Class.java, created Fri Jan 11 17:08:41 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.ClassLib.ibm13_linux.java.lang;

import joeq.Clazz.PrimordialClassLoader;

/**
 * Class
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
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

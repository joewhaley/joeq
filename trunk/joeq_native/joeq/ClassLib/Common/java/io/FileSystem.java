// FileSystem.java, created Thu Jul  4  4:50:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.ClassLib.Common.java.io;

import java.lang.reflect.Method;

import joeq.Clazz.PrimordialClassLoader;
import joeq.Clazz.jq_Class;
import joeq.Main.jq;
import joeq.Run_Time.Reflection;
import joeq.Util.Assert;

/**
 * FileSystem
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
abstract class FileSystem {

    public static FileSystem getFileSystem() { return (FileSystem) DEFAULT_FS; }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileSystem;");
    //public static final jq_StaticMethod _getFileSystem = _class.getOrCreateStaticMethod("getFileSystem", "()Ljava/io/FileSystem;");
    
    static final Object DEFAULT_FS;
    static {
        if (!jq.RunningNative) {
            Object o;
            try {
                Class klass = Reflection.getJDKType(_class);
                Method m = klass.getMethod("getFileSystem", null);
                m.setAccessible(true);
                o = m.invoke(null, null);
            } catch (Error x) {
                throw x;
            } catch (Throwable x) {
                Assert.UNREACHABLE();
                o = null;
            }
            DEFAULT_FS = o;
        } else {
            Object o = null;
            Assert.UNREACHABLE();
            DEFAULT_FS = o;
        }
    }
}

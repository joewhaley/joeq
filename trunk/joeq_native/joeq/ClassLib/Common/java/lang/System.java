/*
 * System.java
 *
 * Created on January 29, 2001, 10:26 AM
 *
 */

package ClassLib.Common.java.lang;

import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Run_Time.ArrayCopy;
import Run_Time.HashCode;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Run_Time.StackWalker;
import Bootstrap.PrimordialClassLoader;
import Main.jq;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Properties;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class System {
    
    private static void registerNatives() { }
    private static void setIn0(InputStream in) {
        Reflection.putstatic_A(_in, in);
    }
    private static void setOut0(PrintStream out) {
        Reflection.putstatic_A(_out, out);
    }
    private static void setErr0(PrintStream err) {
        Reflection.putstatic_A(_err, err);
    }
    public static long currentTimeMillis() {
        return SystemInterface.currentTimeMillis();
    }
    public static void arraycopy(java.lang.Object src, int src_position,
                                 java.lang.Object dst, int dst_position,
                                 int length) {
        ArrayCopy.arraycopy(src, src_position, dst, dst_position, length);
    }
    public static int identityHashCode(java.lang.Object x) {
        return HashCode.identityHashCode(x);
    }
    public static native void initializeSystemClass();
    static java.lang.Class getCallerClass() {
        StackWalker sw = new StackWalker(0, Unsafe.EBP());
        sw.gotoNext(); sw.gotoNext(); sw.gotoNext();
        jq_CompiledCode cc = sw.getCode();
        if (cc == null) return null;
        return Reflection.getJDKType(cc.getMethod().getDeclaringClass());
    }
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/System;");
    public static final jq_StaticField _in = _class.getOrCreateStaticField("in", "Ljava/io/InputStream;");
    public static final jq_StaticField _out = _class.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
    public static final jq_StaticField _err = _class.getOrCreateStaticField("err", "Ljava/io/PrintStream;");
    //public static final jq_StaticField _props = _class.getOrCreateStaticField("props", "Ljava/util/Properties;");
    //public static final jq_StaticMethod _initializeSystemClass = _class.getOrCreateStaticMethod("initializeSystemClass", "()V");

}

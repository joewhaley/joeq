/*
 * System.java
 *
 * Created on January 29, 2001, 10:26 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.lang;

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
import jq;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Properties;

public abstract class System {
    
    private static void registerNatives(jq_Class clazz) { }
    private static void setIn0(jq_Class clazz, InputStream in) {
        Reflection.putstatic_A(_in, in);
    }
    private static void setOut0(jq_Class clazz, PrintStream out) {
        Reflection.putstatic_A(_out, out);
    }
    private static void setErr0(jq_Class clazz, PrintStream err) {
        Reflection.putstatic_A(_err, err);
    }
    public static long currentTimeMillis(jq_Class clazz) {
        return SystemInterface.currentTimeMillis();
    }
    public static void arraycopy(jq_Class clazz,
                                 java.lang.Object src, int src_position,
                                 java.lang.Object dst, int dst_position,
                                 int length) {
        ArrayCopy.arraycopy(src, src_position, dst, dst_position, length);
    }
    public static int identityHashCode(jq_Class clazz, java.lang.Object x) {
        return HashCode.identityHashCode(x);
    }
    private static Properties initProperties(jq_Class clazz, Properties props) {
        props.setProperty("java.version", "1.3.0");
        props.setProperty("java.vendor", "joeq");
        props.setProperty("java.vendor.url", "http://www.joewhaley.com");
        props.setProperty("java.class.version", "47.0");
        
        // TODO: read these properties from environment.
        props.setProperty("java.home", "G:\\jdk1.3\\jre");
        props.setProperty("os.name", "Windows 2000");
        props.setProperty("os.arch", "x86");
        props.setProperty("os.version", "5.0");
        props.setProperty("file.separator", "\\");
        props.setProperty("path.separator", ";");
        props.setProperty("line.separator", "\r\n");
        props.setProperty("user.name", "jwhaley");
        props.setProperty("user.home", "G:\\Documents and Settings\\John Whaley");
        props.setProperty("user.dir", "G:\\joeq");
        props.setProperty("java.class.path", PrimordialClassLoader.loader.classpathToString());
        return props;
    }
    static java.lang.Class getCallerClass(jq_Class clazz) {
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
    public static final jq_StaticField _props = _class.getOrCreateStaticField("props", "Ljava/util/Properties;");
    public static final jq_StaticMethod _initializeSystemClass = _class.getOrCreateStaticMethod("initializeSystemClass", "()V");

}

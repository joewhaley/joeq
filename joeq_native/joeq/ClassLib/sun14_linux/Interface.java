/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.sun14_linux;

import java.util.Iterator;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import ClassLib.Common.Interface.CommonObjectTraverser;
import Clazz.jq_Class;
import Scheduler.jq_NativeThread;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.Interface {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/sun14_linux/"+desc.toString().substring(1));
            return new Util.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }

    public ObjectTraverser getObjectTraverser() {
    	return sun14_linuxObjectTraverser.INSTANCE;
    }
    
    public static class sun14_linuxObjectTraverser extends CommonObjectTraverser {
    	public static sun14_linuxObjectTraverser INSTANCE = new sun14_linuxObjectTraverser();
    	protected sun14_linuxObjectTraverser() {}
        public void initialize() {
            super.initialize();
            
            jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Unsafe;");
            nullStaticFields.add(k.getOrCreateStaticField("theUnsafe", "Lsun/misc/Unsafe;"));
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/reflect/UnsafeFieldAccessorImpl;");
            nullStaticFields.add(k.getOrCreateStaticField("unsafe", "Lsun/misc/Unsafe;"));
            
            k = PrimordialClassLoader.getJavaLangClass();
            nullInstanceFields.add(k.getOrCreateInstanceField("declaredFields", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("publicFields", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("declaredMethods", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("publicMethods", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("declaredConstructors", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("publicConstructors", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("declaredPublicFields", "Ljava/lang/ref/SoftReference;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("declaredPublicMethods", "Ljava/lang/ref/SoftReference;"));
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Field;");
            nullInstanceFields.add(k.getOrCreateInstanceField("fieldAccessor", "Lsun/reflect/FieldAccessor;"));
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Method;");
            nullInstanceFields.add(k.getOrCreateInstanceField("methodAccessor", "Lsun/reflect/MethodAccessor;"));
            k = (Clazz.jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Constructor;");
            nullInstanceFields.add(k.getOrCreateInstanceField("constructorAccessor", "Lsun/reflect/ConstructorAccessor;"));
            
            // for some reason, thread local gets created during bootstrapping. (SoftReference)
            // for now, just kill all thread locals.
            k = (Clazz.jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Thread;");
            nullInstanceFields.add(k.getOrCreateInstanceField("threadLocals", "Ljava/lang/ThreadLocal$ThreadLocalMap;"));
            
            jq_NativeThread.USE_INTERRUPTER_THREAD = false;
            
            // access the ISO-8859-1 character encoding, as it is used during bootstrapping
            PrimordialClassLoader.loader.getOrCreateBSType("Lsun/nio/cs/ISO_8859_1;");
        }
    }
    
}

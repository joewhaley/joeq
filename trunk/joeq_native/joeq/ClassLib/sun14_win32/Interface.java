/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.sun14_win32;

import ClassLib.ClassLibInterface;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.Interface {

    /** Creates new Interface */
    public Interface() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/sun14_win32/"+desc.toString().substring(1));
            return new Util.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public void initialize() {
        super.initialize();
        
        Clazz.jq_Class jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Unsafe;");
        nullStaticFields.add(jq_class.getOrCreateStaticField("theUnsafe", "Lsun/misc/Unsafe;"));
        jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Lsun/reflect/UnsafeFieldAccessorImpl;");
        nullStaticFields.add(jq_class.getOrCreateStaticField("unsafe", "Lsun/misc/Unsafe;"));
        
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("declaredFields", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("publicFields", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("declaredMethods", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("publicMethods", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("declaredConstructors", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("publicConstructors", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("declaredPublicFields", "Ljava/lang/ref/SoftReference;"));
        nullInstanceFields.add(Bootstrap.PrimordialClassLoader.loader.getJavaLangClass().getOrCreateInstanceField("declaredPublicMethods", "Ljava/lang/ref/SoftReference;"));
        jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Field;");
        nullInstanceFields.add(jq_class.getOrCreateInstanceField("fieldAccessor", "Lsun/reflect/FieldAccessor;"));
        jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Method;");
        nullInstanceFields.add(jq_class.getOrCreateInstanceField("methodAccessor", "Lsun/reflect/MethodAccessor;"));
        jq_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Constructor;");
        nullInstanceFields.add(jq_class.getOrCreateInstanceField("constructorAccessor", "Lsun/reflect/ConstructorAccessor;"));
        
        Scheduler.jq_NativeThread.USE_INTERRUPTER_THREAD = true;
    }
    
}

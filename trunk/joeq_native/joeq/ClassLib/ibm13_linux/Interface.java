/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.ibm13_linux;

import ClassLib.ClassLibInterface;
import Main.jq;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.Interface {

    /** Creates new Interface */
    public Interface() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && (desc.toString().startsWith("Ljava/") ||
                                                    desc.toString().startsWith("Lcom/ibm/jvm/"))) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/ibm13_linux/"+desc.toString().substring(1));
            return new Util.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public void initialize() {
    	super.initialize();
    	
        Clazz.jq_Class launcher_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Launcher;");
        nullStaticFields.add(launcher_class.getOrCreateStaticField("launcher", "Lsun/misc/Launcher;"));
        //jq_Class urlclassloader_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader;");
        //nullStaticFields.add(urlclassloader_class.getOrCreateStaticField("extLoader", "Ljava/net/URLClassLoader;"));
        Clazz.jq_Class zipfile_class = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
        nullStaticFields.add(zipfile_class.getOrCreateStaticField("inflaters", "Ljava/util/Vector;"));
        Clazz.jq_Class string_class = Bootstrap.PrimordialClassLoader.getJavaLangString();
        nullStaticFields.add(string_class.getOrCreateStaticField("btcConverter", "Ljava/lang/ThreadLocal;"));
        nullStaticFields.add(string_class.getOrCreateStaticField("ctbConverter", "Ljava/lang/ThreadLocal;"));
    	
        Clazz.jq_Class k = (Clazz.jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader$ClassFinder;");
        nullInstanceFields.add(k.getOrCreateInstanceField("name", "Ljava/lang/String;"));
        
        // access the ISO-8859-1 character encoding, as it is used during bootstrapping
        try {
            String s = new String(new byte[0], 0, 0, "ISO-8859-1");
        } catch (java.io.UnsupportedEncodingException x) {}
        Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Lsun/io/CharToByteISO8859_1;");

        // we need to reinitialize the inflaters array on startup.
        Object[] args = { } ;
        Clazz.jq_Method init_inflaters = zipfile_class.getOrCreateStaticMethod("init_inflaters", "()V");
        Bootstrap.MethodInvocation mi = new Bootstrap.MethodInvocation(init_inflaters, args);
        jq.on_vm_startup.add(mi);
        System.out.println("Added call to reinitialize java.util.zip.ZipFile.inflaters field on joeq startup: "+mi);
    }
    
    public java.lang.Object mapInstanceField(java.lang.Object o, Clazz.jq_InstanceField f) {
        Clazz.jq_Class c = f.getDeclaringClass();
        if (c == Bootstrap.PrimordialClassLoader.getJavaLangThread()) {
	        String fieldName = f.getName().toString();
            if (fieldName.equals("threadLocals")) {
                return java.util.Collections.EMPTY_MAP;
            }
            if (fieldName.equals("inheritableThreadLocals")) {
                return java.util.Collections.EMPTY_MAP;
            }
        }
        return super.mapInstanceField(o, f);
    }
}

// Interface.java, created Fri Jan 11 17:07:15 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.ibm13_linux;

import java.util.Iterator;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Main.jq;

/**
 * Interface
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.InterfaceImpl {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && (desc.toString().startsWith("Ljava/") ||
                                                    desc.toString().startsWith("Lcom/ibm/jvm/"))) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/ibm13_linux/"+desc.toString().substring(1));
            return new Util.Collections.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public ObjectTraverser getObjectTraverser() {
        return ibm13_linuxObjectTraverser.INSTANCE;
    }
    
    public static class ibm13_linuxObjectTraverser extends CommonObjectTraverser {
        public static ibm13_linuxObjectTraverser INSTANCE = new ibm13_linuxObjectTraverser();
        protected ibm13_linuxObjectTraverser() {}
        public void initialize() {
            super.initialize();
            
            // access the ISO-8859-1 character encoding, as it is used during bootstrapping
            try {
                new String(new byte[0], 0, 0, "ISO-8859-1");
            } catch (java.io.UnsupportedEncodingException x) {}
            PrimordialClassLoader.loader.getOrCreateBSType("Lsun/io/CharToByteISO8859_1;");
    
            jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader$ClassFinder;");
            nullInstanceFields.add(k.getOrCreateInstanceField("name", "Ljava/lang/String;"));
            
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Launcher;");
            nullStaticFields.add(k.getOrCreateStaticField("launcher", "Lsun/misc/Launcher;"));
            //k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/URLClassLoader;");
            //nullStaticFields.add(k.getOrCreateStaticField("extLoader", "Ljava/net/URLClassLoader;"));
            k = PrimordialClassLoader.getJavaLangString();
            nullStaticFields.add(k.getOrCreateStaticField("btcConverter", "Ljava/lang/ThreadLocal;"));
            nullStaticFields.add(k.getOrCreateStaticField("ctbConverter", "Ljava/lang/ThreadLocal;"));
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
            nullStaticFields.add(k.getOrCreateStaticField("inflaters", "Ljava/util/Vector;"));
            
            // we need to reinitialize the inflaters array on startup.
            if (jq.on_vm_startup != null) {
                Object[] args = { } ;
                Clazz.jq_Method init_inflaters = k.getOrCreateStaticMethod("init_inflaters", "()V");
                Bootstrap.MethodInvocation mi = new Bootstrap.MethodInvocation(init_inflaters, args);
                jq.on_vm_startup.add(mi);
                System.out.println("Added call to reinitialize java.util.zip.ZipFile.inflaters field on joeq startup: "+mi);
            }
        }
        
        public java.lang.Object mapInstanceField(java.lang.Object o, Clazz.jq_InstanceField f) {
            if (IGNORE_THREAD_LOCALS) {
                jq_Class c = f.getDeclaringClass();
                if (c == PrimordialClassLoader.getJavaLangThread()) {
                    String fieldName = f.getName().toString();
                    if (fieldName.equals("threadLocals"))
                        return java.util.Collections.EMPTY_MAP;
                    if (fieldName.equals("inheritableThreadLocals"))
                        return java.util.Collections.EMPTY_MAP;
                }
            }
            return super.mapInstanceField(o, f);
        }
    }
}

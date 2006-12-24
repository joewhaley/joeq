// Interface.java, created Fri Apr  5 18:36:41 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.ClassLib.sun15_linux;

import java.util.Collections;
import java.util.Iterator;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_InstanceField;
import joeq.Class.jq_StaticField;
import joeq.ClassLib.ClassLibInterface;
import joeq.Main.jq;
import joeq.Runtime.ObjectTraverser;
import jwutil.collections.AppendIterator;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Interface extends joeq.ClassLib.sun142_linux.Interface {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(joeq.UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB &&
                (desc.toString().startsWith("Ljava/") ||
                 desc.toString().startsWith("Lsun/misc/"))) {
            joeq.UTF.Utf8 u = joeq.UTF.Utf8.get("Ljoeq/ClassLib/sun15_linux/"+desc.toString().substring(1));
            return new AppendIterator(super.getImplementationClassDescs(desc),
                                      Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public ObjectTraverser getObjectTraverser() {
        return sun15_linuxObjectTraverser.INSTANCE;
    }
    
    public static class sun15_linuxObjectTraverser extends sun142_linuxObjectTraverser {
        public static sun15_linuxObjectTraverser INSTANCE = new sun15_linuxObjectTraverser();
        protected sun15_linuxObjectTraverser() {}
        public void initialize() {
            super.initialize();
            
            jq_Class k;

            // jdk1.5 caches name string.
            k = (jq_Class) PrimordialClassLoader.getJavaLangClass();
            nullInstanceFields.add(k.getOrCreateInstanceField("name", "Ljava/lang/String;"));
            
            // generated during reflective access.
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/reflect/Field;");
            nullInstanceFields.add(k.getOrCreateInstanceField("fieldAccessor", "Lsun/reflect/FieldAccessor;"));
            nullInstanceFields.add(k.getOrCreateInstanceField("overrideFieldAccessor", "Lsun/reflect/FieldAccessor;"));

            // crashes on reflective access.
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Lsun/reflect/UnsafeStaticFieldAccessorImpl;");
            nullInstanceFields.add(k.getOrCreateInstanceField("base", "Ljava/lang/Object;"));

            // zip files use mappedBuffer
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
            nullInstanceFields.add(k.getOrCreateInstanceField("mappedBuffer", "Ljava/nio/MappedByteBuffer;"));

            // leads to sun.security.rsa junk
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/security/jca/Providers;");
            nullStaticFields.add(k.getOrCreateStaticField("threadLists", "Ljava/lang/ThreadLocal;"));
            nullStaticFields.add(k.getOrCreateStaticField("providerList", "Lsun/security/jca/ProviderList;"));

            // leads to sun.security.provider.certpath junk
            k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lsun/net/www/protocol/jar/JarFileFactory;");
            nullStaticFields.add(k.getOrCreateStaticField("fileCache", "Ljava/util/HashMap;"));
            nullStaticFields.add(k.getOrCreateStaticField("urlCache", "Ljava/util/HashMap;"));
            
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Shutdown;");
            nullStaticFields.add(k.getOrCreateStaticField("hooks", "Ljava/util/HashSet;"));

            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Lsun/misc/Cleaner;");
            nullStaticFields.add(k.getOrCreateStaticField("first", "Lsun/misc/Cleaner;"));
            nullStaticFields.add(k.getOrCreateStaticField("dummyQueue", "Ljava/lang/ref/ReferenceQueue;"));

            // Leads to an open file handle.
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Lsun/security/provider/NativePRNG;");
            nullStaticFields.add(k.getOrCreateStaticField("INSTANCE", "Lsun/security/provider/NativePRNG$RandomIO;"));

            // Leads to an open file handle.
            // TODO: needs reinit.
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Lsun/security/provider/SeedGenerator;");
            nullStaticFields.add(k.getOrCreateStaticField("instance", "Lsun/security/provider/SeedGenerator;"));
        }
        
        public static final jq_StaticField valueOffsetField;
        
        static {
            jq_Class k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/concurrent/atomic/AtomicLong;");
            valueOffsetField = k.getOrCreateStaticField("valueOffset", "J");
        }
        
        public java.lang.Object mapStaticField(jq_StaticField f) {
            // valueOffset is the offset of the "value" field in AtomicLong
            if (f == valueOffsetField) {
                jq_Class k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/concurrent/atomic/AtomicLong;");
                k.prepare();
                int offset = ((jq_InstanceField) k.getDeclaredMember("value", "J")).getOffset();
                return new Long(offset);
            }
            return super.mapStaticField(f);
        }
    }

}

// Interface.java, created Thu Sep  5 10:48:53 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.apple13_osx;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Run_Time.SystemInterface;
import UTF.Utf8;

/**
 * Interface
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.InterfaceImpl {

    /** Creates new Interface */
    public Interface() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/apple13_osx/"+desc.toString().substring(1));
            return new Util.Collections.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }

    public ObjectTraverser getObjectTraverser() {
        return apple13_osxObjectTraverser.INSTANCE;
    }
    
    public static class apple13_osxObjectTraverser extends CommonObjectTraverser {
        public static apple13_osxObjectTraverser INSTANCE = new apple13_osxObjectTraverser();
        protected apple13_osxObjectTraverser() {}
        public java.lang.Object mapInstanceField(java.lang.Object o, Clazz.jq_InstanceField f) {
            if (IGNORE_THREAD_LOCALS) {
                jq_Class c = f.getDeclaringClass();
                if (c == PrimordialClassLoader.getJavaLangThread()) {
                    String fieldName = f.getName().toString();
                    if (fieldName.equals("threadLocals") || fieldName.equals("inheritableThreadLocals")) {
                        // 1.3.1_05 and greater uses the 1.4 implementation of thread locals.
                        // see Sun BugParade id#4667411.
                        // we check the type of the field and return the appropriate object.
                        if (f.getType().getDesc() == Utf8.get("Ljava/util/Map;")) {
                            return java.util.Collections.EMPTY_MAP;
                        } else if (f.getType().getDesc() == Utf8.get("Ljava/lang/ThreadLocal$ThreadLocalMap;")) {
                            return null;
                        } else {
                            SystemInterface.debugwriteln("Unknown type for field "+f);
                            return null;
                        }
                    }
                }
            }
            return super.mapInstanceField(o, f);
        }
    }
    
}

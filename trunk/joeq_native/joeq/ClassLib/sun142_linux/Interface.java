// Interface.java, created Fri Apr  5 18:36:41 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.sun142_linux;

import java.util.Iterator;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public final class Interface extends ClassLib.sun14_linux.Interface {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/sun142_linux/"+desc.toString().substring(1));
            return new Util.Collections.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public ObjectTraverser getObjectTraverser() {
        return sun142_linuxObjectTraverser.INSTANCE;
    }
    
    public static class sun142_linuxObjectTraverser extends sun14_linuxObjectTraverser {
        public static sun142_linuxObjectTraverser INSTANCE = new sun142_linuxObjectTraverser();
        protected sun142_linuxObjectTraverser() {}
        public void initialize() {
            super.initialize();
            
            jq_Class k;
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LClassLib/Common/java/util/zip/DeflaterHuffman;");
            k.load();
            
            // used during bootstrapping.
            k = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/ObjectInputStream$GetFieldImpl;");
            k.load();
            
        }
    }
}

// Interface.java, created Wed Feb  4 12:10:06 PST 2004
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.pa;

import java.util.Iterator;

import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Bootstrap.ObjectTraverser;
import Util.Assert;

/*
 * Classes we replace for pointer analysis purposes ('pa')
 * If we can model the effect of a native method from the java.* hierarchy
 * in Java code, we can add an implementation whose bytecode is then
 * analyzed.
 *
 * @author  Godmar Back <gback@stanford.edu>
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.InterfaceImpl {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/pa/"+desc.toString().substring(1));
            return java.util.Collections.singleton(u).iterator();
        }
        return java.util.Collections.EMPTY_SET.iterator();
    }

    public ObjectTraverser getObjectTraverser() {
        return new ObjectTraverser() {
            public void initialize() { }
            public Object mapStaticField(jq_StaticField f) { Assert.UNREACHABLE(); return null; }
            public Object mapInstanceField(Object o, jq_InstanceField f) { Assert.UNREACHABLE(); return null; }
            public Object mapValue(Object o) { Assert.UNREACHABLE(); return null; }
        };
    }
}

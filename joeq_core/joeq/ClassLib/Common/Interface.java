// Interface.java, created Wed Sep 11 15:00:54 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common;

import Bootstrap.ObjectTraverser;

/**
 * Interface
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public interface Interface {

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc);
    
    public ObjectTraverser getObjectTraverser();
    
    public java.lang.Class createNewClass(Clazz.jq_Type f);
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f);
    
    public void initNewConstructor(java.lang.reflect.Constructor dis, Clazz.jq_Initializer f);
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f);
    
    public void initNewField(java.lang.reflect.Field dis, Clazz.jq_Field f);
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f);
    
    public void initNewMethod(java.lang.reflect.Method dis, Clazz.jq_Method f);
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f);
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f);
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f);
    
    public Clazz.jq_Type getJQType(java.lang.Class k);
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc);
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t);
    
    public void init_zipfile(java.util.zip.ZipFile dis, java.lang.String name) throws java.io.IOException;
    
    public void init_inflater(java.util.zip.Inflater dis, boolean nowrap) throws java.io.IOException;

    public void initializeSystemClass() throws java.lang.Throwable;    
}

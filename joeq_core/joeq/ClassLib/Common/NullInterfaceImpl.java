/*
 * NullInterfaceImpl.java
 *
 * Created on December 6, 2002, 3:49 PM
 *
 */

package ClassLib.Common;

import Bootstrap.ObjectTraverser;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Member;
import Clazz.jq_Method;
import Clazz.jq_StaticField;
import Clazz.jq_Type;
import Clazz.jq_Reference.jq_NullType;
import Main.jq;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Scheduler.jq_Thread;

/*
 * @author  John Whaley
 * @author  Michael Martin
 * @version $Id$
 */
public class NullInterfaceImpl implements ClassLib.Common.Interface {

    /** Creates new Interface */
    public NullInterfaceImpl() {}

    public java.util.Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        return java.util.Collections.EMPTY_SET.iterator();
    }
    
    public ObjectTraverser getObjectTraverser() {
        return NullObjectTraverser.INSTANCE;
    }

    public static class NullObjectTraverser extends ObjectTraverser {
	public void initialize () { }
	public Object mapStaticField(jq_StaticField f) { return NO_OBJECT; }
	public Object mapInstanceField(Object o, jq_InstanceField f) { return NO_OBJECT; }
	public Object mapValue(Object o) { return NO_OBJECT; }
	public static final NullObjectTraverser INSTANCE = new NullObjectTraverser();
    }
    
    public java.lang.Class createNewClass(Clazz.jq_Type f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public java.lang.reflect.Constructor createNewConstructor(Clazz.jq_Initializer f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void initNewConstructor(java.lang.reflect.Constructor dis, Clazz.jq_Initializer f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public java.lang.reflect.Field createNewField(Clazz.jq_Field f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void initNewField(java.lang.reflect.Field dis, Clazz.jq_Field f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public java.lang.reflect.Method createNewMethod(Clazz.jq_Method f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void initNewMethod(java.lang.reflect.Method dis, Clazz.jq_Method f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public Clazz.jq_Field getJQField(java.lang.reflect.Field f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public Clazz.jq_Initializer getJQInitializer(java.lang.reflect.Constructor f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public Clazz.jq_Method getJQMethod(java.lang.reflect.Method f) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public Scheduler.jq_Thread getJQThread(java.lang.Thread t) {
	jq_Class k = Bootstrap.PrimordialClassLoader.getJavaLangThread();
	Clazz.jq_InstanceField f = k.getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
	return (Scheduler.jq_Thread)Reflection.getfield_A(t, f);
    }
    
    public Clazz.jq_Type getJQType(java.lang.Class k) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public Clazz.jq_Type getOrCreateType(java.lang.ClassLoader cl, UTF.Utf8 desc) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void unloadType(java.lang.ClassLoader cl, Clazz.jq_Type t) {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void init_zipfile(java.util.zip.ZipFile dis, java.lang.String name) throws java.io.IOException {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void init_inflater(java.util.zip.Inflater dis, boolean nowrap) throws java.io.IOException {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
    
    public void initializeSystemClass() throws java.lang.Throwable {
	throw new UnsupportedOperationException("Using a Null ClassLib Interface!");
    }
}

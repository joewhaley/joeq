/*
 * ObjectTraverser.java
 *
 * Created on January 14, 2001, 11:38 AM
 *
 */

package Bootstrap;

import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Member;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_Type;
import Main.jq;
import Run_Time.Reflection;
import Run_Time.TypeCheck;
import Scheduler.jq_Thread;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ObjectTraverser {

	public static class Empty extends ObjectTraverser {
		public Empty() { }
        public void initialize() { }
        public Object mapStaticField(jq_StaticField f) { return NO_OBJECT; }
        public Object mapInstanceField(Object o, jq_InstanceField f) { return NO_OBJECT; }
        public Object mapValue(Object o) { return o; }
    }

    public abstract void initialize();
    public abstract Object mapStaticField(jq_StaticField f);
    public abstract Object mapInstanceField(Object o, jq_InstanceField f);
    public abstract Object mapValue(Object o);
    
    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    public static final java.lang.Object NO_OBJECT = new java.lang.Object();
    
    public Object getStaticFieldValue(jq_StaticField f) {
        java.lang.Object result = this.mapStaticField(f);
        if (result != NO_OBJECT) return this.mapValue(result);
        // get the value via real reflection.
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        return getStaticFieldValue_reflection(c, fieldName);
    }
    public Object getStaticFieldValue_reflection(Class c, String fieldName) {
        if (TRACE) out.println("Getting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        if (f2 == null) {
        	jq_Class klass = (jq_Class)Reflection.getJQType(c);
            for (Iterator i=ClassLibInterface.DEFAULT.getImplementationClassDescs(klass.getDesc()); i.hasNext(); ) {
                UTF.Utf8 u = (UTF.Utf8)i.next();
                if (TRACE) out.println("Checking mirror class "+u);
                String s = u.toString();
                jq.Assert(s.charAt(0) == 'L');
                try {
                    c = Class.forName(s.substring(1, s.length()-1).replace('/', '.'));
                    f2 = Reflection.getJDKField(c, fieldName);
                    if (f2 != null) break;
                } catch (ClassNotFoundException x) {
                    if (TRACE) out.println("Mirror class "+s+" doesn't exist");
                }
            }
        }
        jq.Assert(f2 != null, "host jdk does not contain static field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) != 0);
        try {
            return this.mapValue(f2.get(null));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public Object getInstanceFieldValue(Object base, jq_InstanceField f) {
        java.lang.Object result = this.mapInstanceField(base, f);
        if (result != NO_OBJECT) return this.mapValue(result);
        // get the value via real reflection.
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        return getInstanceFieldValue_reflection(base, c, fieldName);
    }
    public Object getInstanceFieldValue_reflection(Object base, Class c, String fieldName) {
        if (TRACE) out.println("Getting value of instance field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.Assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            return this.mapValue(f2.get(base));
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public void putStaticFieldValue(jq_StaticField f, Object o) {
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        putStaticFieldValue(c, fieldName, o);
    }
    public void putStaticFieldValue(Class c, String fieldName, Object o) {
        if (TRACE) out.println("Setting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        if (f2 == null) {
        	jq_Class klass = (jq_Class)Reflection.getJQType(c);
            for (Iterator i=ClassLibInterface.DEFAULT.getImplementationClassDescs(klass.getDesc()); i.hasNext(); ) {
                UTF.Utf8 u = (UTF.Utf8)i.next();
                if (TRACE) out.println("Checking mirror class "+u);
                String s = u.toString();
                jq.Assert(s.charAt(0) == 'L');
                try {
                    c = Class.forName(s.substring(1, s.length()-1).replace('/', '.'));
                    f2 = Reflection.getJDKField(c, fieldName);
                    if (f2 != null) break;
                } catch (ClassNotFoundException x) {
                    if (TRACE) out.println("Mirror class "+s+" doesn't exist");
                }
            }
        }
        jq.Assert(f2 != null, "host jdk does not contain static field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) != 0);
        try {
            f2.set(null, o);
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
        }
    }
    
    public void putInstanceFieldValue(Object base, jq_InstanceField f, Object o) {
        Class c = Reflection.getJDKType(f.getDeclaringClass());
        String fieldName = f.getName().toString();
        putInstanceFieldValue(base, c, fieldName, o);
    }
    public void putInstanceFieldValue(Object base, Class c, String fieldName, Object o) {
        if (TRACE) out.println("Setting value of static field "+c+"."+fieldName+" via reflection");
        Field f2 = Reflection.getJDKField(c, fieldName);
        jq.Assert(f2 != null, "host jdk does not contain instance field "+c.getName()+"."+fieldName);
        f2.setAccessible(true);
        jq.Assert((f2.getModifiers() & Modifier.STATIC) == 0);
        try {
            f2.set(base, o);
        } catch (IllegalAccessException x) {
            jq.UNREACHABLE();
        }
    }
}

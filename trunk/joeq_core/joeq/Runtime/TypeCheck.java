/*
 * TypeCheck.java
 *
 * Created on January 2, 2001, 10:41 AM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Clazz.jq_Reference;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_Array;
import Clazz.jq_NameAndDesc;
import UTF.Utf8;
import Run_Time.Unsafe;
import jq;

public abstract class TypeCheck implements jq_ClassFileConstants {
    
    public static Object checkcast(Object k, jq_Type t) {
        if (k != null) {
            jq_Type t2 = Unsafe.getTypeOf(k);
            if (!TypeCheck.isAssignable(t2, t)) 
                throw new ClassCastException(t2+" cannot be cast into "+t);
        }
        return k;
    }
    public static boolean instance_of(Object k, jq_Type t) {
        if (k == null)
            return false;
        jq_Type t2 = Unsafe.getTypeOf(k);
        if (!TypeCheck.isAssignable(t2, t)) 
            return false;
        return true;
    }
    public static void arrayStoreCheck(Object value, Object[] arrayref) 
    throws ArrayStoreException {
        if (value == null) return;
        jq_Type t = Unsafe.getTypeOf(value);
        jq_Array a = (jq_Array)Unsafe.getTypeOf(arrayref);
        jq_Type t2 = a.getElementType();
        if (!isAssignable(t, t2))
            throw new ArrayStoreException(t+" into array "+a);
    }
    
    // From the algorithm in vm spec under "checkcast"
    // returns true if "T = S;" would be legal. (T is same or supertype of S)
    // S and T should already be prepared.
    public static boolean isAssignable(jq_Type S, jq_Type T) {
        if (S == T)
            return true;
        jq_Type s2 = S, t2 = T;
        while (t2.isArrayType()) {
            if (!s2.isArrayType()) {
                return false;
            }
            s2 = ((jq_Array)s2).getElementType(); s2.load(); s2.verify(); s2.prepare();
            t2 = ((jq_Array)t2).getElementType(); t2.load(); t2.verify(); t2.prepare();
        }
        if (s2.isPrimitiveType() || t2.isPrimitiveType()) {
            return false;
        }
        // t2 is a class
        boolean is_t2_loaded = t2.isLoaded();
        t2.load();
        if (s2.isArrayType()) {
            ((jq_Array)s2).chkState(STATE_PREPARED);
            if (((jq_Class)t2).isInterface()) {
                //s2.load(); s2.verify(); s2.prepare();
                return ((jq_Array)s2).implementsInterface((jq_Class)t2);
            }
            return t2 == PrimordialClassLoader.loader.getJavaLangObject();
        }
        // both are classes
        ((jq_Class)s2).chkState(STATE_PREPARED);
        if (((jq_Class)t2).isInterface()) {
            if (((jq_Class)s2).isInterface()) {
                if (!is_t2_loaded) return false;
                return isSuperclassOf((jq_Class)t2, (jq_Class)s2);
            }
            return ((jq_Class)s2).implementsInterface((jq_Class)t2);
        }
        // t2 is not an interface
        if (!is_t2_loaded) return false;
        return isSuperclassOf((jq_Class)t2, (jq_Class)s2);
    }
    
    public static boolean isSuperclassOf(jq_Class t1, jq_Class t2) {
        // doesn't do equality test.
        for (;;) {
            t2 = t2.getSuperclass();
            if (t2 == null) return false;
            if (t1 == t2) return true;
        }
    }
    
    public static final jq_StaticMethod _checkcast;
    public static final jq_StaticMethod _instance_of;
    public static final jq_StaticMethod _arrayStoreCheck;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/TypeCheck;");
        _checkcast = k.getOrCreateStaticMethod("checkcast", "(Ljava/lang/Object;LClazz/jq_Type;)Ljava/lang/Object;");
        _instance_of = k.getOrCreateStaticMethod("instance_of", "(Ljava/lang/Object;LClazz/jq_Type;)Z");
        _arrayStoreCheck = k.getOrCreateStaticMethod("arrayStoreCheck", "(Ljava/lang/Object;[Ljava/lang/Object;)V");
    }
}

/*
 * TypeCheck.java
 *
 * Created on January 2, 2001, 10:41 AM
 * 
 */

package Run_Time;

import java.util.Stack;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Main.jq;
import Memory.Address;
import Memory.HeapAddress;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class TypeCheck implements jq_ClassFileConstants {
    
    public static Object checkcast(Object k, jq_Type t) {
        if (k != null) {
            jq_Type t2 = jq_Reference.getTypeOf(k);
            if (!TypeCheck.isAssignable(t2, t)) 
                throw new ClassCastException(t2+" cannot be cast into "+t);
        }
        return k;
    }
    public static boolean instance_of(Object k, jq_Type t) {
        if (k == null)
            return false;
        jq_Type t2 = jq_Reference.getTypeOf(k);
        if (!TypeCheck.isAssignable(t2, t)) 
            return false;
        return true;
    }
    public static void arrayStoreCheck(HeapAddress value, Object[] arrayref) 
    throws ArrayStoreException {
        if (value.isNull()) return;
        jq_Array a = (jq_Array)jq_Reference.getTypeOf(arrayref);
        jq_Type t2 = a.getElementType();
        if (t2.isAddressType()) return;
        Object v = value.asObject();
        jq_Type t = jq_Reference.getTypeOf(v);
        if (!isAssignable(t, t2))
            throw new ArrayStoreException(t+" into array "+a);
    }
    
    // From the algorithm in vm spec under "checkcast"
    // returns true if "T = S;" would be legal. (T is same or supertype of S)
    // S should already be prepared.
    public static boolean isAssignable(jq_Type S, jq_Type T) {
        if (S == T)
            return true;
        jq_Type s2 = S, t2 = T;
        if (false) {
            if (t2 == Address._class)
                return s2.isAddressType();
            if (t2.isAddressType() || s2.isAddressType())
                return false;
        }
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
    
    public static final byte YES = 2;
    public static final byte MAYBE = 1;
    public static final byte NO = 0;
    // returns YES if "T = S;" would be legal. (T is same or supertype of S)
    public static byte isAssignable_noload(jq_Type S, jq_Type T) {
        if (S == jq_Reference.jq_NullType.NULL_TYPE) {
            if (T.isReferenceType()) return YES;
            else return NO;
        }
        if (T == jq_Reference.jq_NullType.NULL_TYPE) return NO;
        if (T == S) return YES;
        if (T.isIntLike() && S.isIntLike()) return YES;
        if (T == PrimordialClassLoader.loader.getJavaLangObject() && S.isReferenceType()) return YES;
        if (!T.isPrepared() || !S.isPrepared()) return MAYBE;
        if (T.isArrayType()) {
            jq_Type elemType = ((jq_Array)T).getInnermostElementType();
            if (!elemType.isPrepared()) return MAYBE;
        }
        if (S.isArrayType()) {
            jq_Type elemType = ((jq_Array)S).getInnermostElementType();
            if (!elemType.isPrepared()) return MAYBE;
        }
        if (TypeCheck.isAssignable(S, T)) return YES;
        else return NO;
    }
    
    // Returns true if t1 is a superclass of t2
    public static boolean isSuperclassOf(jq_Class t1, jq_Class t2) {
        // doesn't do equality test.
        for (;;) {
            t2.load();
            t2 = t2.getSuperclass();
            if (t2 == null) return false;
            if (t1 == t2) return true;
        }
    }
    
    public static jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2) {
        return findCommonSuperclass(t1, t2, false);
    }
    
    public static jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2, boolean load) {
        if (t1 == t2) return t1;
        if (t1.isPrimitiveType() && t2.isPrimitiveType()) {
            jq_Type result = null;
            if (t1.isIntLike() && t2.isIntLike()) {
                if (t1 == jq_Primitive.INT || t2 == jq_Primitive.INT) return jq_Primitive.INT;
                if (t1 == jq_Primitive.CHAR) {
                    if (t2 == jq_Primitive.SHORT) return jq_Primitive.INT;
                    return jq_Primitive.CHAR;
                }
                if (t2 == jq_Primitive.CHAR) return jq_Primitive.CHAR;
                if (t1 == jq_Primitive.SHORT) {
                    if (t2 == jq_Primitive.CHAR) return jq_Primitive.INT;
                    return jq_Primitive.SHORT;
                }
                if (t2 == jq_Primitive.SHORT) return jq_Primitive.SHORT;
                if (t1 == jq_Primitive.BYTE || t2 == jq_Primitive.BYTE) return jq_Primitive.BYTE;
                if (t1 == jq_Primitive.BOOLEAN || t2 == jq_Primitive.BOOLEAN) return jq_Primitive.BOOLEAN;
            }
            return null;
        }
        if (!t1.isReferenceType() || !t2.isReferenceType()) return null;
        if (t1 == jq_Reference.jq_NullType.NULL_TYPE) return t2;
        if (t2 == jq_Reference.jq_NullType.NULL_TYPE) return t1;
        int dim = 0;
        while (t1.isArrayType() && t2.isArrayType()) {
            ++dim;
            t1 = ((jq_Array)t1).getElementType();
            t2 = ((jq_Array)t2).getElementType();
        }
        if (t1.isPrimitiveType() || t2.isPrimitiveType()) {
            jq_Reference result = PrimordialClassLoader.loader.getJavaLangObject();
            --dim;
            while (--dim >= 0) result = result.getArrayTypeForElementType();
            return result;
        }
        if (!t1.isClassType() || !t2.isClassType()) {
            jq_Reference result = PrimordialClassLoader.loader.getJavaLangObject();
            while (--dim >= 0) result = result.getArrayTypeForElementType();
            return result;
        }
        jq_Class c1 = (jq_Class)t1;
        jq_Class c2 = (jq_Class)t2;
        Stack s1 = new Stack();
        do {
            if (!c1.isLoaded()) {
                if (load) c1.load();
                else c1 = PrimordialClassLoader.loader.getJavaLangObject();
            }
            s1.push(c1);
            if (c1.isLoaded()) c1 = c1.getSuperclass();
            else break;
        } while (c1 != null);
        Stack s2 = new Stack();
        do {
            if (!c2.isLoaded()) {
                if (load) c2.load();
                else c2 = PrimordialClassLoader.loader.getJavaLangObject();
            }
            s2.push(c2);
            if (c2.isLoaded()) c2 = c2.getSuperclass();
            else break;
        } while (c2 != null);
        jq_Class result = PrimordialClassLoader.loader.getJavaLangObject();
        while (!s1.empty() && !s2.empty()) {
            jq_Class temp = (jq_Class)s1.pop();
            if (temp == s2.pop()) result = temp;
            else break;
        }
        jq_Reference result2 = result;
        while (--dim >= 0) result2 = result2.getArrayTypeForElementType();
        return result2;
    }
    
    public static final jq_StaticMethod _checkcast;
    public static final jq_StaticMethod _instance_of;
    public static final jq_StaticMethod _arrayStoreCheck;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/TypeCheck;");
        _checkcast = k.getOrCreateStaticMethod("checkcast", "(Ljava/lang/Object;LClazz/jq_Type;)Ljava/lang/Object;");
        _instance_of = k.getOrCreateStaticMethod("instance_of", "(Ljava/lang/Object;LClazz/jq_Type;)Z");
        _arrayStoreCheck = k.getOrCreateStaticMethod("arrayStoreCheck", "(LMemory/HeapAddress;[Ljava/lang/Object;)V");
    }
}

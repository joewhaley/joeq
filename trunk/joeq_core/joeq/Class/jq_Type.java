/*
 * jq_Type.java
 *
 * Created on December 19, 2000, 8:38 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import UTF.Utf8;
import jq;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Bootstrap.PrimordialClassLoader;

public abstract class jq_Type {
    
    protected final Utf8 desc;
    protected final Class class_object;  // pointer to our associated java.lang.Class object

    protected jq_Type(Utf8 desc) {
        this.desc = desc;
        Class c = null;
        if (!jq.Bootstrapping)
            c = ClassLib.sun13.java.lang.Class.createNewClass(ClassLib.sun13.java.lang.Class._class, this);
        this.class_object = c;
    }
    
    public abstract String getName();
    public final Utf8 getDesc() { return desc; }
    public abstract String getJDKDesc();
    public abstract boolean isClassType();
    public abstract boolean isArrayType();
    public abstract boolean isPrimitiveType();
    public abstract boolean isIntLike();
    public final boolean isReferenceType() { return !isPrimitiveType(); }
    public abstract ClassLoader getClassLoader();
    public abstract int getReferenceSize();
    public final jq_Array getArrayTypeForElementType() {
        return (jq_Array)ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(getClassLoader(), desc.getAsArrayDescriptor());
    }
    public boolean needsDynamicLink(jq_Method method) { return false; }
    public final Class getJavaLangClassObject() {
        jq.assert(!jq.Bootstrapping);
        return class_object;
    }

    public abstract boolean isLoaded();
    public abstract boolean isVerified();
    public abstract boolean isPrepared();
    public abstract boolean isSFInitialized();
    public abstract boolean isClsInitRunning();
    public abstract boolean isClsInitialized();
    
    public abstract boolean isFinal();
    
    public abstract void load();
    public abstract void verify();
    public abstract void prepare();
    public abstract void sf_initialize();
    public abstract void cls_initialize();

    public String toString() { return getName(); }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Type;");
    public static final jq_InstanceMethod _getJavaLangClassObject = _class.getOrCreateInstanceMethod("getJavaLangClassObject", "()Ljava/lang/Class;");
}

/*
 * jq_Type.java
 *
 * Created on December 19, 2000, 8:38 AM
 *
 */

package Clazz;

import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Main.jq;
import Run_Time.Reflection;
import UTF.Utf8;

import Compil3r.Quad.AndersenInterface.AndersenType;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class jq_Type implements AndersenType {
    
    protected final Utf8 desc;
    private /*final*/ Class class_object;  // pointer to our associated java.lang.Class object

    public static final boolean USE_CLASS_OBJECT_FIELD = true;

    protected jq_Type(Utf8 desc, ClassLoader class_loader) {
        this.desc = desc;
        initializeClassObject();
    }
    
    private void initializeClassObject() {
        if (jq.RunningNative)
            this.class_object = ClassLibInterface.DEFAULT.createNewClass(this);
        else if (USE_CLASS_OBJECT_FIELD)
            this.class_object = Reflection.getJDKType(this);
    }
    
    public abstract String getName();
    public abstract String shortName();
    public final Utf8 getDesc() { return desc; }
    public abstract String getJDKDesc();
    public abstract boolean isClassType();
    public abstract boolean isArrayType();
    public abstract boolean isPrimitiveType();
    public abstract boolean isAddressType();
    public abstract boolean isIntLike();
    public final boolean isReferenceType() { return !isPrimitiveType(); }
    public abstract ClassLoader getClassLoader();
    public abstract int getReferenceSize();
    public final jq_Array getArrayTypeForElementType() {
        return (jq_Array)PrimordialClassLoader.getOrCreateType(getClassLoader(), desc.getAsArrayDescriptor());
    }
    public boolean needsDynamicLink(jq_Method method) { return false; }
    public final Class getJavaLangClassObject() {
        if (jq.RunningNative && this.class_object == null)
            initializeClassObject();
        return class_object;
    }

    public abstract boolean isLoaded();
    public abstract boolean isVerified();
    public abstract boolean isPrepared();
    public abstract boolean isSFInitialized();
    public abstract boolean isCompiled();
    public abstract boolean isClsInitRunning();
    public abstract boolean isClsInitialized();
    
    public abstract boolean isFinal();
    
    public abstract void load();
    public abstract void verify();
    public abstract void prepare();
    public abstract void sf_initialize();
    public abstract void compile();
    public abstract void cls_initialize();

    public void accept(jq_TypeVisitor tv) { tv.visitType(this); }
    
    public String toString() { return getName(); }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Type;");
}

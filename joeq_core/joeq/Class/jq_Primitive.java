/*
 * jq_Primitive.java
 *
 * Created on December 19, 2000, 8:54 AM
 *
 */

package Clazz;

import UTF.Utf8;
import Bootstrap.PrimordialClassLoader;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class jq_Primitive extends jq_Type implements jq_ClassFileConstants {

    public final boolean isClassType() { return false; }
    public final boolean isArrayType() { return false; }
    public final boolean isPrimitiveType() { return true; }
    public final String getName() { return name; }
    public final String shortName() { return name; }
    public final String getJDKDesc() { return desc.toString(); }
    public final int getReferenceSize() { return size; }
    public final ClassLoader getClassLoader() { return PrimordialClassLoader.loader; }
    public final boolean isIntLike() {
        return this == jq_Primitive.INT || this == jq_Primitive.BOOLEAN || this == jq_Primitive.BYTE ||
               this == jq_Primitive.CHAR || this == jq_Primitive.SHORT;
    }
    
    public final boolean isLoaded() { return true; }
    public final boolean isVerified() { return true; }
    public final boolean isPrepared() { return true; }
    public final boolean isSFInitialized() { return true; }
    public final boolean isClsInitRunning() { return true; }
    public final boolean isClsInitialized() { return true; }
    
    public final void load() { }
    public final void verify() { }
    public final void prepare() { }
    public final void sf_initialize() { }
    public final void cls_initialize() { }
    
    public final boolean isFinal() { return true; }
    
    public void accept(jq_TypeVisitor tv) {
        tv.visitPrimitive(this);
        super.accept(tv);
    }
    
    private final String name;
    private final int size;
    
    /** Creates new jq_Primitive */
    private jq_Primitive(Utf8 desc, String name, int size) {
        super(desc);
        this.name = name;
        this.size = size;
    }
    // ONLY to be called by PrimordialClassLoader!
    public static jq_Primitive newPrimitive(Utf8 desc, String name, int size) {
        return new jq_Primitive(desc, name, size);
    }

    public static final jq_Primitive BYTE = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_BYTE+"");
    public static final jq_Primitive CHAR = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_CHAR+"");
    public static final jq_Primitive DOUBLE = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_DOUBLE+"");
    public static final jq_Primitive FLOAT = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_FLOAT+"");
    public static final jq_Primitive INT = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_INT+"");
    public static final jq_Primitive LONG = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_LONG+"");
    public static final jq_Primitive SHORT = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_SHORT+"");
    public static final jq_Primitive BOOLEAN = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_BOOLEAN+"");
    public static final jq_Primitive VOID = (jq_Primitive)PrimordialClassLoader.loader.getOrCreateBSType((char)TC_VOID+"");
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Primitive;");
}

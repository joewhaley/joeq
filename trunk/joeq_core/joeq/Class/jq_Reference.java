/*
 * jq_Reference.java
 *
 * Created on December 19, 2000, 8:38 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import jq;
import UTF.Utf8;
import Bootstrap.PrimordialClassLoader;

public abstract class jq_Reference extends jq_Type implements jq_ClassFileConstants {

    public final int getState() { return state; }
    public final boolean isLoaded() { return state >= STATE_LOADED; }
    public final boolean isVerified() { return state >= STATE_VERIFIED; }
    public final boolean isPrepared() { return state >= STATE_PREPARED; }
    public final boolean isSFInitialized() { return state >= STATE_SFINITIALIZED; }
    public final boolean isClsInitRunning() { return state >= STATE_CLSINITRUNNING; }
    public final boolean isClsInitialized() { return state >= STATE_CLSINITIALIZED; }
    
    public final boolean isPrimitiveType() { return false; }
    
    public final ClassLoader getClassLoader() { return class_loader; }
    public final int getReferenceSize() { return 4; }
    public final Object getVTable() { chkState(STATE_PREPARED); return vtable; }
    
    public abstract String getJDKName();
    public abstract jq_Class[] getInterfaces();
    public abstract jq_Class getInterface(Utf8 desc);
    public abstract boolean implementsInterface(jq_Class k);

    public abstract jq_InstanceMethod getVirtualMethod(jq_NameAndDesc nd);
    
    public final void chkState(byte s) {
        if (state >= s) return;
        jq.UNREACHABLE(this+" actual state: "+state+" expected state: "+s);
    }

    protected jq_Reference(Utf8 desc, ClassLoader class_loader) {
        super(desc);
        jq.assert(class_loader != null);
        this.class_loader = class_loader;
    }
    protected final ClassLoader class_loader;
    protected int/*byte*/ state; // use an 'int' so we can do cas4 on it
    protected Object vtable;

    public static final jq_Class _class;
    public static final jq_InstanceField _vtable;
    public static /*final*/ jq_InstanceField _state; // set after PrimordialClassLoader finishes initialization
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Reference;");
        _vtable = _class.getOrCreateInstanceField("vtable", "Ljava/lang/Object;");
        // primitive types have not yet been created!
        _state = _class.getOrCreateInstanceField("state", "I");
    }
}

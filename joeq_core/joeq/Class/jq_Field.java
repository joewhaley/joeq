/*
 * jq_Field.java
 *
 * Created on December 19, 2000, 11:20 AM
 *
 */

package Clazz;

import java.io.DataInput;
import java.io.IOException;
import java.util.Map;

import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Compil3r.Quad.AndersenInterface.AndersenField;
import Compil3r.Quad.AndersenInterface.AndersenType;
import Main.jq;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class jq_Field extends jq_Member implements AndersenField {

    protected jq_Type type;
    
    // clazz, name, desc, access_flags are inherited
    protected jq_Field(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
        type = PrimordialClassLoader.getOrCreateType(clazz.getClassLoader(), nd.getDesc());
    }
    
    public void load(char access_flags, DataInput in) 
    throws IOException, ClassFormatError {
        super.load(access_flags, in);
        if (jq.RunningNative)
            ClassLibInterface.DEFAULT.initNewField((java.lang.reflect.Field)this.getJavaLangReflectMemberObject(), this);
    }

    public void load(char access_flags, Map attributes) {
        super.load(access_flags, attributes);
        if (jq.RunningNative)
            ClassLibInterface.DEFAULT.initNewField((java.lang.reflect.Field)this.getJavaLangReflectMemberObject(), this);
    }

    public final jq_Type getType() { return type; }
    public final AndersenType and_getType() { return getType(); }
    public boolean isVolatile() { chkState(STATE_LOADING2); return (access_flags & ACC_VOLATILE) != 0; }
    public boolean isTransient() { chkState(STATE_LOADING2); return (access_flags & ACC_TRANSIENT) != 0; }
    
    public abstract int getWidth();

    public void accept(jq_FieldVisitor mv) {
        mv.visitField(this);
    }

    static interface Delegate {
        boolean isCodeAddressType(jq_Field t);
        boolean isHeapAddressType(jq_Field t);
        boolean isStackAddressType(jq_Field t);
    }
    
    private static Delegate _delegate;

    public final boolean isCodeAddressType() {
        return _delegate.isCodeAddressType(this);
    }
    public final boolean isHeapAddressType() {
        return _delegate.isHeapAddressType(this);
    }
    public final boolean isStackAddressType() {
        return _delegate.isStackAddressType(this);
    }
    
    public String toString() { return getDeclaringClass()+"."+getName(); }

    static {
        /* Set up delegates. */
        _delegate = null;
        boolean nullVM = jq.nullVM || System.getProperty("joeq.nullvm") != null;
        if (!nullVM) {
            _delegate = attemptDelegate("Clazz.Delegates$Field");
        }
        if (_delegate == null) {
            _delegate = new NullDelegates.Field();
        }
    }

    private static Delegate attemptDelegate(String s) {
        String type = "field delegate";
        try {
            Class c = Class.forName(s);
            return (Delegate)c.newInstance();
        } catch (java.lang.ClassNotFoundException x) {
            System.err.println("Cannot find "+type+" "+s+": "+x);
        } catch (java.lang.InstantiationException x) {
            System.err.println("Cannot instantiate "+type+" "+s+": "+x);
        } catch (java.lang.IllegalAccessException x) {
            System.err.println("Cannot access "+type+" "+s+": "+x);
        }
        return null;
    }

}

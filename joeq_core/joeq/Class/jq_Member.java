/*
 * jq_Member.java
 *
 * Created on December 19, 2000, 11:29 AM
 *
 */

package Clazz;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Member;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ClassLib.ClassLibInterface;
import Main.jq;
import Memory.CodeAddress;
import Memory.StackAddress;
import Run_Time.Reflection;
import Run_Time.StackWalker;
import Run_Time.TypeCheck;
import Run_Time.Unsafe;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class jq_Member implements jq_ClassFileConstants {

    protected final void chkState(int s) {
        if (getState() >= s) return;
        jq.UNREACHABLE(this + " actual state: " + getState() + " expected state: " + s);
    }

    public final int getState() {
        return state;
    }

    public final boolean isLoaded() {
        return state >= STATE_LOADED;
    }

    //  Always available
    protected byte state;
    // pointer to the jq_Class object it's a member of
    protected /*final*/ jq_Class clazz; // made not final for class replacement
    protected /*final*/ jq_NameAndDesc nd;

    // Available after loading
    protected char access_flags;
    protected Map attributes;

    protected final Member member_object;  // pointer to our associated java.lang.reflect.Member object

    public static final boolean USE_MEMBER_OBJECT_FIELD = false;

    protected jq_Member(jq_Class clazz, jq_NameAndDesc nd) {
        jq.Assert(clazz != null);
        jq.Assert(nd != null);
        this.clazz = clazz;
        this.nd = nd;
        Member c = null;
        try {
            if (this instanceof jq_Field) {
                if (!jq.Bootstrapping) {
                    c = ClassLibInterface.DEFAULT.createNewField((jq_Field) this);
                } else if (USE_MEMBER_OBJECT_FIELD) {
                    c = Reflection.getJDKField(Class.forName(clazz.getJDKName()), nd.getName().toString());
                }
            } else if (this instanceof jq_Initializer) {
                if (!jq.Bootstrapping) {
                    c = ClassLibInterface.DEFAULT.createNewConstructor((jq_Initializer) this);
                } else if (USE_MEMBER_OBJECT_FIELD) {
                    Class[] args = Reflection.getArgTypesFromDesc(nd.getDesc());
                    c = Reflection.getJDKConstructor(Class.forName(clazz.getJDKName()), args);
                }
            } else {
                if (!jq.Bootstrapping) {
                    c = ClassLibInterface.DEFAULT.createNewMethod((jq_Method) this);
                } else if (USE_MEMBER_OBJECT_FIELD) {
                    Class[] args = Reflection.getArgTypesFromDesc(nd.getDesc());
                    c = Reflection.getJDKMethod(Class.forName(clazz.getJDKName()), nd.getName().toString(), args);
                }
            }
        } catch (ClassNotFoundException x) {
            System.err.println("Error! Class " + clazz + " not found in host JVM.");
            x.printStackTrace();
        }
        this.member_object = c;
    }

    public final Member getJavaLangReflectMemberObject() {
        //jq.Assert(!jq.Bootstrapping);
        return member_object;
    }

    /* attribute_info {
           u2 attribute_name_index;
           u4 attribute_length;
           u1 info[attribute_length];
       }
       VM Spec 4.7 */

    public void load(char access_flags, DataInput in)
            throws IOException, ClassFormatError {
        state = STATE_LOADING1;
        this.access_flags = access_flags;
        attributes = new HashMap();
        char n_attributes = (char) in.readUnsignedShort();
        for (int i = 0; i < n_attributes; ++i) {
            char attribute_name_index = (char) in.readUnsignedShort();
            if (clazz.getCPtag(attribute_name_index) != CONSTANT_Utf8)
                throw new ClassFormatError("constant pool entry " + attribute_name_index + ", referred to by attribute " + i +
                        ", is wrong type tag (expected=" + CONSTANT_Utf8 + ", actual=" + clazz.getCPtag(attribute_name_index));
            Utf8 attribute_desc = clazz.getCPasUtf8(attribute_name_index);
            int attribute_length = in.readInt();
            // todo: maybe we only want to read in attributes we care about...
            byte[] attribute_data = new byte[attribute_length];
            in.readFully(attribute_data);
            attributes.put(attribute_desc, attribute_data);
        }
        state = STATE_LOADING2;
    }

    public void load(char access_flags, Map attributes) {
        this.access_flags = access_flags;
        this.attributes = attributes;
        state = STATE_LOADING2;
    }

    public void unload() {
        state = STATE_UNLOADED;
    }

    public final void dump(DataOutput out, jq_ConstantPool.ConstantPoolRebuilder cpr) throws IOException {
        out.writeChar(access_flags);
        out.writeChar(cpr.get(getName()));
        out.writeChar(cpr.get(getDesc()));
        dumpAttributes(out, cpr);
    }

    public void dumpAttributes(DataOutput out, jq_ConstantPool.ConstantPoolRebuilder cpr) throws IOException {
        int nattributes = attributes.size();
        jq.Assert(nattributes <= Character.MAX_VALUE);
        out.writeChar(nattributes);
        for (Iterator i = attributes.entrySet().iterator(); i.hasNext();) {
            Map.Entry e = (Map.Entry) i.next();
            Utf8 name = (Utf8) e.getKey();
            out.writeChar(cpr.get(name));
            byte[] value = (byte[]) e.getValue();
            out.writeInt(value.length);
            out.write(value);
        }
    }

    // Always available
    public final jq_Class getDeclaringClass() {
        return clazz;
    }

    public final jq_NameAndDesc getNameAndDesc() {
        return nd;
    }

    public final Utf8 getName() {
        return nd.getName();
    }

    public final Utf8 getDesc() {
        return nd.getDesc();
    }

    public abstract boolean needsDynamicLink(jq_Method method);

    public final void setDeclaringClass(jq_Class k) {
        this.clazz = k;
    }
    
    public final void setNameAndDesc(jq_NameAndDesc nd) {
        this.nd = nd;
    }

    public abstract jq_Member resolve();

    // Available after loading
    public final byte[] getAttribute(Utf8 name) {
        chkState(STATE_LOADING2);
        return (byte[]) attributes.get(name);
    }

    public final byte[] getAttribute(String name) {
        return getAttribute(Utf8.get(name));
    }

    public final Map getAttributes() {
        chkState(STATE_LOADING2);
        return attributes;
    }

    public final boolean checkAccessFlag(char f) {
        chkState(STATE_LOADING2);
        return (access_flags & f) != 0;
    }

    public final char getAccessFlags() {
        chkState(STATE_LOADING2);
        return access_flags;
    }

    public final boolean isPublic() {
        return checkAccessFlag(ACC_PUBLIC);
    }

    public final boolean isPrivate() {
        return checkAccessFlag(ACC_PRIVATE);
    }

    public final boolean isProtected() {
        return checkAccessFlag(ACC_PROTECTED);
    }

    public final boolean isFinal() {
        return checkAccessFlag(ACC_FINAL);
    }

    public final boolean isSynthetic() {
        return getAttribute("Synthetic") != null;
    }

    public final boolean isDeprecated() {
        return getAttribute("Deprecated") != null;
    }

    public void checkCallerAccess(int depth) throws IllegalAccessException {
        jq_Class field_class = this.getDeclaringClass();
        if (this.isPublic() && field_class.isPublic()) {
            // completely public!
            return;
        }
        StackWalker sw = new StackWalker(CodeAddress.min(), StackAddress.getBasePointer());
        while (--depth >= 0) sw.gotoNext();
        jq_CompiledCode cc = sw.getCode();
        if (cc != null) {
            jq_Class caller_class = cc.getMethod().getDeclaringClass();
            if (caller_class == field_class) {
                // same class! access allowed!
                return;
            }
            if (field_class.isPublic() || caller_class.isInSamePackage(field_class)) {
                if (this.isPublic()) {
                    // class is accessible and field is public!
                    return;
                }
                if (this.isProtected()) {
                    if (TypeCheck.isAssignable(caller_class, field_class)) {
                        // field is protected and field_class is supertype of caller_class!
                        return;
                    }
                }
                if (!this.isPrivate()) {
                    if (caller_class.isInSamePackage(field_class)) {
                        // field is package-private and field_class and caller_class are in the same package!
                        return;
                    }
                }
            }
        }
        throw new IllegalAccessException();
    }

    // available after resolution
    public abstract boolean isStatic();
}

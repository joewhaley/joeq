// jq_Member.java, created Mon Feb  5 23:23:20 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Clazz;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Member;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ClassLib.ClassLibInterface;
import Compil3r.Quad.AndersenInterface.AndersenClass;
import Compil3r.Quad.AndersenInterface.AndersenMember;
import Main.jq;
import Run_Time.Reflection;
import UTF.Utf8;
import Util.Assert;
import Util.Collections.FilterIterator;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class jq_Member implements jq_ClassFileConstants, AndersenMember {

    protected final void chkState(int s) {
        if (getState() >= s) return;
        Assert.UNREACHABLE(this + " actual state: " + getState() + " expected state: " + s);
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

    private /*final*/ Member member_object;  // pointer to our associated java.lang.reflect.Member object

    public static final boolean USE_MEMBER_OBJECT_FIELD = true;

    protected jq_Member(jq_Class clazz, jq_NameAndDesc nd) {
        Assert._assert(clazz != null);
        Assert._assert(nd != null);
        this.clazz = clazz;
        this.nd = nd;
        initializeMemberObject();
    }

    private final void initializeMemberObject() {
        if (this instanceof jq_Field) {
            if (jq.RunningNative) {
                this.member_object = ClassLibInterface.DEFAULT.createNewField((jq_Field) this);
            } else if (USE_MEMBER_OBJECT_FIELD) {
                this.member_object = Reflection.getJDKField(Reflection.getJDKType(clazz), nd.getName().toString());
            }
        } else if (this instanceof jq_Initializer) {
            if (jq.RunningNative) {
                this.member_object = ClassLibInterface.DEFAULT.createNewConstructor((jq_Initializer) this);
            } else if (USE_MEMBER_OBJECT_FIELD) {
                Class[] args = Reflection.getArgTypesFromDesc(nd.getDesc());
                this.member_object = Reflection.getJDKConstructor(Reflection.getJDKType(clazz), args);
            }
        } else {
            if (jq.RunningNative) {
                this.member_object = ClassLibInterface.DEFAULT.createNewMethod((jq_Method) this);
            } else if (USE_MEMBER_OBJECT_FIELD) {
                Class[] args = Reflection.getArgTypesFromDesc(nd.getDesc());
                this.member_object = Reflection.getJDKMethod(Reflection.getJDKType(clazz), nd.getName().toString(), args);
            }
        }
    }

    public final Member getJavaLangReflectMemberObject() {
        if (jq.RunningNative && member_object == null)
            initializeMemberObject();
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
        Assert._assert(nattributes <= Character.MAX_VALUE);
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
    public final AndersenClass and_getDeclaringClass() {
        return getDeclaringClass();
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

    public final void removeAttribute(String name) {
        removeAttribute(Utf8.get(name));
    }
    public final void removeAttribute(Utf8 name) {
        attributes.remove(name);
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

    // Available after resolving
    public abstract boolean isStatic();
    
    public static class FilterByName extends FilterIterator.Filter {
        private java.util.regex.Pattern p;
        public FilterByName(java.util.regex.Pattern p) { this.p = p; }
        public FilterByName(String s) { this(java.util.regex.Pattern.compile(s)); }
        public boolean isElement(Object o2) {
            jq_Member m = (jq_Member) o2;
            Object o = m.getName().toString();
            CharSequence cs;
            if (o instanceof CharSequence) cs = (CharSequence) o;
            else cs = new Util.CharSequenceWrapper((String) o);
            return p.matcher(cs).matches();
        }
    }
    public static class FilterByShortClassName extends FilterIterator.Filter {
        private java.util.regex.Pattern p;
        public FilterByShortClassName(java.util.regex.Pattern p) { this.p = p; }
        public FilterByShortClassName(String s) { this(java.util.regex.Pattern.compile(s)); }
        public boolean isElement(Object o2) {
            jq_Member m = (jq_Member) o2;
            Object o = m.getDeclaringClass().shortName();
            CharSequence cs;
            if (o instanceof CharSequence) cs = (CharSequence) o;
            else cs = new Util.CharSequenceWrapper((String) o);
            return p.matcher(cs).matches();
        }
    }
    
}

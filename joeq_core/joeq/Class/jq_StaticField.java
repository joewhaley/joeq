/*
 * jq_StaticField.java
 *
 * Created on December 19, 2000, 12:34 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

//friend jq_ClassLoader;

import java.io.DataInput;
import java.io.IOException;
import java.io.DataOutput;

import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
import UTF.Utf8;
import jq;

public final class jq_StaticField extends jq_Field {

    // null if not a constant.
    private Object constantValue;

    private int/*HeapAddress*/ address;
    
    // clazz, name, desc, access_flags are inherited
    private jq_StaticField(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_StaticField newStaticField(jq_Class clazz, jq_NameAndDesc nd) {
        return new jq_StaticField(clazz, nd);
    }

    public final void load(char access_flags, DataInput in) 
    throws IOException, ClassFormatError {
        super.load(access_flags, in);
        byte[] a = getAttribute("ConstantValue");
        if (a != null) {
            if (a.length != 2) throw new ClassFormatError();
            char cpidx = jq.twoBytesToChar(a, 0);
            jq_Class clazz = getDeclaringClass();
            switch (clazz.getCPtag(cpidx)) {
                case CONSTANT_Long:
                    if (type != jq_Primitive.LONG)
                        throw new ClassFormatError();
                    constantValue = clazz.getCPasLong(cpidx);
                    break;
                case CONSTANT_Float:
                    if (type != jq_Primitive.FLOAT)
                        throw new ClassFormatError();
                    constantValue = clazz.getCPasFloat(cpidx);
                    break;
                case CONSTANT_Double:
                    if (type != jq_Primitive.DOUBLE)
                        throw new ClassFormatError();
                    constantValue = clazz.getCPasDouble(cpidx);
                    break;
                case CONSTANT_String:
                    if (type != PrimordialClassLoader.loader.getJavaLangString())
                        throw new ClassFormatError();
                    constantValue = clazz.getCPasString(cpidx);
                    break;
                case CONSTANT_Integer:
                    if (!type.isPrimitiveType() ||
                        type == jq_Primitive.LONG ||
                        type == jq_Primitive.FLOAT ||
                        type == jq_Primitive.DOUBLE)
                        throw new ClassFormatError();
                    constantValue = clazz.getCPasInt(cpidx);
                    break;
                default:
                    throw new ClassFormatError();
            }
        }
        state = STATE_LOADED;
    }
    
    public final jq_StaticField resolve() {
        this.clazz.load();
        if (this.state >= STATE_LOADED) return this;
        // this reference may be to a superclass or superinterface.
        jq_StaticField m = this.clazz.getStaticField(nd);
        if (m != null) return m;
        throw new NoSuchFieldError(this.toString());
    }
    
    public void dumpAttributes(DataOutput out, jq_ConstantPool.ConstantPoolRebuilder cpr) throws IOException {
	if (constantValue != null) {
	    byte[] b = new byte[2]; jq.charToTwoBytes(cpr.get(constantValue), b, 0);
	    attributes.put(Utf8.get("ConstantValue"), b);
	}
	super.dumpAttributes(out, cpr);
    }

    public final void sf_initialize(int[] static_data, int offset) { jq.assert(state == STATE_PREPARED); state = STATE_SFINITIALIZED; this.address = Unsafe.addressOf(static_data) + offset; }
    public final int getAddress() { chkState(STATE_SFINITIALIZED); return address; }
    public final void setValue(int v) { getDeclaringClass().setStaticData(this, v); }
    public final void setValue(float v) { getDeclaringClass().setStaticData(this, v); }
    public final void setValue(long v) { getDeclaringClass().setStaticData(this, v); }
    public final void setValue(double v) { getDeclaringClass().setStaticData(this, v); }
    public final void setValue(Object v) { getDeclaringClass().setStaticData(this, v); }
    
    public final boolean needsDynamicLink(jq_Method method) {
	return getDeclaringClass().needsDynamicLink(method);
    }
    public final boolean isConstant() { chkState(STATE_LOADED); return constantValue != null; }
    public final Object getConstantValue() { return constantValue; }
    public final boolean isStatic() { return true; }

    public final void prepare() { jq.assert(state == STATE_LOADED); state = STATE_PREPARED; }
    public final void unprepare() { jq.assert(state == STATE_PREPARED); state = STATE_LOADED; }
    
    public void accept(jq_FieldVisitor mv) {
        mv.visitStaticField(this);
        super.accept(mv);
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceField _address;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_StaticField;");
        _address = _class.getOrCreateInstanceField("address", "I");
    }
}

/*
 * jq_InstanceMethod.java
 *
 * Created on December 19, 2000, 11:23 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

//friend jq_ClassLoader;

import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;
import jq;

public class jq_InstanceMethod extends jq_Method {

    // available after preparation
    public static final int INVALID_OFFSET = 0x80000000;
    private int offset;
    private boolean isOverriding, isOverridden;
    
    // inherited: clazz, name, desc, access_flags, attributes
    //            max_stack, max_locals, bytecode, exception_table, codeattribMap,
    //            param_types, return_type
    protected jq_InstanceMethod(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
        offset = INVALID_OFFSET;
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_InstanceMethod newInstanceMethod(jq_Class clazz, jq_NameAndDesc nd) {
        return new jq_InstanceMethod(clazz, nd);
    }
    protected void parseMethodSignature() {
        Utf8.MethodDescriptorIterator i = nd.getDesc().getParamDescriptors();
        // count them up
        int num = 1, words = 1;
        while (i.hasNext()) { i.nextUtf8(); ++num; }
        // get them for real
        param_types = new jq_Type[num];
        param_types[0] = clazz;
        i = nd.getDesc().getParamDescriptors();
        for (int j=1; j<num; ++j) {
            Utf8 pd = (Utf8)i.nextUtf8();
            param_types[j] = ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(clazz.getClassLoader(), pd);
            ++words;
            if ((param_types[j] == jq_Primitive.LONG) ||
                (param_types[j] == jq_Primitive.DOUBLE)) ++words;
        }
        param_words = words;
        Utf8 rd = i.getReturnDescriptor();
        return_type = ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(clazz.getClassLoader(), rd);
    }
    public final void clearOverrideFlags() { this.isOverridden = false; this.isOverriding = false; }
    public final void isOverriddenBy(jq_InstanceMethod that) {
        this.isOverridden = true; that.isOverriding = true;
    }
    public final boolean isOverriding() { return isOverriding; }
    public final boolean isOverridden() { return isOverridden; }
    public final void setOffset(int offset) {
        jq.assert(state == STATE_LOADED); state = STATE_PREPARED; this.offset = offset;
    }
    public final int getOffset() { chkState(STATE_PREPARED); return offset; }
    public final boolean isVirtual() { chkState(STATE_PREPARED); return offset != INVALID_OFFSET; }
    public final boolean needsDynamicLink(jq_Method method) { 
        if (jq.Bootstrapping) return getDeclaringClass().needsDynamicLink(method);
        return state >= STATE_PREPARED;
    } 
    public final boolean isStatic() { return false; }
    public final void unprepare() { chkState(STATE_PREPARED); offset = INVALID_OFFSET; state = STATE_LOADED; }
    
    public boolean isInitializer() { return false; }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_InstanceMethod;");
}

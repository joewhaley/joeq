/*
 * jq_StaticMethod.java
 *
 * Created on December 19, 2000, 11:24 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

//friend jq_ClassLoader;

import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import UTF.Utf8;
import jq;

public class jq_StaticMethod extends jq_Method {

    // clazz, name, desc, access_flags are inherited
    protected jq_StaticMethod(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_StaticMethod newStaticMethod(jq_Class clazz, jq_NameAndDesc nd) {
        return new jq_StaticMethod(clazz, nd);
    }
    protected void parseMethodSignature() {
        Utf8.MethodDescriptorIterator i = nd.getDesc().getParamDescriptors();
        // count them up
        int num = 0, words = 0;
        while (i.hasNext()) { i.nextUtf8(); ++num; }
        // get them for real
        param_types = new jq_Type[num];
        i = nd.getDesc().getParamDescriptors();
        for (int j=0; j<num; ++j) {
            Utf8 pd = (Utf8)i.nextUtf8();
            param_types[j] = ClassLibInterface.i.getOrCreateType(clazz.getClassLoader(), pd);
            ++words;
            if ((param_types[j] == jq_Primitive.LONG) ||
                (param_types[j] == jq_Primitive.DOUBLE)) ++words;
        }
        param_words = words;
        Utf8 rd = i.getReturnDescriptor();
        return_type = ClassLibInterface.i.getOrCreateType(clazz.getClassLoader(), rd);
    }
    
    public final boolean needsDynamicLink(jq_Method method) {
        return getDeclaringClass().needsDynamicLink(method);
    }

    public final boolean isStatic() { return true; }
    public boolean isClassInitializer() { return false; }

    public jq_Method resolve() {
        this.clazz.load();
        if (this.state >= STATE_LOADED) return this;
        // this reference may be to a superclass or superinterface.
        jq_StaticMethod m = this.clazz.getStaticMethod(nd);
        if (m != null) return m;
        throw new NoSuchMethodError(this.toString());
    }
    
    public final void prepare() { jq.assert(state == STATE_LOADED); state = STATE_PREPARED; }

    public final void unprepare() { jq.assert(state == STATE_PREPARED); state = STATE_LOADED; }
    
    public void accept(jq_MethodVisitor mv) {
        mv.visitStaticMethod(this);
        super.accept(mv);
    }
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_StaticMethod;");
    }
}

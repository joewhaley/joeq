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
import UTF.Utf8;

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
            param_types[j] = ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(clazz.getClassLoader(), pd);
            ++words;
            if ((param_types[j] == jq_Primitive.LONG) ||
                (param_types[j] == jq_Primitive.DOUBLE)) ++words;
        }
        param_words = words;
        Utf8 rd = i.getReturnDescriptor();
        return_type = ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(clazz.getClassLoader(), rd);
    }
    
    public final boolean needsDynamicLink(jq_Method method) {
        return getDeclaringClass().needsDynamicLink(method);
    }

    public final boolean isStatic() { return true; }
    public boolean isClassInitializer() { return false; }

    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_StaticMethod;");
    }
}

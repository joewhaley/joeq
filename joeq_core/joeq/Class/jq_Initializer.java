/*
 * jq_Initializer.java
 *
 * Created on December 19, 2000, 11:26 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

//friend jq_ClassLoader;

import jq;
import Bootstrap.PrimordialClassLoader;
import UTF.Utf8;

public final class jq_Initializer extends jq_InstanceMethod {

    // clazz, name, desc are inherited
    
    private jq_Initializer(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_Initializer newInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        jq.assert(nd.getName() == Utf8.get("<init>"));
        return new jq_Initializer(clazz, nd);
    }
    protected final void parseMethodSignature() {
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
        return_type = jq_Primitive.VOID;
    }

    public final boolean isInitializer() { return true; }

    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Initializer;");
    }
}

/*
 * jq_ClassInitializer.java
 *
 * Created on December 19, 2000, 11:25 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

//friend jq_ClassLoader;

import Bootstrap.PrimordialClassLoader;
import jq;
import UTF.Utf8;

public final class jq_ClassInitializer extends jq_StaticMethod {

    // clazz, name, desc are inherited
    
    private jq_ClassInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_ClassInitializer newClassInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        jq.assert(nd.getName() == Utf8.get("<clinit>"));
        jq.assert(nd.getDesc() == Utf8.get("()V"));
        return new jq_ClassInitializer(clazz, nd);
    }

    protected final void parseMethodSignature() {
        // no need to parse anything
        param_types = new jq_Type[0];
        return_type = jq_Primitive.VOID;
    }
    
    public final boolean isClassInitializer() { return true; }

    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_ClassInitializer;");
    }
}

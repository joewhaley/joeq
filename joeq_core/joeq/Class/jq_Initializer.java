/*
 * jq_Initializer.java
 *
 * Created on December 19, 2000, 11:26 AM
 *
 */

package Clazz;

//friend jq_ClassLoader;

import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Main.jq;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class jq_Initializer extends jq_InstanceMethod {

    // clazz, name, desc are inherited
    
    private jq_Initializer(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_Initializer newInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        jq.Assert(nd.getName() == Utf8.get("<init>"));
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
            param_types[j] = PrimordialClassLoader.getOrCreateType(clazz.getClassLoader(), pd);
            ++words;
            if ((param_types[j] == jq_Primitive.LONG) ||
                (param_types[j] == jq_Primitive.DOUBLE)) ++words;
        }
        param_words = words;
        return_type = jq_Primitive.VOID;
    }

    public final jq_InstanceMethod resolve1() {
        this.clazz.load();
        if (this.state >= STATE_LOADED) return this;
        throw new NoSuchMethodError(this.toString());
    }
    
    public final boolean isInitializer() { return true; }

    public final void accept(jq_MethodVisitor mv) {
        mv.visitInitializer(this);
        super.accept(mv);
    }
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_Initializer;");
    }
}

/*
 * jq_ClassInitializer.java
 *
 * Created on December 19, 2000, 11:25 AM
 *
 */

package Clazz;

//friend jq_ClassLoader;

import Bootstrap.PrimordialClassLoader;
import Compil3r.Quad.AndersenInterface.AndersenClassInitializer;
import UTF.Utf8;
import Util.Assert;

/**
 * @author  John Whaley
 * @version $Id$
 */
public final class jq_ClassInitializer extends jq_StaticMethod implements AndersenClassInitializer {

    // clazz, nd are inherited
    
    private jq_ClassInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
    }
    // ONLY TO BE CALLED BY jq_ClassLoader!!!
    static jq_ClassInitializer newClassInitializer(jq_Class clazz, jq_NameAndDesc nd) {
        Assert._assert(nd.getName() == Utf8.get("<clinit>"));
        Assert._assert(nd.getDesc() == Utf8.get("()V"));
        return new jq_ClassInitializer(clazz, nd);
    }

    protected final void parseMethodSignature() {
        // no need to parse anything
        param_types = new jq_Type[0];
        return_type = jq_Primitive.VOID;
    }
    
    public final jq_StaticMethod resolve1() {
        this.clazz.load();
        if (this.state >= STATE_LOADED) return this;
        throw new NoSuchMethodError(this.toString());
    }
    
    public final boolean isClassInitializer() { return true; }

    public final void accept(jq_MethodVisitor mv) {
        mv.visitClassInitializer(this);
        super.accept(mv);
    }
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_ClassInitializer;");
    }
}

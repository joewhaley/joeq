/*
 * jq_Field.java
 *
 * Created on December 19, 2000, 11:20 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import java.io.DataInput;
import java.io.IOException;

public abstract class jq_Field extends jq_Member {

    protected jq_Type type;
    
    // clazz, name, desc, access_flags are inherited
    protected jq_Field(jq_Class clazz, jq_NameAndDesc nd) {
        super(clazz, nd);
        type = ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(clazz.getClassLoader(), nd.getDesc());
    }
    
    public final jq_Type getType() { return type; }
    public boolean isVolatile() { chkState(STATE_LOADING2); return (access_flags & ACC_VOLATILE) != 0; }
    public boolean isTransient() { chkState(STATE_LOADING2); return (access_flags & ACC_TRANSIENT) != 0; }
    
    public String toString() { return getDeclaringClass()+"."+getName(); }
}

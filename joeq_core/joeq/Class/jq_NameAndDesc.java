/*
 * jq_NameAndDesc.java
 *
 * Created on January 2, 2001, 12:26 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import UTF.Utf8;

public class jq_NameAndDesc {

    private Utf8 name, desc;
    
    /** Creates new jq_NameAndDesc */
    public jq_NameAndDesc(Utf8 name, Utf8 desc) {
        this.name = name;
        this.desc = desc;
    }
    
    public final Utf8 getName() { return name; }
    public final Utf8 getDesc() { return desc; }
    
    public boolean equals(Object o) { return equals((jq_NameAndDesc)o); }
    public boolean equals(jq_NameAndDesc that) {
        return this.name == that.name && this.desc == that.desc;
    }
    public int hashCode() {
        return name.hashCode() ^ desc.hashCode();
    }
    public String toString() {
        return name+" "+desc;
    }

}

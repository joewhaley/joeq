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

public final class jq_MemberReference {

    private jq_Class clazz;
    private jq_NameAndDesc nd;
    
    /** Creates new jq_NameAndDesc */
    public jq_MemberReference(jq_Class clazz, jq_NameAndDesc nd) {
        this.clazz = clazz;
        this.nd = nd;
    }
    
    public final jq_Class getReferencedClass() { return clazz; }
    public final jq_NameAndDesc getNameAndDesc() { return nd; }
    public final Utf8 getName() { return nd.getName(); }
    public final Utf8 getDesc() { return nd.getDesc(); }
    
    public boolean equals(Object o) { return equals((jq_MemberReference)o); }
    public boolean equals(jq_MemberReference that) {
        return this.clazz == that.clazz && this.nd.equals(that.nd);
    }
    public int hashCode() {
        return clazz.hashCode() ^ nd.hashCode();
    }

}
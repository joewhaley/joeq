/*
 * jq_MemberReference.java
 *
 * Created on January 2, 2001, 12:26 AM
 *
 */

package Clazz;

import UTF.Utf8;

/**
 * Objects of this class represent unresolved references to class members.
 *
 * @author  John Whaley
 * @version $Id$
 */
public final class jq_MemberReference {

    private jq_Class clazz;
    private jq_NameAndDesc nd;
    
    /** Creates new member reference to the named member in the given class.
     * @param clazz  class of the referenced member
     * @param nd  name and descriptor of the referenced member
     */
    public jq_MemberReference(jq_Class clazz, jq_NameAndDesc nd) {
        this.clazz = clazz;
        this.nd = nd;
    }
    
    /** Returns the class of the referenced member.
     * @return  class of referenced member
     */
    public final jq_Class getReferencedClass() { return clazz; }
    /** Returns the name and descriptor of the referenced member.
     * @return  name and descriptor of referenced member
     */
    public final jq_NameAndDesc getNameAndDesc() { return nd; }
    /** Returns the name of the referenced member.
     * @return  name of referenced member
     */
    public final Utf8 getName() { return nd.getName(); }
    /** Returns the descriptor of the referenced member.
     * @return  descriptor of referenced member
     */
    public final Utf8 getDesc() { return nd.getDesc(); }
    
    public boolean equals(Object o) { return equals((jq_MemberReference)o); }
    public boolean equals(jq_MemberReference that) {
        return this.clazz == that.clazz && this.nd.equals(that.nd);
    }
    public int hashCode() {
        return clazz.hashCode() ^ nd.hashCode();
    }

}
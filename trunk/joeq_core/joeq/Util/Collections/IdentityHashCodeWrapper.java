/*
 * IdentityHashCodeWrapper.java
 *
 * Created on January 11, 2001, 4:48 PM
 *
 */

package Util.Collections;

import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class IdentityHashCodeWrapper {
    
    private Object o;
    private IdentityHashCodeWrapper(Object o) {
        this.o = o;
    }
    public static IdentityHashCodeWrapper create(Object o) {
        Assert._assert(o != null);
        return new IdentityHashCodeWrapper(o);
    }
    public boolean equals(Object that) {
        if (this == that) return true;
        if (!(that instanceof IdentityHashCodeWrapper)) return false;
        return this.o == ((IdentityHashCodeWrapper)that).o;
    }
    public int hashCode() {
        return System.identityHashCode(o);
    }
    
    public Object getObject() { return o; }
    
}

/*
 * ArrayCopy.java
 *
 * Created on January 12, 2001, 12:12 PM
 *
 */

package Run_Time;

import Allocator.ObjectLayout;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Memory.HeapAddress;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class HashCode {

    public static int identityHashCode(Object x) {
        HeapAddress a = HeapAddress.addressOf(x);
        int status = a.offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
        if ((status & ObjectLayout.HASHED_MOVED) != 0) {
            jq_Reference t = jq_Reference.getTypeOf(x);
            if (t.isClassType()) {
                jq_Class k = (jq_Class) t;
                return a.offset(k.getInstanceSize() - ObjectLayout.OBJ_HEADER_SIZE).peek4();
            }
            Assert._assert(t.isArrayType());
            jq_Array k = (jq_Array) t;
            int arraylength = a.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).peek4();
            return a.offset(k.getInstanceSize(arraylength) - ObjectLayout.ARRAY_HEADER_SIZE).peek4();
        }
        a.offset(ObjectLayout.STATUS_WORD_OFFSET).poke4(status | ObjectLayout.HASHED);
        return a.to32BitValue();
    }
    
}

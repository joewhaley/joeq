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
import Main.jq;
import Memory.HeapAddress;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class HashCode implements ObjectLayout {

    public static int identityHashCode(Object x) {
        HeapAddress a = HeapAddress.addressOf(x);
        int status = a.offset(STATUS_WORD_OFFSET).peek4();
        if ((status & HASHED_MOVED) != 0) {
            jq_Reference t = jq_Reference.getTypeOf(x);
            if (t.isClassType()) {
                jq_Class k = (jq_Class) t;
                return a.offset(k.getInstanceSize() - OBJ_HEADER_SIZE).peek4();
            }
            jq.Assert(t.isArrayType());
            jq_Array k = (jq_Array) t;
            int arraylength = a.offset(ARRAY_LENGTH_OFFSET).peek4();
            return a.offset(k.getInstanceSize(arraylength) - ARRAY_HEADER_SIZE).peek4();
        }
        a.offset(STATUS_WORD_OFFSET).poke4(status | HASHED);
        return a.to32BitValue();
    }
    
}

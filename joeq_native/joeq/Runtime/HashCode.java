/*
 * ArrayCopy.java
 *
 * Created on January 12, 2001, 12:12 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Allocator.ObjectLayout;
import Clazz.jq_Reference;
import Clazz.jq_Class;
import Clazz.jq_Array;
import Run_Time.Unsafe;
import Main.jq;

public abstract class HashCode implements ObjectLayout {

    public static int identityHashCode(Object x) {
        int/*Address*/ a = Unsafe.addressOf(x);
        int status = Unsafe.peek(a+STATUS_WORD_OFFSET);
        if ((status & HASHED_MOVED) != 0) {
            jq_Reference t = Unsafe.getTypeOf(x);
            if (t.isClassType())
                return Unsafe.peek(((jq_Class)t).getInstanceSize() - OBJ_HEADER_SIZE);
            jq.Assert(t.isArrayType());
            return Unsafe.peek(((jq_Array)t).getInstanceSize(Unsafe.peek(a+ARRAY_LENGTH_OFFSET)) - ARRAY_HEADER_SIZE);
        }
        Unsafe.poke4(a+STATUS_WORD_OFFSET, status | HASHED);
        return Unsafe.addressOf(x);
    }
    
}

package Memory.Heap;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Memory.HeapAddress;

/**
 * @author John Whaley
 */
public class SizeControl {
    int first_block;
    int current_block;
    /// TODO: remove last_allocated.
    int last_allocated;
    int ndx;
    HeapAddress next_slot;
    int lastBlockToKeep;        // GSC
    
    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LMemory/Heap/SizeControl;");
    public static final jq_Array _array = _class.getArrayTypeForElementType();
}

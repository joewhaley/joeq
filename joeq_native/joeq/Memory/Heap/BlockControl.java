package Memory.Heap;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Memory.HeapAddress;

/**
 * @author John Whaley
 */
public class BlockControl {
    HeapAddress baseAddr;
    int slotsize;   // slotsize
    byte[] mark;
    byte[] alloc;
    int nextblock;
    byte[] Alloc1;
    byte[] Alloc2;
    boolean live;
    boolean sticky;
    int alloc_size; // allocated length of mark and alloc arrays

    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LMemory/Heap/BlockControl;");
    public static final jq_Array _array = _class.getArrayTypeForElementType();
}

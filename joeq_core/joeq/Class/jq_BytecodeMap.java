/*
 * jq_BytecodeMap.java
 *
 * Created on January 23, 2001, 9:12 AM
 *
 * @author  John Whaley
 * @version 
 */

package Clazz;

import Main.jq;

/**
 * This class implements a mapping from code offsets to bytecode indices.
 *
 * @author  John Whaley
 * @version 
 */
public class jq_BytecodeMap {

    /** Stores the code offsets. */
    private final int[] offset;
    /** Stores the bytecode indices. */
    private final int[] bytecode_index;
    
    /** Constructs a new bytecode map, using the given code offset and bytecode index array.
     *  The two arrays are co-indexed.  Each entry in the code offset array corresponds
     *  to an inclusive start offset of the instructions corresponding to the bytecode index
     *  in the co-indexed bytecode array.
     *  The length of the two arrays must be equal.
     *
     * @param offset  code offset array
     * @param bytecode_index  bytecode index array
     */
    public jq_BytecodeMap(int[] offset, int[] bytecode_index) {
        jq.Assert(offset.length == bytecode_index.length);
        this.offset = offset;
        this.bytecode_index = bytecode_index;
    }
    
    /** Returns the bytecode index corresponding to the given code offset, or -1 if the
     *  offset is out of range.
     * @param off  code offset to match
     * @return  bytecode index for the code offset, or -1
     */
    public int getBytecodeIndex(int off) {
        // todo: binary search
        for (int i=offset.length-1; i>=0; --i) {
            if (off > offset[i]) return bytecode_index[i];
        }
        return -1;
    }
}

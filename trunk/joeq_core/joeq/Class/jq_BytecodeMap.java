/*
 * jq_BytecodeMap.java
 *
 * Created on January 23, 2001, 9:12 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import jq;
import Run_Time.SystemInterface;

public class jq_BytecodeMap {

    final int[] offset;
    final int[] bytecode_index;
    
    public jq_BytecodeMap(int[] offset, int[] bytecode_index) {
        jq.assert(offset.length == bytecode_index.length);
        this.offset = offset;
        this.bytecode_index = bytecode_index;
    }
    
    public int getBytecodeIndex(int off) {
        // todo: binary search
        for (int i=offset.length-1; i>=0; --i) {
            if (off > offset[i]) return bytecode_index[i];
        }
        return -1;
    }
}

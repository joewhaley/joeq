package Memory.Manager;

import Allocator.ObjectLayout;

/**
 * @author John Whaley
 */
public interface GCConstants {
    /*
     * Data Fields that control the allocation of memory
     * subpools for the heap; allocate from 
     * fixed size blocks in subpools; never-copying collector
     */
    static final int[] GC_SIZEVALUES =
        {
            ObjectLayout.OBJ_HEADER_SIZE + 4,
            ObjectLayout.OBJ_HEADER_SIZE + 8,
            ObjectLayout.OBJ_HEADER_SIZE + 12,
            32,
            64,
            84,
            128,
            256,
            512,
            524,
            1024,
            2048 };
    static final int GC_SIZES = GC_SIZEVALUES.length;
    static final int GC_MAX_SMALL_SIZE =
        GC_SIZEVALUES[GC_SIZEVALUES.length - 1];
    static final int LOG_GC_BLOCKSIZE  = 14;
    static final int GC_BLOCKSIZE      = 1 << LOG_GC_BLOCKSIZE;

}

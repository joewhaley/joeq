// GCConstants.java, created Tue Dec 10 14:02:30 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Manager;

import joeq.Allocator.ObjectLayout;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
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

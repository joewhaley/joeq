// BlockControl.java, created Tue Dec 10 14:02:01 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Heap;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Memory.HeapAddress;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
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

    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Memory/Heap/BlockControl;");
    public static final jq_Array _array = _class.getArrayTypeForElementType();
}

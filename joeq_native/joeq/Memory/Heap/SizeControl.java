// SizeControl.java, created Tue Dec 10 14:02:01 2002 by joewhaley
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
public class SizeControl {
    int first_block;
    int current_block;
    /// TODO: remove last_allocated.
    int last_allocated;
    int ndx;
    HeapAddress next_slot;
    int lastBlockToKeep;        // GSC
    
    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Memory/Heap/SizeControl;");
    public static final jq_Array _array = _class.getArrayTypeForElementType();
}

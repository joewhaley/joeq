// Heap2HeapReference.java, created Tue Feb 27  2:59:43 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Memory.HeapAddress;

/**
 * Heap2HeapReference
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Heap2HeapReference extends Reloc {

    private HeapAddress from_heaploc;
    private HeapAddress to_heaploc;
    
    /** Creates new Heap2HeapReference */
    public Heap2HeapReference(HeapAddress from_heaploc, HeapAddress to_heaploc) {
        this.from_heaploc = from_heaploc; this.to_heaploc = to_heaploc;
    }

    public HeapAddress getFrom() { return from_heaploc; }
    public HeapAddress getTo() { return to_heaploc; }
    
    public void patch() {
        from_heaploc.poke(to_heaploc);
    }
    
    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(from_heaploc.to32BitValue()); // r_vaddr
        out.writeInt(1);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
    public String toString() {
        return "from heap:"+from_heaploc.stringRep()+" to heap:"+to_heaploc.stringRep();
    }
    
    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Assembler/x86/Heap2HeapReference;");
}

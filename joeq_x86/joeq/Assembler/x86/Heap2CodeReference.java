// Heap2CodeReference.java, created Tue Feb 27  2:59:43 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

import Allocator.DefaultCodeAllocator;
import Memory.CodeAddress;
import Memory.HeapAddress;

/**
 * Heap2CodeReference
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Heap2CodeReference extends Reloc {

    HeapAddress from_heaploc;
    CodeAddress to_codeloc;
    
    public Heap2CodeReference(HeapAddress from_heaploc, CodeAddress to_codeloc) {
        this.from_heaploc = from_heaploc; this.to_codeloc = to_codeloc;
    }

    public HeapAddress getFrom() { return from_heaploc; }
    public CodeAddress getTo() { return to_codeloc; }
    
    public void patch() {
        DefaultCodeAllocator.patchAbsolute(from_heaploc, to_codeloc);
    }
    
    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(from_heaploc.to32BitValue()); // r_vaddr
        out.writeInt(0);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
    public String toString() {
        return "from heap:"+from_heaploc.stringRep()+" to code:"+to_codeloc.stringRep();
    }

}

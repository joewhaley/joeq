/*
 * HeapReference.java
 *
 * Created on February 13, 2001, 9:45 PM
 *
 */

package Assembler.x86;

import java.io.IOException;
import java.io.DataOutput;

import Allocator.DefaultCodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Main.jq;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Util.LittleEndianOutputStream;

/*
 * @author  John Whaley
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

/*
 * HeapReference.java
 *
 * Created on February 13, 2001, 9:45 PM
 *
 */

package Assembler.x86;

import java.io.IOException;
import java.io.OutputStream;

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
    
    public void dumpCOFF(OutputStream out) throws IOException {
        LittleEndianOutputStream.write_s32(out, from_heaploc.to32BitValue()); // r_vaddr
        LittleEndianOutputStream.write_s32(out, 0);                 // r_symndx
        LittleEndianOutputStream.write_u16(out, Reloc.RELOC_ADDR32);// r_type
    }
    
    public String toString() {
        return "from heap:"+from_heaploc.stringRep()+" to code:"+to_codeloc.stringRep();
    }

}

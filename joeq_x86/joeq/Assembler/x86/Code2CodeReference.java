/*
 * Code2CodeReference.java
 *
 * Created on February 13, 2001, 9:58 PM
 *
 */

package Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

import Allocator.DefaultCodeAllocator;
import Memory.CodeAddress;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class Code2CodeReference extends Reloc {

    private CodeAddress from_codeloc;
    private CodeAddress to_codeloc;
    
    /** Creates new Code2HeapReference */
    public Code2CodeReference(CodeAddress from_codeloc, CodeAddress to_codeloc) {
        this.from_codeloc = from_codeloc; this.to_codeloc = to_codeloc;
    }

    public CodeAddress getFrom() { return from_codeloc; }
    public CodeAddress getTo() { return to_codeloc; }
    
    public void patch() {
        // hack: don't patch empty code references.
        if (to_codeloc != null)
            DefaultCodeAllocator.patchAbsolute(from_codeloc, to_codeloc);
    }
    
    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(from_codeloc.to32BitValue()); // r_vaddr
        out.writeInt(0);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
    public String toString() {
        return "from code:"+from_codeloc.stringRep()+" to code:"+to_codeloc.stringRep();
    }
    
}

/*
 * Code2HeapReference.java
 *
 * Created on February 13, 2001, 9:58 PM
 *
 */

package Assembler.x86;

import java.io.IOException;
import java.io.DataOutput;

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
public class Code2HeapReference extends Reloc {

    private CodeAddress from_codeloc;
    private HeapAddress to_heaploc;
    
    /** Creates new Code2HeapReference */
    public Code2HeapReference(CodeAddress from_codeloc, HeapAddress to_heaploc) {
        this.from_codeloc = from_codeloc; this.to_heaploc = to_heaploc;
    }

    public CodeAddress getFrom() { return from_codeloc; }
    public HeapAddress getTo() { return to_heaploc; }
    
    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(from_codeloc.to32BitValue()); // r_vaddr
        out.writeInt(1);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
    public String toString() {
        return "from code:"+from_codeloc.stringRep()+" to heap:"+to_heaploc.stringRep();
    }
    
}

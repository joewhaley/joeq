/*
 * Heap2HeapReference.java
 *
 * Created on February 13, 2001, 11:11 PM
 *
 */

package Assembler.x86;

import java.io.IOException;
import java.io.DataOutput;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Memory.HeapAddress;
import Util.LittleEndianOutputStream;

/*
 * @author  John Whaley
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
    
    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(from_heaploc.to32BitValue()); // r_vaddr
        out.writeInt(1);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
    public String toString() {
        return "from heap:"+from_heaploc.stringRep()+" to heap:"+to_heaploc.stringRep();
    }
    
    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/Heap2HeapReference;");
}

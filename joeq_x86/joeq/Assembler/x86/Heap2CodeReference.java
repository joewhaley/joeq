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
import Util.LittleEndianOutputStream;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class Heap2CodeReference extends Reloc {

    int/*HeapAddress*/ from_heaploc;
    int/*CodeAddress*/ to_codeloc;
    
    public Heap2CodeReference(int from_heaploc, int to_codeloc) {
        this.from_heaploc = from_heaploc; this.to_codeloc = to_codeloc;
    }

    public int/*HeapAddress*/ getFrom() { return from_heaploc; }
    public int/*CodeAddress*/ getTo() { return to_codeloc; }
    
    public void patch() {
        DefaultCodeAllocator.patchAbsolute(from_heaploc, to_codeloc);
    }
    
    public void dumpCOFF(OutputStream out) throws IOException {
        LittleEndianOutputStream.write_s32(out, from_heaploc);      // r_vaddr
        LittleEndianOutputStream.write_s32(out, 0);                 // r_symndx
        LittleEndianOutputStream.write_u16(out, Reloc.RELOC_ADDR32);// r_type
    }
    
    public String toString() {
        return "from heap:"+jq.hex8(from_heaploc)+" to code:"+jq.hex8(to_codeloc);
    }

    public static final jq_InstanceField _from_heaploc;
    public static final jq_InstanceField _to_codeloc;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/Heap2CodeReference;");
        _from_heaploc = k.getOrCreateInstanceField("from_heaploc", "I");
        _to_codeloc = k.getOrCreateInstanceField("to_codeloc", "I");
    }
}

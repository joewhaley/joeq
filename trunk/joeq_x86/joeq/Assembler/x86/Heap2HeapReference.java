/*
 * Heap2HeapReference.java
 *
 * Created on February 13, 2001, 11:11 PM
 *
 * @author  John Whaley
 * @version 
 */

package Assembler.x86;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Util.LittleEndianOutputStream;

import java.io.OutputStream;
import java.io.IOException;

public class Heap2HeapReference extends Reloc {

    private int/*HeapAddress*/ from_heaploc;
    private int/*HeapAddress*/ to_heaploc;
    
    /** Creates new Heap2HeapReference */
    public Heap2HeapReference(int from_heaploc, int to_heaploc) {
        this.from_heaploc = from_heaploc; this.to_heaploc = to_heaploc;
    }

    public void dump(OutputStream out) throws IOException {
        LittleEndianOutputStream.write_s32(out, from_heaploc);      // r_vaddr
        LittleEndianOutputStream.write_s32(out, 1);                 // r_symndx
        LittleEndianOutputStream.write_u16(out, Reloc.RELOC_ADDR32);// r_type
    }
    
    public static final jq_InstanceField _from_heaploc;
    public static final jq_InstanceField _to_heaploc;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/Heap2HeapReference;");
        _from_heaploc = k.getOrCreateInstanceField("from_heaploc", "I");
        _to_heaploc = k.getOrCreateInstanceField("to_heaploc", "I");
    }
}

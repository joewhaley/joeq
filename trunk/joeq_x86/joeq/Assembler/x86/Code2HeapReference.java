/*
 * Code2HeapReference.java
 *
 * Created on February 13, 2001, 9:58 PM
 *
 */

package Assembler.x86;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Util.LittleEndianOutputStream;
import Main.jq;

import java.io.IOException;
import java.io.OutputStream;

/*
 * @author  John Whaley
 * @version 
 */
public class Code2HeapReference extends Reloc {

    private int/*CodeAddress*/ from_codeloc;
    private int/*HeapAddress*/ to_heaploc;
    
    /** Creates new Code2HeapReference */
    public Code2HeapReference(int from_codeloc, int to_heaploc) {
        this.from_codeloc = from_codeloc; this.to_heaploc = to_heaploc;
    }

    public int/*CodeAddress*/ getFrom() { return from_codeloc; }
    public int/*HeapAddress*/ getTo() { return to_heaploc; }
    
    public void dumpCOFF(OutputStream out)
    throws IOException {
        LittleEndianOutputStream.write_s32(out, from_codeloc);      // r_vaddr
        LittleEndianOutputStream.write_s32(out, 1);                 // r_symndx
        LittleEndianOutputStream.write_u16(out, Reloc.RELOC_ADDR32);// r_type
    }
    
    public String toString() {
        return "from code:"+jq.hex8(from_codeloc)+" to heap:"+jq.hex8(to_heaploc);
    }
    
    public static final jq_InstanceField _from_codeloc;
    public static final jq_InstanceField _to_heaploc;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/Code2HeapReference;");
        _from_codeloc = k.getOrCreateInstanceField("from_codeloc", "I");
        _to_heaploc = k.getOrCreateInstanceField("to_heaploc", "I");
    }
}

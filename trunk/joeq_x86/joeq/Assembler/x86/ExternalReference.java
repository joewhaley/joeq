/*
 * ExternalReference.java
 *
 * Created on February 13, 2001, 11:22 PM
 *
 */

package Assembler.x86;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Util.LittleEndianOutputStream;

import Main.jq;

import java.io.OutputStream;
import java.io.IOException;

/*
 * @author  John Whaley
 * @version 
 */
public class ExternalReference extends Reloc {

    private int/*HeapAddress*/ heap_from;
    private int symbol_ndx;
    private String external_name;
    
    /** Creates new ExternalReference */
    public ExternalReference(int heap_from, String external_name) {
        this.heap_from = heap_from;
        this.external_name = external_name;
    }

    public void setSymbolIndex(int ndx) { jq.Assert(ndx != 0); this.symbol_ndx = ndx; }
    
    public void dumpCOFF(OutputStream out) throws IOException {
        jq.Assert(symbol_ndx != 0);
        LittleEndianOutputStream.write_s32(out, heap_from);         // r_vaddr
        LittleEndianOutputStream.write_s32(out, symbol_ndx);        // r_symndx
        LittleEndianOutputStream.write_u16(out, Reloc.RELOC_ADDR32);// r_type
    }
    
    public int/*HeapAddress*/ getAddress() { return heap_from; }
    public int getSymbolIndex() { return symbol_ndx; }
    public String getName() { return external_name; }
    
    public String toString() {
        return "from heap:"+jq.hex8(heap_from)+" to external:"+external_name+" (symndx "+symbol_ndx+")";
    }
    
    /*
    public static final jq_InstanceField _heap_from;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/ExternalReference;");
        _heap_from = k.getOrCreateInstanceField("heap_from", "I");
    }
     */
}

/*
 * ExternalReference.java
 *
 * Created on February 13, 2001, 11:22 PM
 *
 */

package Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

import Memory.HeapAddress;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class ExternalReference extends Reloc {

    private HeapAddress heap_from;
    private int symbol_ndx;
    private String external_name;
    
    /** Creates new ExternalReference */
    public ExternalReference(HeapAddress heap_from, String external_name) {
        this.heap_from = heap_from;
        this.external_name = external_name;
    }

    public void setSymbolIndex(int ndx) { Assert._assert(ndx != 0); this.symbol_ndx = ndx; }
    
    public void dumpCOFF(DataOutput out) throws IOException {
        Assert._assert(symbol_ndx != 0);
        out.writeInt(heap_from.to32BitValue()); // r_vaddr
        out.writeInt(symbol_ndx);               // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);      // r_type
    }
    
    public HeapAddress getAddress() { return heap_from; }
    public int getSymbolIndex() { return symbol_ndx; }
    public String getName() { return external_name; }
    
    public void patch() { Assert.UNREACHABLE(); }
    
    public String toString() {
        return "from heap:"+heap_from.stringRep()+" to external:"+external_name+" (symndx "+symbol_ndx+")";
    }
    
}

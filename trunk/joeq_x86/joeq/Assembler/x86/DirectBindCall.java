/*
 * DirectBindCall.java
 *
 * Created on February 13, 2001, 9:36 PM
 *
 */

package Assembler.x86;

import java.io.DataOutput;
import java.io.IOException;

import Allocator.DefaultCodeAllocator;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Memory.CodeAddress;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class DirectBindCall extends Reloc {

    private CodeAddress source;
    private jq_Method target;

    public DirectBindCall(CodeAddress source, jq_Method target) {
        this.source = source; this.target = target;
    }
    
    public void patch() {
        patchTo(target.getDefaultCompiledVersion());
    }
    
    public void patchTo(jq_CompiledCode cc) {
        Assert._assert(cc != null);
        DefaultCodeAllocator.patchRelativeOffset(source, cc.getEntrypoint());
    }
    
    public CodeAddress getSource() { return source; }
    public jq_Method getTarget() { return target; }

    public String toString() {
        return "from code:"+source.stringRep()+" to method:"+target;
    }

    public void dumpCOFF(DataOutput out) throws IOException {
        out.writeInt(source.to32BitValue());       // r_vaddr
        out.writeInt(0);                           // r_symndx
        out.writeChar(Reloc.RELOC_ADDR32);         // r_type
    }
    
}

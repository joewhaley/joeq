/*
 * DirectBindCall.java
 *
 * Created on February 13, 2001, 9:36 PM
 *
 */

package Assembler.x86;

import Allocator.DefaultCodeAllocator;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Memory.CodeAddress;
import Util.Assert;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class DirectBindCall {

    private CodeAddress source;
    private jq_Method target;

    public DirectBindCall(CodeAddress source, jq_Method target) {
        this.source = source; this.target = target;
    }
    
    public void patch() {
        jq_CompiledCode cc = target.getDefaultCompiledVersion();
        Assert._assert(cc != null);
        DefaultCodeAllocator.patchRelativeOffset(source, cc.getEntrypoint());
    }
    
    public void patchTo(jq_CompiledCode cc) {
        DefaultCodeAllocator.patchRelativeOffset(source, cc.getEntrypoint());
    }
    
    public CodeAddress getSource() { return source; }
    public jq_Method getTarget() { return target; }

    public String toString() {
        return "from code:"+source.stringRep()+" to method:"+target;
    }
    
}

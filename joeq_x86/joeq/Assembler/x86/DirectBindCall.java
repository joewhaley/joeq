/*
 * DirectBindCall.java
 *
 * Created on February 13, 2001, 9:36 PM
 *
 */

package Assembler.x86;

import Allocator.DefaultCodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_InstanceField;

import Main.jq;

/*
 * @author  John Whaley
 * @version 
 */
public class DirectBindCall {

    private int/*CodeAddress*/ source;
    private jq_Method target;

    public DirectBindCall(int/*CodeAddress*/ source, jq_Method target) {
        this.source = source; this.target = target;
    }
    
    public void patch() {
        jq_CompiledCode cc = target.getDefaultCompiledVersion();
        jq.Assert(cc != null);
        DefaultCodeAllocator.patchRelativeOffset(source, cc.getEntrypoint());
    }
    
    public int/*CodeAddress*/ getSource() { return source; }
    public jq_Method getTarget() { return target; }

    public String toString() {
        return "from code:"+jq.hex8(source)+" to method:"+target;
    }
    
    public static final jq_InstanceField _source;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAssembler/x86/DirectBindCall;");
        _source = k.getOrCreateInstanceField("source", "I");
    }
}

/*
 * CodeCache.java
 *
 * Created on January 30, 2002, 8:45 PM
 */

package Compil3r.Quad;
import Clazz.*;

/**
 *
 * @author  Administrator
 * @version 
 */
public class CodeCache {

    public static final CodeCache cache = new CodeCache();

    protected java.util.HashMap map = new java.util.HashMap();
    
    /** Creates new CodeCache */
    public CodeCache() { }

    public static ControlFlowGraph getCode(jq_Method m) { return cache._get(m); }
    
    protected ControlFlowGraph _get(jq_Method m) {
        ControlFlowGraph cfg = (ControlFlowGraph)map.get(m);
        if (cfg == null) {
            BytecodeToQuad b2q = new BytecodeToQuad(m);
            cfg = b2q.convert();
            map.put(m, cfg);
        }
        return cfg;
    }
    
}

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

    public static java.util.List/*<ControlFlowGraphVisitor>*/ passes = new java.util.LinkedList();
    
    /** Creates new CodeCache */
    public CodeCache() { }

    public static ControlFlowGraph getCode(jq_Method m) { return cache._get(m); }

    public static boolean TRACE = false;
    
    protected ControlFlowGraph _get(jq_Method m) {
        ControlFlowGraph cfg = (ControlFlowGraph)map.get(m);
        if (cfg == null) {
	    if (TRACE) System.out.println("Generating quads for "+m);
            BytecodeToQuad b2q = new BytecodeToQuad(m);
            cfg = b2q.convert();
            map.put(m, cfg);
            for (java.util.Iterator i = passes.iterator(); i.hasNext(); ) {
                ControlFlowGraphVisitor v = (ControlFlowGraphVisitor)i.next();
                v.visitCFG(cfg);
            }
        }
        return cfg;
    }
    
}

/*
 * CallGraph.java
 * 
 * Created on Mar 3, 2003
 */
package Compil3r.Quad;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import Clazz.jq_Method;
import Compil3r.Quad.Operator.Invoke;
import Util.Assert;

/**
 * Abstract representation of a call graph.
 * 
 * @author John Whaley
 * @version $Id$
 */
public abstract class CallGraph {
    
    /**
     * Returns the possible target methods of the given call site under the given context.
     * The interpretation of the context object is specific to the type of call graph.
     * 
     * @param context
     * @param callSite
     * @return Collection of jq_Methods that are the possible targets
     */
    public abstract Collection/*<jq_Method>*/ getTargetMethods(Object context, ProgramLocation callSite);
    
    public Collection/*<jq_Method>*/ getTargetMethods(ProgramLocation callSite) {
        return getTargetMethods(null, callSite);
    }
    
    public int numberOfTargetMethods(Object context, ProgramLocation callSite) {
        return getTargetMethods(context, callSite).size();
    }
    
    public int numberOfTargetMethods(ProgramLocation callSite) {
        return numberOfTargetMethods(null, callSite);
    }

    public jq_Method getTargetMethod(Object context, ProgramLocation callSite) {
        Collection c = getTargetMethods(context, callSite);
        Assert._assert(c.size() == 1);
        return (jq_Method) c.iterator().next();
    }
    
    public static Collection/*<ProgramLocation>*/ getCallSites(jq_Method caller) {
        if (caller.getBytecode() == null) return Collections.EMPTY_SET;
        ControlFlowGraph cfg = CodeCache.getCode(caller);
        return getCallSites(cfg);
    }
    
    public static Collection/*<ProgramLocation>*/ getCallSites(ControlFlowGraph cfg) {
        LinkedList result = new LinkedList();
        for (QuadIterator i = new QuadIterator(cfg); i.hasNext(); ) {
            Quad q = i.nextQuad();
            if (q.getOperator() instanceof Invoke)
                result.add(new ProgramLocation.QuadProgramLocation(cfg.getMethod(), q));
        }
        return result;
    }
    
    public Collection[] findDepths(Collection/*jq_Method*/ roots) {
        HashSet visited = new HashSet();
        LinkedList result = new LinkedList();
        LinkedList previous = new LinkedList();
        visited.addAll(roots);
        previous.addAll(roots);
        while (!previous.isEmpty()) {
            result.add(previous);
            LinkedList current = new LinkedList();
            for (Iterator i=previous.iterator(); i.hasNext(); ) {
                jq_Method caller = (jq_Method) i.next();
                for (Iterator j=getCallSites(caller).iterator(); j.hasNext(); ) {
                    ProgramLocation cs = (ProgramLocation) j.next();
                    for (Iterator k=getTargetMethods(cs).iterator(); k.hasNext(); ) {
                        jq_Method callee = (jq_Method) k.next();
                        if (visited.contains(callee)) {
                            // back or cross edge in call graph.
                            continue;
                        }
                        visited.add(callee);
                        current.add(callee);
                    }
                }
            }
            previous = current;
        }
        
        return (Collection[]) result.toArray(new Collection[result.size()]);
    }
    
    /*
    public Collection getTargets(Object context, ProgramLocation callSite) {
        Collection c = getTargetMethods(context, callSite);
        Collection r = new LinkedList();
        for (Iterator i=c.iterator(); i.hasNext(); ) {
            jq_Method target = (jq_Method) i.next();
            ControlFlowGraph cfg = CodeCache.getCode(target);
            r.add(cfg);
        }
        return r;
    }
    
    public ControlFlowGraph getTarget(Object context, ProgramLocation callSite) {
        jq_Method target = getTargetMethod(context, callSite);
        ControlFlowGraph cfg = CodeCache.getCode(target);
        return cfg;
    }
    */
}

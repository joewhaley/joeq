// ExceptionAnalysis.java, created May 5, 2004 2:10:39 AM by joewhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Analysis.IPA;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import java.lang.ref.SoftReference;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import joeq.Compiler.Quad.BasicBlock;
import joeq.Compiler.Quad.BasicBlockVisitor;
import joeq.Compiler.Quad.CallGraph;
import joeq.Compiler.Quad.CodeCache;
import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.QuadVisitor;
import joeq.Compiler.Quad.Operator.Invoke;
import joeq.Compiler.Quad.Operator.Return;
import joeq.Util.Graphs.SCComponent;
import joeq.Util.Graphs.Traversals;

/**
 * Uses a call graph to figure out what exceptions can be thrown by a method invocation.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class ExceptionAnalysis {
    
    class QVisit implements BasicBlockVisitor {
        private final jq_Method method;
        private final Set s;
        boolean change;

        private QVisit(jq_Method method, Set s) {
            super();
            this.method = method;
            this.s = s;
        }

        public void visitBasicBlock(final BasicBlock bb) {
            bb.visitQuads(new QuadVisitor.EmptyVisitor() {
                public void visitExceptionThrower(Quad obj) {
                    if (obj.getOperator() instanceof Invoke) return;
                    if (obj.getOperator() instanceof Return.THROW_A) return;
                    for (Iterator i = obj.getThrownExceptions().iterator(); i.hasNext(); ) {
                        jq_Class r = (jq_Class) i.next();
                        if (s.contains(r)) continue;
                        if (bb.getExceptionHandlers().mustCatch(r) != null) continue;
                        if (s.add(r)) change = true;
                    }
                }
                public void visitReturn(Quad obj) {
                    if (obj.getOperator() instanceof Return.THROW_A) {
                        // todo: use-def chain to figure out where this came from.
                    }
                }
                public void visitInvoke(Quad obj) {
                    ProgramLocation pl = new QuadProgramLocation(method, obj);
                    if (getThrownExceptions(pl, s)) change = true;
                }
            });
        }
    }

    static final boolean USE_SOFTREF = true;
    
    CallGraph cg;
    Map cache;
    Map recursive;
    
    /**
     * Construct exception analysis using the given call graph.
     */
    public ExceptionAnalysis(CallGraph cg) {
        this.cg = cg;
        this.cache = new HashMap();
        this.recursive = new HashMap();
    }
    
    private void findRecursiveMethods() {
        Set/*SCComponent*/ roots = SCComponent.buildSCC(cg);
        List list = Traversals.reversePostOrder(SCComponent.SCC_NAVIGATOR, roots);
        for (Iterator i = list.iterator(); i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (scc.isLoop()) {
                for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                    this.recursive.put(j.next(), scc);
                }
            }
        }
    }
    
    /**
     * Return the set of exception types that can be thrown by this call.
     * 
     * @param callSite call site
     * @return set of exception types
     */
    public Set getThrownExceptions(ProgramLocation callSite) {
        Set s = new HashSet();
        getThrownExceptions(callSite, s);
        return s;
    }
    
    /**
     * Add the set of exception types that can be thrown by this call to
     * the given set.  Returns true iff the set changed.
     * 
     * @param callSite call site
     * @param s set
     * @return whether set changed
     */
    public boolean getThrownExceptions(ProgramLocation callSite, Set s) {
        Collection targets = cg.getTargetMethods(callSite);
        boolean change = false;
        for (Iterator i = targets.iterator(); i.hasNext(); ) {
            if (s.addAll(getThrownExceptions((jq_Method) i.next())))
                change = true;
        }
        return change;
    }
    
    private Set checkCache(jq_Method method) {
        Object o = cache.get(method);
        if (USE_SOFTREF && o instanceof SoftReference) {
            return (Set) ((SoftReference) o).get();
        } else {
            return (Set) o;
        }
    }
    
    /**
     * Return the set of exception types that can be thrown by this method.
     * 
     * @param method
     * @return set of exception types
     */
    public Set getThrownExceptions(jq_Method method) {
        Set s = checkCache(method);
        if (s != null) return s;
        
        SCComponent scc = (SCComponent) recursive.get(method);
        if (scc != null) {
            iterateScc(scc);
        } else {
            s = new HashSet();
            cache.put(method, USE_SOFTREF ? (Object) new SoftReference(s) : (Object) s);
            calcThrownExceptions(method, s);
        }
        return s;
    }
    
    private boolean calcThrownExceptions(final jq_Method method, final Set s) {
        if (method.getBytecode() == null) {
            if (method.isAbstract()) {
                return s.add(PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/AbstractMethodError;"));
            }
            if (method.isNative()) {
                // Native methods can throw arbitrary exceptions.
                boolean change = s.add(null);
                return s.add(PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/LinkageError;")) || change;
            }
        }
        
        ControlFlowGraph cfg = CodeCache.getCode(method);
        QVisit qv = new QVisit(method, s);
        cfg.visitBasicBlocks(qv);
        return qv.change;
    }

    private void iterateScc(SCComponent scc) {
        // Pre-allocate all cache entries, so we don't reenter iterateScc on
        // methods in the same SCC.
        for (Iterator i = scc.nodeSet().iterator(); i.hasNext(); ) {
            jq_Method method = (jq_Method) i.next();
            Set s = checkCache(method);
            if (s == null) {
                s = new HashSet();
                cache.put(method, USE_SOFTREF ? (Object) new SoftReference(s) : (Object) s);
            }
        }
        // Iterate until no more changes.
        boolean change;
        do {
            change = false;
            for (Iterator i = scc.nodeSet().iterator(); i.hasNext(); ) {
                jq_Method method = (jq_Method) i.next();
                Set s = checkCache(method);
                if (USE_SOFTREF && s == null) {
                    // Soft reference was cleared, make it a hard reference so that
                    // we can finish the SCC iteration.
                    s = new HashSet();
                    cache.put(method, s);
                }
                if (calcThrownExceptions(method, s)) {
                    change = true;
                }
            }
        } while (change);
    }
}

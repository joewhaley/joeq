package joeq.Compil3r.Analysis.IPA;

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import joeq.Main.Helper;
import joeq.Util.Assert;
import joeq.Util.Graphs.SCCTopSortedGraph;
import joeq.Util.Graphs.SCComponent;
import joeq.Util.Graphs.Traversals;
import joeq.Clazz.jq_Class;
import joeq.Clazz.jq_Method;
import joeq.Compil3r.Quad.BasicBlock;
import joeq.Compil3r.Quad.CallGraph;
import joeq.Compil3r.Quad.CodeCache;
import joeq.Compil3r.Quad.ControlFlowGraph;
import joeq.Compil3r.Quad.ControlFlowGraphVisitor;
import joeq.Compil3r.Quad.LoadedCallGraph;
import joeq.Compil3r.Quad.Quad;
import joeq.Compil3r.Quad.QuadVisitor;

/**
 * @author jwhaley
 * @version $Id$
 */
public class LoopAnalysis implements ControlFlowGraphVisitor {

    public static void main(String[] args) throws IOException {
        jq_Class c = (jq_Class) Helper.load(args[0]);
        CodeCache.AlwaysMap = true;
        CallGraph cg = new LoadedCallGraph("callgraph");
        LoopAnalysis a = new LoopAnalysis(cg);
        Helper.runPass(c, a);
        System.out.println("Visited methods: "+a.visitedMethods);
        System.out.println("Loop methods: "+a.loopMethods);
        System.out.println("Loop BB: "+a.loopBB);
    }

    CallGraph cg;
    jq_Method caller;
    Set visitedMethods = new HashSet();
    Set loopMethods = new HashSet();
    Set loopBB = new HashSet();

    public LoopAnalysis() {
    }
    
    public LoopAnalysis(CallGraph cg) {
        this.cg = cg;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.ControlFlowGraphVisitor#visitCFG(Compil3r.Quad.ControlFlowGraph)
     */
    public void visitCFG(ControlFlowGraph cfg) {
        caller = cfg.getMethod();
        if (visitedMethods.contains(caller))
            return;
        visitedMethods.add(caller);
        
        // Find SCCs.
        Set roots = SCComponent.buildSCC(cfg);
        SCCTopSortedGraph g = SCCTopSortedGraph.topSort(roots);
        
        // Find loops.
        for (Iterator i = Traversals.reversePostOrder(g.getNavigator(), roots).iterator();
             i.hasNext(); ) {
            SCComponent scc = (SCComponent) i.next();
            if (scc.isLoop()) {
                for (Iterator j = scc.nodeSet().iterator(); j.hasNext(); ) {
                    BasicBlock bb = (BasicBlock) j.next();
                    loopBB.add(bb);
                    if (cg != null)
                        bb.visitQuads(invoke_visitor);
                }
            }
        }
    }
    
    public boolean isInLoop(jq_Method m, BasicBlock bb) {
        if (loopMethods.contains(m)) return true;
        if (!visitedMethods.contains(m)) {
            visitCFG(CodeCache.getCode(m));
            if (loopMethods.contains(m)) return true;
        }
        if (loopBB.contains(bb)) return true;
        return false;
    }
    
    InvokeVisitor invoke_visitor = new InvokeVisitor();
    public class InvokeVisitor extends QuadVisitor.EmptyVisitor {
        public void visitInvoke(Quad q) {
            super.visitInvoke(q);
            Assert._assert(caller != null);
            Assert._assert(q != null);
            ProgramLocation mc = new ProgramLocation.QuadProgramLocation(caller, q);
            LinkedList w = new LinkedList();
            w.add(mc);
            while (!w.isEmpty()) {
                mc = (ProgramLocation) w.removeFirst();
                Collection targets = cg.getTargetMethods(mc);
                for (Iterator i = targets.iterator(); i.hasNext(); ) {
                    jq_Method callee = (jq_Method) i.next();
                    boolean change = loopMethods.add(callee);
                    if (change) {
                        w.addAll(cg.getCallSites(callee));
                    }
                }
            }
        }
    }
}

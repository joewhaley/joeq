package Compil3r.Analysis.IPA;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.BasicBlockVisitor;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.LoadedCallGraph;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadVisitor;
import Compil3r.Quad.Operator.New;
import Main.HostedVM;
import Util.Collections.GenericMultiMap;
import Util.Collections.MultiMap;
import Util.Graphs.Graph;
import Util.Graphs.Navigator;
import Util.Graphs.PathNumbering;

/**
 * @author jwhaley
 */
public class ObjectCreationGraph extends QuadVisitor.EmptyVisitor
    implements ControlFlowGraphVisitor, BasicBlockVisitor, Graph
{

    jq_Method currentMethod;
    Set/*jq_Class*/ roots = new HashSet();
    MultiMap succ = new GenericMultiMap(), pred = new GenericMultiMap();

    class Nav implements Navigator {

        /* (non-Javadoc)
         * @see Util.Graphs.Navigator#next(java.lang.Object)
         */
        public Collection next(Object node) {
            return succ.getValues(node);
        }

        /* (non-Javadoc)
         * @see Util.Graphs.Navigator#prev(java.lang.Object)
         */
        public Collection prev(Object node) {
            return pred.getValues(node);
        }
        
    }
    
    /* (non-Javadoc)
     * @see Util.Graphs.Graph#getRoots()
     */
    public Collection getRoots() {
        return roots;
    }

    /* (non-Javadoc)
     * @see Util.Graphs.Graph#getNavigator()
     */
    public Navigator getNavigator() {
        return new Nav();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.ControlFlowGraphVisitor#visitCFG(Compil3r.Quad.ControlFlowGraph)
     */
    public void visitCFG(ControlFlowGraph cfg) {
        currentMethod = cfg.getMethod();
        cfg.visitBasicBlocks(this);
    }
    

    /* (non-Javadoc)
     * @see Compil3r.Quad.BasicBlockVisitor#visitBasicBlock(Compil3r.Quad.BasicBlock)
     */
    public void visitBasicBlock(BasicBlock bb) {
        bb.visitQuads(this);
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.QuadVisitor#visitNew(Compil3r.Quad.Quad)
     */
    public void visitNew(Quad obj) {
        jq_Class c1, c2;
        if (currentMethod.isStatic()) {
            c1 = null;
        } else {
            c1 = currentMethod.getDeclaringClass();
        }
        c2 = (jq_Class) New.getType(obj).getType();
        addEdge(c1, c2);
    }
    
    public void addRoot(jq_Class c) {
        roots.add(c);
    }
    
    public void addEdge(jq_Class c1, jq_Class c2) {
        succ.add(c1, c2);
        pred.add(c2, c1);
    }
    
    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        ObjectCreationGraph g = new ObjectCreationGraph();
        CallGraph cg = new LoadedCallGraph("callgraph");
        g.addRoot(null);
        if (false) {
            for (Iterator i = cg.getRoots().iterator(); i.hasNext(); ) {
                jq_Method m = (jq_Method) i.next();
                jq_Class c = m.getDeclaringClass();
                g.addRoot(c);
            }
        }
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            ControlFlowGraph cfg = CodeCache.getCode(m);
            g.visitCFG(cfg);
        }
        
        PathNumbering n = new PathNumbering(g);
        DataOutputStream out = new DataOutputStream(new FileOutputStream("creation_graph.dot"));
        n.dotGraph(out);
        out.close();
    }

}

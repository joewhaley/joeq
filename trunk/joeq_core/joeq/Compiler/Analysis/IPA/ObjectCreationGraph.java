package Compil3r.Analysis.IPA;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
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
import Compil3r.Quad.Operator.NewArray;
import Main.HostedVM;
import Util.Collections.GenericMultiMap;
import Util.Collections.MultiMap;
import Util.Graphs.Graph;
import Util.Graphs.Navigator;
import Util.Graphs.PathNumbering;
import Util.Graphs.Traversals;

/**
 * @author jwhaley
 */
public class ObjectCreationGraph extends QuadVisitor.EmptyVisitor
    implements ControlFlowGraphVisitor, BasicBlockVisitor, Graph
{

    boolean TRACE = false;
    PrintStream out = System.out;
    boolean MERGE_SITES = false;
    
    jq_Method currentMethod;
    Set/*jq_Class*/ roots = new HashSet();
    MultiMap succ = new GenericMultiMap(), pred = new GenericMultiMap();

    public Set getAllNodes() {
        HashSet s = new HashSet();
        s.addAll(succ.keySet());
        s.addAll(pred.keySet());
        return s;
    }
    
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
        HashSet set = new HashSet();
        set.addAll(succ.keySet());
        set.removeAll(succ.values());
        set.addAll(roots);
        if (TRACE) System.out.println("Roots = "+roots);
        return set;
    }

    /* (non-Javadoc)
     * @see Util.Graphs.Graph#getNavigator()
     */
    public Navigator getNavigator() {
        return new Nav();
    }

    public void visitMethodSummary(MethodSummary ms) {
        for (Iterator j = ms.nodeIterator(); j.hasNext(); ) {
            Node n = (Node) j.next();
            visitNode(n);
        }
    }
    
    public void visitNode(Node n) {
        if (n instanceof ConcreteTypeNode ||
            n instanceof UnknownTypeNode ||
            n instanceof ConcreteObjectNode) {
            jq_Reference type = n.getDeclaredType(); 
            if (type != null) {
                jq_Class c = currentMethod.isStatic() ? null : currentMethod.getDeclaringClass();
                addEdge(c, n, type);
            }
        }
        
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
        jq_Reference c1, c2;
        if (currentMethod.isStatic()) {
            c1 = null;
        } else {
            c1 = currentMethod.getDeclaringClass();
        }
        c2 = (jq_Reference) New.getType(obj).getType();
        ProgramLocation pl = new ProgramLocation.QuadProgramLocation(currentMethod, obj);
        addEdge(c1, pl, c2);
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.QuadVisitor#visitNew(Compil3r.Quad.Quad)
     */
    public void visitNewArray(Quad obj) {
        jq_Reference c1, c2;
        if (currentMethod.isStatic()) {
            c1 = null;
        } else {
            c1 = currentMethod.getDeclaringClass();
        }
        c2 = (jq_Reference) NewArray.getType(obj).getType();
        ProgramLocation pl = new ProgramLocation.QuadProgramLocation(currentMethod, obj);
        addEdge(c1, pl, c2);
    }
    
    public void addRoot(jq_Class c) {
        if (TRACE) out.println("Adding root "+c);
        roots.add(c);
    }
    
    public void addEdge(jq_Reference c1, Node n, jq_Reference c2) {
        if (MERGE_SITES || n == null) {
            if (TRACE) out.println("Adding edge "+c1+" -> "+c2);
            succ.add(c1, c2);
            pred.add(c2, c1);
        } else {
            if (TRACE) out.println("Adding edge "+c1+" -> "+n+" -> "+c2);
            succ.add(c1, n);
            succ.add(n, c2);
            pred.add(c2, n);
            pred.add(n, c1);
        }
    }
    
    public void addEdge(jq_Reference c1, ProgramLocation pl, jq_Reference c2) {
        if (MERGE_SITES || pl == null) {
            if (TRACE) out.println("Adding edge "+c1+" -> "+c2);
            succ.add(c1, c2);
            pred.add(c2, c1);
        } else {
            if (TRACE) out.println("Adding edge "+c1+" -> "+pl+" -> "+c2);
            succ.add(c1, pl);
            succ.add(pl, c2);
            pred.add(c2, pl);
            pred.add(pl, c1);
        }
    }
    
    public void handleCallGraph(CallGraph cg) {
        addRoot(null);
        int j = 0;
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ++j) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            ControlFlowGraph cfg = CodeCache.getCode(m);
            visitCFG(cfg);
            if (j % 100 == 0)
                System.out.print("Visited methods: "+j+"\r");
        }
        System.out.println("Visited methods: "+j);
    }
    
    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        
        ObjectCreationGraph g = new ObjectCreationGraph();
        String callgraphFile = System.getProperty("callgraph", "callgraph");
        CallGraph cg = new LoadedCallGraph(callgraphFile);
        
        g.handleCallGraph(cg);
        
        PathNumbering n = new PathNumbering();
        Number paths = n.countPaths(g);
        System.out.println("Number of paths: "+paths.longValue());
        
        for (Iterator i = Traversals.reversePostOrder(g.getNavigator(), (Object)null).iterator(); i.hasNext(); ) {
            jq_Reference c = (jq_Reference) i.next();
            long v = n.numberOfPathsTo(c).longValue();
            if (v > 1L) {
                System.out.println(v+" paths to "+c);
            }
        }
        
        DataOutputStream out = new DataOutputStream(new FileOutputStream("creation_graph.dot"));
        n.dotGraph(out);
        out.close();
    }

}

/*
 * AndersenPointerAnalysis.java
 *
 * Created on May 2, 2002, 1:03 PM
 */

package Compil3r.Quad;

import Clazz.*;
import Compil3r.BytecodeAnalysis.CallTargets;
import MethodSummary.MethodCall;
import MethodSummary.PassedParameter;
import MethodSummary.CallSite;
import MethodSummary.Node;
import MethodSummary.ConcreteTypeNode;
import MethodSummary.OutsideNode;
import MethodSummary.GlobalNode;
import MethodSummary.FieldNode;
import MethodSummary.ParamNode;
import MethodSummary.ReturnValueNode;
import MethodSummary.ThrownExceptionNode;
import Operator.Invoke;
import Operand.ParamListOperand;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedList;
import Util.LinkedHashSet;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class AndersenPointerAnalysis {

    public static java.io.PrintStream out = System.out;
    public static final boolean TRACE = false;
    public static final boolean TRACE_CHANGE = false;
    public static final boolean TRACE_CYCLES = false;
    public static boolean FORCE_GC = false;

    public static final class Visitor implements ControlFlowGraphVisitor {
        public void visitCFG(ControlFlowGraph cfg) {
            INSTANCE.methodsToVisit.add(cfg);
            INSTANCE.iterate();
            System.out.println("Result after analyzing "+cfg.getMethod()+":");
            System.out.println(INSTANCE.dumpResults());
        }
    }
    
    public void initializeStatics() {
        // add initializations for System.in/out/err
        jq_Class fd = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileDescriptor;");
        fd.load();
        ConcreteTypeNode fd_n1 = new ConcreteTypeNode(fd);
        jq_Initializer fd_init = (jq_Initializer)fd.getOrCreateInstanceMethod("<init>", "(I)V");
        jq.assert(fd_init.isLoaded());
        MethodCall mc_fd_init = new MethodCall(fd_init, null);
        fd_n1.recordPassedParameter(mc_fd_init, 0);
        
        jq_Class fis = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileInputStream;");
        fis.load();
        ConcreteTypeNode fis_n = new ConcreteTypeNode(fis);
        jq_Initializer fis_init = (jq_Initializer)fis.getOrCreateInstanceMethod("<init>", "(Ljava/io/FileDescriptor;)V");
        jq.assert(fis_init.isLoaded());
        MethodCall mc_fis_init = new MethodCall(fis_init, null);
        fis_n.recordPassedParameter(mc_fis_init, 0);
        fd_n1.recordPassedParameter(mc_fis_init, 1);
        jq_Class bis = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/BufferedInputStream;");
        bis.load();
        ConcreteTypeNode bis_n = new ConcreteTypeNode(bis);
        jq_Initializer bis_init = (jq_Initializer)bis.getOrCreateInstanceMethod("<init>", "(Ljava/io/InputStream;)V");
        jq.assert(bis_init.isLoaded());
        MethodCall mc_bis_init = new MethodCall(bis_init, null);
        bis_n.recordPassedParameter(mc_bis_init, 0);
        fis_n.recordPassedParameter(mc_bis_init, 1);
        
        jq_Class jls = Bootstrap.PrimordialClassLoader.getJavaLangSystem();
	jls.load();
        jq_StaticField si = jls.getOrCreateStaticField("in", "Ljava/io/InputStream;");
        jq.assert(si.isLoaded());
        GlobalNode.GLOBAL.addEdge(si, bis_n);
        
        ControlFlowGraph fd_init_cfg = CodeCache.getCode(fd_init);
        MethodSummary fd_init_summary = MethodSummary.getSummary(fd_init_cfg);
        OutsideNode on = fd_init_summary.getParamNode(0);
        addInclusionEdge(on, fd_n1);
        ControlFlowGraph fis_init_cfg = CodeCache.getCode(fis_init);
        MethodSummary fis_init_summary = MethodSummary.getSummary(fis_init_cfg);
        on = fis_init_summary.getParamNode(0);
        addInclusionEdge(on, fis_n);
        on = fis_init_summary.getParamNode(1);
        addInclusionEdge(on, fd_n1);
        ControlFlowGraph bis_init_cfg = CodeCache.getCode(bis_init);
        MethodSummary bis_init_summary = MethodSummary.getSummary(bis_init_cfg);
        on = bis_init_summary.getParamNode(0);
        addInclusionEdge(on, bis_n);
        on = bis_init_summary.getParamNode(1);
        addInclusionEdge(on, fis_n);
        
        ConcreteTypeNode fd_n2 = new ConcreteTypeNode(fd);
        fd_n2.recordPassedParameter(mc_fd_init, 0);
        jq_Class fos = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileOutputStream;");
        fos.load();
        ConcreteTypeNode fos_n1 = new ConcreteTypeNode(fos);
        jq_Initializer fos_init = (jq_Initializer)fos.getOrCreateInstanceMethod("<init>", "(Ljava/io/FileDescriptor;)V");
        jq.assert(fos_init.isLoaded());
        MethodCall mc_fos_init = new MethodCall(fos_init, null);
        fos_n1.recordPassedParameter(mc_fos_init, 0);
        fd_n2.recordPassedParameter(mc_fos_init, 1);
        jq_Class bos = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/BufferedOutputStream;");
        bos.load();
        ConcreteTypeNode bos_n1 = new ConcreteTypeNode(bos);
        jq_Initializer bos_init = (jq_Initializer)bos.getOrCreateInstanceMethod("<init>", "(Ljava/io/OutputStream;I)V");
        jq.assert(bos_init.isLoaded());
        MethodCall mc_bos_init = new MethodCall(bos_init, null);
        bos_n1.recordPassedParameter(mc_bos_init, 0);
        fos_n1.recordPassedParameter(mc_bos_init, 1);
        
        jq_Class ps = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/PrintStream;");
        ps.load();
        ConcreteTypeNode ps_n1 = new ConcreteTypeNode(ps);
        jq_Initializer ps_init = (jq_Initializer)ps.getOrCreateInstanceMethod("<init>", "(Ljava/io/OutputStream;Z)V");
        jq.assert(ps_init.isLoaded());
        MethodCall mc_ps_init = new MethodCall(ps_init, null);
        ps_n1.recordPassedParameter(mc_ps_init, 0);
        bos_n1.recordPassedParameter(mc_ps_init, 1);
        
        jq_StaticField so = jls.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
        jq.assert(so.isLoaded());
        GlobalNode.GLOBAL.addEdge(so, ps_n1);
        
        ConcreteTypeNode fd_n3 = new ConcreteTypeNode(fd);
        fd_n3.recordPassedParameter(mc_fd_init, 0);
        ConcreteTypeNode fos_n2 = new ConcreteTypeNode(fos);
        fos_n2.recordPassedParameter(mc_fos_init, 0);
        fd_n3.recordPassedParameter(mc_fos_init, 1);
        ConcreteTypeNode bos_n2 = new ConcreteTypeNode(bos);
        bos_n2.recordPassedParameter(mc_bos_init, 0);
        fos_n2.recordPassedParameter(mc_bos_init, 1);
        ConcreteTypeNode ps_n2 = new ConcreteTypeNode(ps);
        ps_n2.recordPassedParameter(mc_ps_init, 0);
        bos_n2.recordPassedParameter(mc_ps_init, 1);
        
        so = jls.getOrCreateStaticField("err", "Ljava/io/PrintStream;");
        jq.assert(so.isLoaded());
        GlobalNode.GLOBAL.addEdge(so, ps_n2);
        
        on = fd_init_summary.getParamNode(0);
        addInclusionEdge(on, fd_n2);
        addInclusionEdge(on, fd_n3);
        ControlFlowGraph fos_init_cfg = CodeCache.getCode(fos_init);
        MethodSummary fos_init_summary = MethodSummary.getSummary(fos_init_cfg);
        on = fos_init_summary.getParamNode(0);
        addInclusionEdge(on, fos_n1);
        addInclusionEdge(on, fos_n2);
        on = fos_init_summary.getParamNode(1);
        addInclusionEdge(on, fd_n2);
        addInclusionEdge(on, fd_n3);
        ControlFlowGraph bos_init_cfg = CodeCache.getCode(bos_init);
        MethodSummary bos_init_summary = MethodSummary.getSummary(bos_init_cfg);
        on = bos_init_summary.getParamNode(0);
        addInclusionEdge(on, bos_n1);
        addInclusionEdge(on, bos_n2);
        on = bos_init_summary.getParamNode(1);
        addInclusionEdge(on, fos_n1);
        addInclusionEdge(on, fos_n2);
        ControlFlowGraph ps_init_cfg = CodeCache.getCode(ps_init);
        MethodSummary ps_init_summary = MethodSummary.getSummary(ps_init_cfg);
        on = ps_init_summary.getParamNode(0);
        addInclusionEdge(on, ps_n1);
        addInclusionEdge(on, ps_n2);
        on = ps_init_summary.getParamNode(1);
        addInclusionEdge(on, bos_n1);
        addInclusionEdge(on, bos_n2);
        
        methodsToVisit.add(fd_init_cfg);
        methodsToVisit.add(fis_init_cfg);
        methodsToVisit.add(bis_init_cfg);
        methodsToVisit.add(fos_init_cfg);
        methodsToVisit.add(bos_init_cfg);
        methodsToVisit.add(ps_init_cfg);
    }
    
    /** Cache: Maps a node to its set of corresponding concrete nodes. */
    final HashMap nodeToConcreteNodes;
    
    /** Maps a node to its set of outgoing inclusion edges. */
    final HashMap nodeToInclusionEdges;
    
    /** Set of all MethodSummary's that we care about. */
    final HashSet methodsToVisit;
    
    /** Maps a call site to its set of targets. */
    final HashMap callSiteToTargets;
    
    /** The set of method call->targets that have already been linked. */
    final HashSet linkedTargets;
    
    /** Change flag, for iterations. */
    boolean change;
    
    /** Creates new AndersenPointerAnalysis */
    public AndersenPointerAnalysis() {
        nodeToConcreteNodes = new HashMap();
        nodeToInclusionEdges = new HashMap();
        methodsToVisit = new HashSet();
        callSiteToTargets = new HashMap();
        linkedTargets = new HashSet();
        this.initializeStatics();
    }

    public static AndersenPointerAnalysis INSTANCE = new AndersenPointerAnalysis();
    
    public static final String lineSep = System.getProperty("line.separator");
    
    public static final int HISTOGRAM_SIZE = 50;
    
    public String dumpResults() {
        StringBuffer sb = new StringBuffer();
        int[] histogram = new int[HISTOGRAM_SIZE];
        for (Iterator i=callSiteToTargets.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            CallSite cs = (CallSite)e.getKey();
            Set s = (Set)e.getValue();
            sb.append(cs.toString());
            sb.append(": {");
            int x = s.size();
            sb.append(x);
            if (x >= HISTOGRAM_SIZE) x = HISTOGRAM_SIZE-1;
            histogram[x]++;
            sb.append("} ");
            sb.append(s.toString());
            sb.append(lineSep);
        }
        for (int i=0; i<HISTOGRAM_SIZE; ++i) {
            if (histogram[i] > 0) {
                sb.append(i);
                sb.append(" targets:\t");
                sb.append(histogram[i]);
                sb.append(" call site");
                if (histogram[i] > 1) sb.append('s');
                sb.append(lineSep);
            }
        }
        return sb.toString();
    }

    void iterate() {
        int count = 1;
        for (;;) {
            this.change = false;
            System.err.println("Iteration "+count+": "+methodsToVisit.size()+" methods "+callSiteToTargets.size()+" call sites "+linkedTargets.size()+" call graph edges");
            doGlobals();
            LinkedList ll = new LinkedList();
            ll.addAll(methodsToVisit);
            for (Iterator i=ll.iterator(); i.hasNext(); ) {
                ControlFlowGraph cfg = (ControlFlowGraph)i.next();
                visitMethod(cfg);
            }
            if (!change) break;
            nodeToConcreteNodes.clear();
	    if (FORCE_GC) System.gc();
            ++count;
        }
    }
    
    void doGlobals() {
        if (TRACE) out.println("Doing global variables...");
        for (Iterator j=GlobalNode.GLOBAL.getAccessPathEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry)j.next();
            jq_Field f = (jq_Field)e.getKey();
            Object o = e.getValue();
            if (!MethodSummary.IGNORE_STATIC_FIELDS) {
                jq_Class c = f.getDeclaringClass();
                if (TRACE) out.println("Visiting edge: "+o+" = "+c+((f==null)?"[]":("."+f.getName())));
                c.load();
                jq_ClassInitializer clinit = c.getClassInitializer();
                if (clinit != null) {
                    ControlFlowGraph clinit_cfg = CodeCache.getCode(clinit);
                    if (methodsToVisit.add(clinit_cfg)) {
                        if (TRACE_CHANGE && !this.change) {
                            out.println("Changed! New clinit method: "+clinit);
                        }
                        this.change = true;
                    }
                }
            }
            // o = n.f
            if (o instanceof HashSet) {
                addGlobalEdges((HashSet)o, f);
            } else {
                addGlobalEdges((FieldNode)o, f);
            }
        }
    }
    
    void visitMethod(ControlFlowGraph cfg) {
        if (TRACE) out.println("Visiting method: "+cfg.getMethod());
        MethodSummary ms = MethodSummary.getSummary(cfg);
        // find edges in graph
        for (Iterator i=ms.nodeIterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            for (Iterator j=n.getEdges().iterator(); j.hasNext(); ) {
                Map.Entry e = (Map.Entry)j.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                if (TRACE) out.println("Visiting edge: "+n+((f==null)?"[]":("."+f.getName()))+" = "+o);
                // n.f = o
                if (o instanceof HashSet) {
                    addEdgesFromConcreteNodes(n, f, (HashSet)o);
                } else {
                    addEdgesFromConcreteNodes(n, f, (Node)o);
                }
            }
            for (Iterator j=n.getAccessPathEdges().iterator(); j.hasNext(); ) {
                Map.Entry e = (Map.Entry)j.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                if (TRACE) out.println("Visiting edge: "+o+" = "+n+((f==null)?"[]":("."+f.getName())));
                // o = n.f
                if (o instanceof HashSet) {
                    addInclusionEdgesToConcreteNodes((HashSet)o, n, f);
                } else {
                    addInclusionEdgesToConcreteNodes((FieldNode)o, n, f);
                }
            }
        }
        
        // find all methods that we call.
        for (Iterator i=ms.getCalls().iterator(); i.hasNext(); ) {
            MethodCall mc = (MethodCall)i.next();
            CallSite cs = new CallSite(ms, mc);
            if (TRACE) out.println("Found call: "+cs);
            CallTargets ct = mc.getCallTargets();
            if (TRACE) out.println("Possible targets ignoring type information: "+ct);
            HashSet definite_targets = new HashSet();
            //jq.assert(!callSiteToTargets.containsKey(cs));
            callSiteToTargets.put(cs, definite_targets);
            if (ct.size() == 1 && ct.isComplete()) {
                // call can be statically resolved to a single target.
                if (TRACE) out.println("Call is statically resolved to a single target.");
                definite_targets.add(ct.iterator().next());
            } else {
                // use the type information about the receiver object to find targets.
                HashSet set = new HashSet();
                PassedParameter pp = new PassedParameter(mc, 0);
                ms.getNodesThatCall(pp, set);
                if (TRACE) out.println("Possible nodes for receiver object: "+set);
                for (Iterator j=set.iterator(); j.hasNext(); ) {
                    Node base = (Node)j.next();
                    if (TRACE) out.println("Checking base node: "+base);
                    Set s_cn = getConcreteNodes(base);
                    definite_targets.addAll(mc.getCallTargets(s_cn));
                }
            }
            if (TRACE) out.println("Set of definite targets of "+mc+": "+definite_targets);
            for (Iterator j=definite_targets.iterator(); j.hasNext(); ) {
                jq_Method callee = (jq_Method)j.next();
                callee.getDeclaringClass().load();
                if (callee.getBytecode() == null) {
                    if (TRACE) out.println(callee+" is a native method, skipping analysis...");
                    continue;
                }
                ControlFlowGraph callee_cfg = CodeCache.getCode(callee);
                MethodSummary callee_summary = MethodSummary.getSummary(callee_cfg);
                CallSite cs2 = new CallSite(callee_summary, mc);
                if (linkedTargets.contains(cs2)) continue;
                linkedTargets.add(cs2);
                if (TRACE_CHANGE && !this.change) {
                    out.println("Changed! New target for "+mc+": "+callee_summary.getMethod());
                }
                this.change = true;
                addParameterAndReturnMappings(ms, mc, callee_summary);
                methodsToVisit.add(callee_cfg);
            }
        }
    }

    void addParameterAndReturnMappings(MethodSummary caller, MethodCall mc, MethodSummary callee) {
        if (TRACE) out.println("Adding parameter and return mappings for "+mc+" from "+caller.getMethod()+" to "+callee.getMethod());
        ParamListOperand plo = Invoke.getParamList(mc.q);
        for (int i=0; i<plo.length(); ++i) {
            jq_Type t = plo.get(i).getType();
            if (!(t instanceof jq_Reference)) continue;
            ParamNode pn = callee.getParamNode(i);
            PassedParameter pp = new PassedParameter(mc, i);
            HashSet s = new HashSet();
            caller.getNodesThatCall(pp, s);
            //s.add(pn);
            if (TRACE) out.println("Adding parameter mapping "+pn+" to set "+s);
            OutsideNode on = pn;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s);
        }
        ReturnValueNode rvn = (ReturnValueNode)caller.nodes.get(new ReturnValueNode(mc));
        if (rvn != null) {
            HashSet s = (HashSet)callee.returned.clone();
            //s.add(rvn);
            if (TRACE) out.println("Adding return mapping "+rvn+" to set "+s);
            OutsideNode on = rvn;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s);
        }
        ThrownExceptionNode ten = (ThrownExceptionNode)caller.nodes.get(new ThrownExceptionNode(mc));
        if (ten != null) {
            HashSet s = (HashSet)callee.thrown.clone();
            //s.add(ten);
            if (TRACE) out.println("Adding thrown mapping "+ten+" to set "+s);
            OutsideNode on = ten;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s);
        }
    }
    
    boolean addInclusionEdges(OutsideNode n, Set s) {
        jq.assert(n.skip == null);
        Set s2 = (Set)nodeToInclusionEdges.get(n);
        if (s2 == null) {
            nodeToInclusionEdges.put(n, s);
            if (TRACE_CHANGE && !this.change) {
                out.println("Changed! New set of inclusion edges for node "+n);
            }
            this.change = true;
            return true;
        } else {
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Object o = i.next();
                if (o instanceof OutsideNode) {
                    OutsideNode on = (OutsideNode)o;
                    while (on.skip != null) {
                        on = on.skip;
                    }
                    o = on;
                }
                if (n == o) continue;
                if (s2.add(o)) {
                    if (TRACE_CHANGE && !this.change) {
                        out.println("Changed! New inclusion edge for node "+n+": "+o);
                    }
                    this.change = true;
                }
            }
            return false;
        }
    }
    
    void addInclusionEdge(OutsideNode n, Node s) {
        jq.assert(n.skip == null);
        if (s instanceof OutsideNode) {
            OutsideNode on = (OutsideNode)s;
            while (on.skip != null) {
                on = on.skip;
            }
            s = on;
        }
        if (n == s) return;
        Set s2 = (Set)nodeToInclusionEdges.get(n);
        if (s2 == null) {
            nodeToInclusionEdges.put(n, s2 = new HashSet()); s2.add(s);
            if (TRACE_CHANGE && !this.change) {
                out.println("Changed! New set of inclusion edges for node "+n);
            }
            this.change = true;
        } else if (s2.add(s)) {
            if (TRACE_CHANGE && !this.change) {
                out.println("Changed! New inclusion edge for node "+n+": "+s);
            }
            this.change = true;
        }
    }
    
    Set getInclusionEdges(Node n) { return (Set)nodeToInclusionEdges.get(n); }
    
    static boolean checkInvalidFieldAccess(Node n, jq_Field f) {
        jq_Reference rtype = n.getDeclaredType();
        if (rtype == null) {
            if (TRACE) out.println("Node "+n+" is null, so cannot hold field access");
            return true;
        }
        if (MethodSummary.IGNORE_INSTANCE_FIELDS) return false;
        if (f == null) {
            if (rtype instanceof jq_Class) {
                if (TRACE) out.println("Node "+n+" is a class type, so it cannot hold array access");
                return true;
            }
        } else {
            if (!(rtype instanceof jq_Class)) {
                if (TRACE) out.println("Node "+n+" is an array type, so it cannot hold field access");
                return true;
            }
            jq_Class rclass = (jq_Class)rtype;
            rclass.load();
            if (rclass.getInstanceField(f.getNameAndDesc()) != f) {
                if (TRACE) out.println("Node "+n+" does not contain field "+f);
                return true;
            }
        }
        return false;
    }
    
    // from.f = to
    void addEdgesFromConcreteNodes(Node from, jq_Field f, HashSet to) {
        Set s = getConcreteNodes(from);
        if (TRACE) out.println("Node "+from+" corresponds to concrete nodes "+s);
        for (Iterator i=s.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if (checkInvalidFieldAccess(n, f)) continue;
            if (n.addEdges(f, to)) {
                if (TRACE_CHANGE && !this.change) {
                    out.println("Changed! New edges for concrete node "+n+"."+f+": "+to);
                }
                this.change = true;
            }
        }
    }
    
    // from.f = to
    void addEdgesFromConcreteNodes(Node from, jq_Field f, Node to) {
        Set s = getConcreteNodes(from);
        if (TRACE) out.println("Node "+from+" corresponds to concrete nodes "+s);
        for (Iterator i=s.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if (checkInvalidFieldAccess(n, f)) continue;
            if (n.addEdge(f, to)) {
                if (TRACE_CHANGE && !this.change) {
                    out.println("Changed! New edge for concrete node "+n+"."+f+": "+to);
                }
                this.change = true;
            }
        }
    }
    
    // from = global.f
    void addGlobalEdges(OutsideNode from, jq_Field f) {
        HashSet result = new HashSet();
        GlobalNode.GLOBAL.getEdges(f, result);
        while (from.skip != null) from = from.skip;
        addInclusionEdges(from, result);
    }
    
    // from = global.f
    void addGlobalEdges(HashSet from, jq_Field f) {
        HashSet result = new HashSet();
        GlobalNode.GLOBAL.getEdges(f, result);
        for (Iterator j=from.iterator(); j.hasNext(); ) {
            OutsideNode n2 = (OutsideNode)j.next();
            while (n2.skip != null) n2 = n2.skip;
            if (addInclusionEdges(n2, result)) result = (HashSet)result.clone();
        }
    }
    
    // from = base.f
    void addInclusionEdgesToConcreteNodes(HashSet from, Node base, jq_Field f) {
        Set s = getConcreteNodes(base);
        if (TRACE) out.println("Node "+base+" corresponds to concrete nodes "+s);
        HashSet result = new HashSet();
        for (Iterator j=s.iterator(); j.hasNext(); ) {
            Node n2 = (Node)j.next();
            n2.getEdges(f, result);
        }
        for (Iterator j=from.iterator(); j.hasNext(); ) {
            OutsideNode n2 = (OutsideNode)j.next();
            while (n2.skip != null) n2 = n2.skip;
            if (addInclusionEdges(n2, result)) result = (HashSet)result.clone();
        }
    }
    
    // from = base.f
    void addInclusionEdgesToConcreteNodes(OutsideNode from, Node base, jq_Field f) {
        Set s = getConcreteNodes(base);
        if (TRACE) out.println("Node "+base+" corresponds to concrete nodes "+s);
        HashSet result = new HashSet();
        for (Iterator j=s.iterator(); j.hasNext(); ) {
            Node n2 = (Node)j.next();
            n2.getEdges(f, result);
        }
        while (from.skip != null) from = from.skip;
        addInclusionEdges(from, result);
    }
    
    Set getConcreteNodes(Node from) {
        if (from instanceof OutsideNode) 
            return getConcreteNodes((OutsideNode)from, null);
        else
            return Collections.singleton(from);
    }
    
    Set getConcreteNodes(OutsideNode from, Path p) {
        while (from.skip != null) {
            from = from.skip;
        }
        if (from.visited) {
            Path p2 = p;
            if (TRACE_CYCLES) out.println("cycle detected! node="+from+" path="+p);
            HashSet s = (HashSet)nodeToInclusionEdges.get(from);
            jq.assert(s != null);
            OutsideNode last = from;
            for (;; p = p.cdr()) {
                OutsideNode n = p.car();
                if (n == from) break;
                if (TRACE) out.println("next in path: "+n+", merging into: "+from);
                jq.assert(n.skip == null);
                n.skip = from;
                HashSet s2 = (HashSet)nodeToInclusionEdges.get(n);
                //s2.remove(last);
                if (TRACE) out.println("Set of inclusion edges from node "+n+": "+s2);
                s.addAll(s2);
                nodeToInclusionEdges.put(n, s);
                last = n;
            }
            //s.remove(last);
            for (Iterator i=s.iterator(); i.hasNext(); ) {
                Object o = i.next();
                if (o instanceof OutsideNode) {
                    OutsideNode on = (OutsideNode)o;
                    while (on.skip != null) on = on.skip;
                    o = on;
                }
                if (from == o) {
                    if (TRACE) out.println("Node "+from+" contains transitive self-edge, removing.");
                    i.remove();
                }
            }
            if (TRACE) out.println("Final set of inclusion edges from node "+from+": "+s);
            return null;
        }
        Set cached = (Set)nodeToConcreteNodes.get(from);
        if (cached != null) return cached;
        p = new Path(from, p);
        Set s = (Set)nodeToInclusionEdges.get(from);
        HashSet result = new HashSet();
        nodeToConcreteNodes.put(from, result);
        if (s == null) return Collections.EMPTY_SET;
        for (;;) {
            Iterator i = s.iterator();
            for (;;) {
                if (!i.hasNext()) {
                    return result;
                }
                Node to = (Node)i.next();
                if (to instanceof OutsideNode) {
                    if (TRACE) out.println("Visiting inclusion edge "+from+" --> "+to+"...");
                    from.visited = true;
                    Set s2 = getConcreteNodes((OutsideNode)to, p);
                    from.visited = false;
                    if (from.skip != null) {
                        if (TRACE) out.println("Node "+from+" is skipped...");
                        return null;
                    }
                    if (s2 == null) {
                        if (TRACE) out.println("Nodes were merged into "+from);
                        if (TRACE) out.println("redoing iteration on "+s);
                        jq.assert(nodeToInclusionEdges.get(from) == s);
                        break;
                    } else {
                        result.addAll(s2);
                    }
                } else {
                    if (TRACE) out.println("Found concrete node "+to);
                    result.add(to);
                }
            }
        }
    }
    
    public static class Path {
        private final OutsideNode s;
        private final Path next;
        Path(OutsideNode s, Path n) { this.s = s; this.next = n; }
        OutsideNode car() { return s; }
        Path cdr() { return next; }
        public String toString() {
            if (next == null) return s.toString();
            return s.toString()+"->"+next.toString();
        }
    }
    
}

/*
 * AndersenPointerAnalysis.java
 *
 * Created on May 2, 2002, 1:03 PM
 */

package Compil3r.Quad;

import Clazz.*;
import Compil3r.BytecodeAnalysis.CallTargets;
import Compil3r.Quad.MethodSummary.MethodCall;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.OutsideNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.ReturnValueNode;
import Compil3r.Quad.MethodSummary.ThrownExceptionNode;
import Compil3r.Quad.MethodSummary.UnknownTypeNode;
import Compil3r.Quad.Operator.Invoke;
import Compil3r.Quad.Operand.ParamListOperand;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedList;
import Util.LinkedHashSet;
import Util.Default;
import Main.jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class AndersenPointerAnalysis {

    public static java.io.PrintStream out = System.out;
    public static /*final*/ boolean TRACE = false;
    public static final boolean TRACE_CHANGE = false;
    public static final boolean TRACE_CYCLES = false;
    public static final boolean VerifyAssertions = false;
    public static boolean FULL_DUMP = false;
    public static boolean COMPARE_RTA = false;
    public static boolean DO_TWICE = false;
    
    public static final boolean HANDLE_ESCAPE = true;
    public static final boolean USE_SOFT_REFERENCES = true;
    public static boolean FORCE_GC = false;
    public static final boolean REUSE_CACHES = true;
    public static final boolean TRACK_CHANGES = true;
    public static final boolean TRACK_CHANGED_FIELDS = false; // doesn't work.
    public static boolean TRACK_SOURCE_QUADS = true;

    public boolean addToRootSet(ControlFlowGraph cfg) {
        if (TRACE) out.println("Adding "+cfg.getMethod()+" to root set.");
        return this.rootSet.add(cfg);
    }
    
    public static final class Visitor implements ControlFlowGraphVisitor {
        public static boolean added_hook = false;
        public void visitCFG(ControlFlowGraph cfg) {
            INSTANCE.addToRootSet(cfg);
            if (!added_hook) {
                added_hook = true;
                Runtime.getRuntime().addShutdownHook(new Thread() {
                    public void run() {
                        doIt();
                    }
                });
                if (TRACE) out.println("Added Andersen shutdown hook.");
            }
        }
        public static void doIt() {
            Set rootSet = INSTANCE.rootSet;
            long time = System.currentTimeMillis();
            INSTANCE.iterate();
            time = System.currentTimeMillis() - time;
            System.out.println("First time: "+time);

            if (DO_TWICE) {
                long mem1 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                System.out.println("Used memory before gc: "+mem1);
                INSTANCE = new AndersenPointerAnalysis(true);
                INSTANCE.rootSet.addAll(rootSet);
                System.gc();
                long mem2 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                System.out.println("Used memory after gc: "+mem2);
                time = System.currentTimeMillis();
                INSTANCE.iterate();
                time = System.currentTimeMillis() - time;
            
                long mem3 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                System.out.println("Used memory before gc: "+mem3);
                System.gc();
                long mem4 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                System.out.println("Used memory after gc: "+mem4);

                System.out.println("Our analysis: "+(time/1000.)+" seconds, "+(mem2-mem2)+" bytes of memory");
            }
            
            System.out.println("Result of Andersen pointer analysis:");
            if (FULL_DUMP)
                System.out.println(dumpResults(INSTANCE.callSiteToTargets));
            System.out.println(INSTANCE.computeStats());
            System.out.println(computeHistogram(INSTANCE.callSiteToTargets));
            
            if (COMPARE_RTA) {
                calcRTA();

                System.out.println("Compare to CHA/RTA:");
                HashMap cha_rta_callSiteToTargets = new HashMap();
                for (Iterator i=INSTANCE.callSiteToTargets.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    CallSite cs = (CallSite)e.getKey();
                    CallTargets ct = cs.m.getCallTargets();
                    cha_rta_callSiteToTargets.put(cs, ct);
                }
                if (FULL_DUMP)
                    System.out.println(dumpResults(cha_rta_callSiteToTargets));
                System.out.println(computeHistogram(cha_rta_callSiteToTargets));
            }
        }
        public static void calcRTA() {
            Set/*jq_Type*/ s = Bootstrap.PrimordialClassLoader.loader.getAllTypes();
            for (;;) {
                int size = s.size();
                Iterator i = s.iterator();
                while (i.hasNext()) {
                    jq_Type t = (jq_Type)i.next();
                    t.load(); t.verify(); t.prepare();
                }
                s = Bootstrap.PrimordialClassLoader.loader.getAllTypes();
                //if (s.size() == size)
		    break;
            }
            System.out.println("Number of RTA classes: "+s.size());
            int nMethods = 0;
            Iterator i = s.iterator();
            HashSet methods = new HashSet();
            while (i.hasNext()) {
                jq_Type t = (jq_Type)i.next();
                if (t instanceof jq_Class) {
                    jq_Class k = (jq_Class)t;
		    k.load();
                    jq_Method[] ms = k.getDeclaredInstanceMethods();
                    for (int j=0; j<ms.length; ++j) {
                        methods.add(ms[j]);
                    }
                    ms = k.getDeclaredStaticMethods();
                    for (int j=0; j<ms.length; ++j) {
                        methods.add(ms[j]);
                    }
                }
            }
            System.out.println("Number of RTA methods: "+methods.size());
            int nInvokes = 0, nTargets = 0; long nBytecodes = 0;
            Iterator k = methods.iterator();
            while (k.hasNext()) {
                jq_Method m = (jq_Method)k.next();
                if (m.getBytecode() == null) continue;
                nBytecodes += m.getBytecode().length;
                InvokeCounter ic = new InvokeCounter(m);
                ic.forwardTraversal();
                nInvokes += ic.invokeCount; nTargets += ic.targetCount;
            }
            System.out.println("Number of RTA invocations: "+nInvokes);
            System.out.println("Number of RTA call graph edges: "+nTargets);
            System.out.println("Number of RTA bytecodes: "+nBytecodes);
        }
        
        static class InvokeCounter extends Compil3r.BytecodeAnalysis.BytecodeVisitor {
            int invokeCount = 0; int targetCount = 0;
            InvokeCounter(jq_Method m) { super(m); }
            void visitInvoke(byte op, jq_Method f) {
                invokeCount++;
                CallTargets ct = CallTargets.getTargets(method.getDeclaringClass(), f, op, true);
                targetCount += ct.size();
            }
            public void visitIINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
            public void visitLINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
            public void visitFINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
            public void visitDINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
            public void visitAINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
            public void visitVINVOKE(byte op, jq_Method f) {
                visitInvoke(op, f);
            }
        }
        
        public static void doIt_output() {
            INSTANCE.iterate();
            System.out.println("Result of Andersen pointer analysis:");
            System.out.println(dumpResults(INSTANCE.callSiteToTargets));
            System.out.println(INSTANCE.computeStats());
            System.out.println(computeHistogram(INSTANCE.callSiteToTargets));

            if (COMPARE_RTA) {
                System.out.println("Compare to CHA/RTA:");
                calcRTA();
                HashMap cha_rta_callSiteToTargets = new HashMap();
                for (Iterator i=INSTANCE.callSiteToTargets.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    CallSite cs = (CallSite)e.getKey();
                    CallTargets ct = cs.m.getCallTargets();
                    cha_rta_callSiteToTargets.put(cs, ct);
                }
                System.out.println(dumpResults(cha_rta_callSiteToTargets));
                System.out.println(computeHistogram(cha_rta_callSiteToTargets));
            }
        }
    }
    
    public void initializeStatics(boolean addMethodsToVisit) {
        // add initializations for System.in/out/err
        jq_Class fd = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileDescriptor;");
        fd.load();
        ConcreteTypeNode fd_n1 = new ConcreteTypeNode(fd);
        jq_Initializer fd_init = (jq_Initializer)fd.getOrCreateInstanceMethod("<init>", "(I)V");
        jq.Assert(fd_init.isLoaded());
        MethodCall mc_fd_init = new MethodCall(fd_init, null);
        fd_n1.recordPassedParameter(mc_fd_init, 0);
        
        jq_Class fis = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileInputStream;");
        fis.load();
        ConcreteTypeNode fis_n = new ConcreteTypeNode(fis);
        jq_Initializer fis_init = (jq_Initializer)fis.getOrCreateInstanceMethod("<init>", "(Ljava/io/FileDescriptor;)V");
        jq.Assert(fis_init.isLoaded());
        MethodCall mc_fis_init = new MethodCall(fis_init, null);
        fis_n.recordPassedParameter(mc_fis_init, 0);
        fd_n1.recordPassedParameter(mc_fis_init, 1);
        jq_Class bis = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/BufferedInputStream;");
        bis.load();
        ConcreteTypeNode bis_n = new ConcreteTypeNode(bis);
        jq_Initializer bis_init = (jq_Initializer)bis.getOrCreateInstanceMethod("<init>", "(Ljava/io/InputStream;)V");
        jq.Assert(bis_init.isLoaded());
        MethodCall mc_bis_init = new MethodCall(bis_init, null);
        bis_n.recordPassedParameter(mc_bis_init, 0);
        fis_n.recordPassedParameter(mc_bis_init, 1);
        
        jq_Class jls = Bootstrap.PrimordialClassLoader.getJavaLangSystem();
	jls.load();
        jq_StaticField si = jls.getOrCreateStaticField("in", "Ljava/io/InputStream;");
        jq.Assert(si.isLoaded());
        GlobalNode.GLOBAL.addEdge(si, bis_n, null);
        
        ControlFlowGraph fd_init_cfg = CodeCache.getCode(fd_init);
        MethodSummary fd_init_summary = MethodSummary.getSummary(fd_init_cfg);
        OutsideNode on = fd_init_summary.getParamNode(0);
        addInclusionEdge(on, fd_n1, null);
        ControlFlowGraph fis_init_cfg = CodeCache.getCode(fis_init);
        MethodSummary fis_init_summary = MethodSummary.getSummary(fis_init_cfg);
        on = fis_init_summary.getParamNode(0);
        addInclusionEdge(on, fis_n, null);
        on = fis_init_summary.getParamNode(1);
        addInclusionEdge(on, fd_n1, null);
        ControlFlowGraph bis_init_cfg = CodeCache.getCode(bis_init);
        MethodSummary bis_init_summary = MethodSummary.getSummary(bis_init_cfg);
        on = bis_init_summary.getParamNode(0);
        addInclusionEdge(on, bis_n, null);
        on = bis_init_summary.getParamNode(1);
        addInclusionEdge(on, fis_n, null);
        
        ConcreteTypeNode fd_n2 = new ConcreteTypeNode(fd);
        fd_n2.recordPassedParameter(mc_fd_init, 0);
        jq_Class fos = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/FileOutputStream;");
        fos.load();
        ConcreteTypeNode fos_n1 = new ConcreteTypeNode(fos);
        jq_Initializer fos_init = (jq_Initializer)fos.getOrCreateInstanceMethod("<init>", "(Ljava/io/FileDescriptor;)V");
        jq.Assert(fos_init.isLoaded());
        MethodCall mc_fos_init = new MethodCall(fos_init, null);
        fos_n1.recordPassedParameter(mc_fos_init, 0);
        fd_n2.recordPassedParameter(mc_fos_init, 1);
        jq_Class bos = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/BufferedOutputStream;");
        bos.load();
        ConcreteTypeNode bos_n1 = new ConcreteTypeNode(bos);
        jq_Initializer bos_init = (jq_Initializer)bos.getOrCreateInstanceMethod("<init>", "(Ljava/io/OutputStream;I)V");
        jq.Assert(bos_init.isLoaded());
        MethodCall mc_bos_init = new MethodCall(bos_init, null);
        bos_n1.recordPassedParameter(mc_bos_init, 0);
        fos_n1.recordPassedParameter(mc_bos_init, 1);
        
        jq_Class ps = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/PrintStream;");
        ps.load();
        ConcreteTypeNode ps_n1 = new ConcreteTypeNode(ps);
        jq_Initializer ps_init = (jq_Initializer)ps.getOrCreateInstanceMethod("<init>", "(Ljava/io/OutputStream;Z)V");
        jq.Assert(ps_init.isLoaded());
        MethodCall mc_ps_init = new MethodCall(ps_init, null);
        ps_n1.recordPassedParameter(mc_ps_init, 0);
        bos_n1.recordPassedParameter(mc_ps_init, 1);
        
        jq_StaticField so = jls.getOrCreateStaticField("out", "Ljava/io/PrintStream;");
        jq.Assert(so.isLoaded());
        GlobalNode.GLOBAL.addEdge(so, ps_n1, null);
        
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
        jq.Assert(so.isLoaded());
        GlobalNode.GLOBAL.addEdge(so, ps_n2, null);
        
        on = fd_init_summary.getParamNode(0);
        addInclusionEdge(on, fd_n2, null);
        addInclusionEdge(on, fd_n3, null);
        ControlFlowGraph fos_init_cfg = CodeCache.getCode(fos_init);
        MethodSummary fos_init_summary = MethodSummary.getSummary(fos_init_cfg);
        on = fos_init_summary.getParamNode(0);
        addInclusionEdge(on, fos_n1, null);
        addInclusionEdge(on, fos_n2, null);
        on = fos_init_summary.getParamNode(1);
        addInclusionEdge(on, fd_n2, null);
        addInclusionEdge(on, fd_n3, null);
        ControlFlowGraph bos_init_cfg = CodeCache.getCode(bos_init);
        MethodSummary bos_init_summary = MethodSummary.getSummary(bos_init_cfg);
        on = bos_init_summary.getParamNode(0);
        addInclusionEdge(on, bos_n1, null);
        addInclusionEdge(on, bos_n2, null);
        on = bos_init_summary.getParamNode(1);
        addInclusionEdge(on, fos_n1, null);
        addInclusionEdge(on, fos_n2, null);
        ControlFlowGraph ps_init_cfg = CodeCache.getCode(ps_init);
        MethodSummary ps_init_summary = MethodSummary.getSummary(ps_init_cfg);
        on = ps_init_summary.getParamNode(0);
        addInclusionEdge(on, ps_n1, null);
        addInclusionEdge(on, ps_n2, null);
        on = ps_init_summary.getParamNode(1);
        addInclusionEdge(on, bos_n1, null);
        addInclusionEdge(on, bos_n2, null);
        
        jq_Class nt = (jq_Class)Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("LScheduler/jq_NativeThread;");
        nt.load();
        ConcreteTypeNode nt_n1 = new ConcreteTypeNode(nt);
        jq.Assert(Scheduler.jq_NativeThread._nativeThreadEntry.isLoaded());
        MethodCall mc_nte = new MethodCall(Scheduler.jq_NativeThread._nativeThreadEntry, null);
        nt_n1.recordPassedParameter(mc_nte, 0);
        ControlFlowGraph nte_cfg = CodeCache.getCode(Scheduler.jq_NativeThread._nativeThreadEntry);
        MethodSummary nte_summary = MethodSummary.getSummary(nte_cfg);
        on = nte_summary.getParamNode(0);
        addInclusionEdge(on, nt_n1, null);
        jq.Assert(Scheduler.jq_NativeThread._threadSwitch.isLoaded());
        MethodCall mc_ts = new MethodCall(Scheduler.jq_NativeThread._threadSwitch, null);
        nt_n1.recordPassedParameter(mc_ts, 0);
        ControlFlowGraph ts_cfg = CodeCache.getCode(Scheduler.jq_NativeThread._threadSwitch);
        MethodSummary ts_summary = MethodSummary.getSummary(ts_cfg);
        on = ts_summary.getParamNode(0);
        addInclusionEdge(on, nt_n1, null);
        
        if (addMethodsToVisit) {
            methodsToVisit.add(fd_init_cfg);
            methodsToVisit.add(fis_init_cfg);
            methodsToVisit.add(bis_init_cfg);
            methodsToVisit.add(fos_init_cfg);
            methodsToVisit.add(bos_init_cfg);
            methodsToVisit.add(ps_init_cfg);
            methodsToVisit.add(nte_cfg);
            methodsToVisit.add(ts_cfg);
        }
    }
    
    /** Cache: Maps a node to its set of corresponding concrete nodes. */
    final HashMap nodeToConcreteNodes;
    
    /** Maps a node to its set of outgoing inclusion edges. */
    final HashMap nodeToInclusionEdges;
    
    /** Maps an inclusion edge to the quad that caused the edge.
     *  Only used if TRACK_SOURCE_QUADS is set. */
    final HashMap edgesToQuads;
    
    /** Set of all MethodSummary's that we care about. */
    final LinkedHashSet rootSet;
    final LinkedHashSet methodsToVisit;
    
    /** Maps a call site to its set of targets. */
    final HashMap callSiteToTargets;
    
    /** The set of method call->targets that have already been linked. */
    final HashSet linkedTargets;
    
    /** Records if the cache for the node is current, and whether it has changed
     *  since the last iteration.  Only used if REUSE_CACHES is true. */
    final HashMap cacheIsCurrent;

    /** Records edges that have not yet been propagated.
     *  Only used if TRACK_CHANGES is true. */
    final HashSet unpropagatedEdges;
    
    /** Records nodes that have been collapsed, and which predecessors have
     *  seen the collapse.  Only used if TRACK_CHANGES is true. */
    final HashMap collapsedNodes;

    /** Records what fields have changed.  Only used if TRACK_CHANGED_FIELDS is true. */
    HashSet oldChangedFields;
    HashSet newChangedFields;
    HashSet changedFields_Methods;
    
    /** Change flag, for iterations. */
    boolean change;
    
    /** Creates new AndersenPointerAnalysis */
    public AndersenPointerAnalysis(boolean addDefaults) {
        nodeToConcreteNodes = new HashMap();
        nodeToInclusionEdges = new HashMap();
        rootSet = new LinkedHashSet();
        methodsToVisit = new LinkedHashSet();
        callSiteToTargets = new HashMap();
        linkedTargets = new HashSet();
        if (REUSE_CACHES)
            cacheIsCurrent = new HashMap();
        else
            cacheIsCurrent = null;
        if (TRACK_CHANGES) {
            unpropagatedEdges = new HashSet();
            collapsedNodes = new HashMap();
        } else {
            unpropagatedEdges = null;
            collapsedNodes = null;
        }
        if (TRACK_SOURCE_QUADS)
            edgesToQuads = new HashMap();
        else
            edgesToQuads = null;
        if (TRACK_CHANGED_FIELDS) {
            /*oldChangedFields =*/ newChangedFields = new HashSet();
            changedFields_Methods = new HashSet();
        }
        this.initializeStatics(addDefaults);
    }

    public static AndersenPointerAnalysis INSTANCE = new AndersenPointerAnalysis(true);
    
    public static final String lineSep = System.getProperty("line.separator");
    
    public static final int HISTOGRAM_SIZE = 100;
    
    public String computeStats() {
        StringBuffer sb = new StringBuffer();
        HashSet classes = new HashSet();
        long bytecodes = 0;
        for (Iterator i=methodsToVisit.iterator(); i.hasNext(); ) {
            ControlFlowGraph ms = (ControlFlowGraph)i.next();
            bytecodes += ms.getMethod().getBytecode().length;
            jq_Class c = ms.getMethod().getDeclaringClass();
            while (c != null) {
                classes.add(c);
                c.load();
                c = c.getSuperclass();
            }
        }
        sb.append(" Classes: ");
        sb.append(classes.size());
        sb.append(" Methods: ");
        sb.append(methodsToVisit.size());
        sb.append(" Calls: ");
        sb.append(callSiteToTargets.size());
        sb.append(" Bytecodes ");
        sb.append(bytecodes);
        sb.append(" Iteration ");
        sb.append(count);
        sb.append(lineSep);
        return sb.toString();
    }
    public static String computeHistogram(Map m) {
        StringBuffer sb = new StringBuffer();
        int[] histogram = new int[HISTOGRAM_SIZE];
        long total = 0;
        for (Iterator i=m.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            CallSite cs = (CallSite)e.getKey();
            Set s = (Set)e.getValue();
            int x = s.size();
            if (x >= HISTOGRAM_SIZE) x = HISTOGRAM_SIZE-1;
            histogram[x]++;
            total += x;
        }
        sb.append(" Total # of call graph edges: ");
        sb.append(total);
        sb.append('/');
        sb.append(total+histogram[0]);
        sb.append(lineSep);
        for (int i=0; i<HISTOGRAM_SIZE; ++i) {
            if (histogram[i] > 0) {
                if (i == HISTOGRAM_SIZE-1) sb.append(">=");
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
    
    public static String dumpResults(Map m) {
        TreeSet ts = new TreeSet(
            new Comparator() {
                public int compare(Object o1, Object o2) {
                    Map.Entry e1 = (Map.Entry)o1;
                    CallSite cs1 = (CallSite)e1.getKey();
                    Set s1 = (Set)e1.getValue();
                    Map.Entry e2 = (Map.Entry)o2;
                    CallSite cs2 = (CallSite)e2.getKey();
                    Set s2 = (Set)e2.getValue();
                    int s1s = s1.size(); int s2s = s2.size();
                    if (s1s < s2s) return 1;
                    else if (s1s > s2s) return -1;
                    else return cs1.toString().compareTo(cs2.toString());
                }
            });
        ts.addAll(m.entrySet());
        StringBuffer sb = new StringBuffer();
        for (Iterator i=ts.iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry)i.next();
            CallSite cs = (CallSite)e.getKey();
            Set s = (Set)e.getValue();
            sb.append(cs.toString());
            sb.append(": {");
            int x = s.size();
            sb.append(x);
            sb.append("} ");
            sb.append(s.toString());
            sb.append(lineSep);
        }
        return sb.toString();
    }

    public Map getCallGraph() { return callSiteToTargets; }
    
    int count;
    
    public void iterate() {
        methodsToVisit.addAll(rootSet);
        count = 1;
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
            if (REUSE_CACHES)
                cacheIsCurrent.clear();
            else
                nodeToConcreteNodes.clear();
            if (TRACK_CHANGED_FIELDS) {
                oldChangedFields = newChangedFields;
                System.err.println(oldChangedFields.size()+" changed fields");
                newChangedFields = new HashSet();
            }
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
                        if ((TRACE_CHANGE && !this.change) || TRACE) {
                            out.println("Changed! New clinit method: "+clinit);
                        }
                        this.change = true;
                    }
                }
            }
            // o = n.f
            if (o instanceof LinkedHashSet) {
                addGlobalEdges((LinkedHashSet)o, f);
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
                if (TRACK_CHANGED_FIELDS) {
                    if (!changedFields_Methods.contains(ms.getMethod())) {
                        newChangedFields.add(f);
                    }
                }
                Object o = e.getValue();
                if (TRACE) out.println("Visiting edge: "+n+((f==null)?"[]":("."+f.getName()))+" = "+o);
                // n.f = o
                if (o instanceof LinkedHashSet) {
                    addEdgesFromConcreteNodes(n, f, (LinkedHashSet)o);
                } else {
                    addEdgesFromConcreteNodes(n, f, (Node)o);
                }
            }
	    if (HANDLE_ESCAPE) {
		if (n instanceof OutsideNode && n.getEscapes()) {
		    Set s = getConcreteNodes(n);
		    if (TRACE) out.println("Escaping node "+n+" corresponds to concrete nodes "+s);
		    for (Iterator j=s.iterator(); j.hasNext(); ) {
			Node n2 = (Node)j.next();
			if (!n2.getEscapes()) {
			    n2.setEscapes();
			    if ((TRACE_CHANGE && !this.change) || TRACE) {
				out.println("Changed! Concrete node "+n2+" escapes");
			    }
			    this.change = true;
			}
		    }
		}
	    }
            for (Iterator j=n.getAccessPathEdges().iterator(); j.hasNext(); ) {
                Map.Entry e = (Map.Entry)j.next();
                jq_Field f = (jq_Field)e.getKey();
                if (TRACK_CHANGED_FIELDS) {
                    if (!changedFields_Methods.contains(ms.getMethod())) {
                        changedFields_Methods.add(ms.getMethod());
                    } else if (oldChangedFields != null && !oldChangedFields.contains(f) && !newChangedFields.contains(f)) continue;
                }
                Object o = e.getValue();
                if (TRACE) out.println("Visiting edge: "+o+" = "+n+((f==null)?"[]":("."+f.getName())));
                // o = n.f
                if (o instanceof LinkedHashSet) {
                    addInclusionEdgesToConcreteNodes((LinkedHashSet)o, n, f);
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
            LinkedHashSet definite_targets = new LinkedHashSet();
            //jq.Assert(!callSiteToTargets.containsKey(cs));
            callSiteToTargets.put(cs, definite_targets);
            if (ct.size() == 1 && ct.isComplete()) {
                // call can be statically resolved to a single target.
                if (TRACE) out.println("Call is statically resolved to a single target.");
                definite_targets.add(ct.iterator().next());
            } else {
                // use the type information about the receiver object to find targets.
                LinkedHashSet set = new LinkedHashSet();
                PassedParameter pp = new PassedParameter(mc, 0);
                ms.getNodesThatCall(pp, set);
                if (TRACE) out.println("Possible nodes for receiver object: "+set);
                for (Iterator j=set.iterator(); j.hasNext(); ) {
                    Node base = (Node)j.next();
                    if (TRACE) out.println("Checking base node: "+base);
                    Set s_cn = getConcreteNodes(base);
                    Set targets = mc.getCallTargets(s_cn);
                    definite_targets.addAll(targets);
                }
            }
            if (TRACE) out.println("Set of definite targets of "+mc+": "+definite_targets);
            for (Iterator j=definite_targets.iterator(); j.hasNext(); ) {
                jq_Method callee = (jq_Method)j.next();
                callee.getDeclaringClass().load();
                if (callee.getBytecode() == null) {
                    CallSite cs2 = new CallSite(null, mc);
                    if (linkedTargets.contains(cs2)) continue;
                    linkedTargets.add(cs2);
                    if ((TRACE_CHANGE && !this.change) || TRACE) {
                        out.println("Changed! New target for "+mc+": "+callee+" (unanalyzable)");
                    }
                    this.change = true;
                    if (TRACE) out.println(callee+" is a native method, skipping analysis...");
                    addParameterAndReturnMappings_native(ms, mc);
                    continue;
                }
                ControlFlowGraph callee_cfg = CodeCache.getCode(callee);
                MethodSummary callee_summary = MethodSummary.getSummary(callee_cfg);
                CallSite cs2 = new CallSite(callee_summary, mc);
                if (linkedTargets.contains(cs2)) continue;
                linkedTargets.add(cs2);
                if ((TRACE_CHANGE && !this.change) || TRACE) {
                    out.println("Changed! New target for "+mc+": "+callee_summary.getMethod());
                }
                this.change = true;
                addParameterAndReturnMappings(ms, mc, callee_summary);
                methodsToVisit.add(callee_cfg);
            }
        }
    }

    void addParameterAndReturnMappings_native(MethodSummary caller, MethodCall mc) {
        if (TRACE) out.println("Adding parameter and return mappings for "+mc+" from "+caller.getMethod()+" to an unanalyzable method.");
        ParamListOperand plo = Invoke.getParamList(mc.q);
        for (int i=0; i<plo.length(); ++i) {
            jq_Type t = plo.get(i).getType();
            if (!(t instanceof jq_Reference)) continue;
            // parameters passed into native methods escape.
            PassedParameter pp = new PassedParameter(mc, i);
            LinkedHashSet s = new LinkedHashSet();
            caller.getNodesThatCall(pp, s);
            for (Iterator j=s.iterator(); j.hasNext(); ) {
                Node n = (Node)j.next();
                if (!n.getEscapes()) {
                    n.setEscapes();
                    if ((TRACE_CHANGE && !this.change) || TRACE) {
                        out.println("Changed! Node "+n+" escapes");
                    }
                    this.change = true;
                }
                // TODO: add edges to escape.
            }
        }
        ReturnValueNode rvn = (ReturnValueNode)caller.nodes.get(new ReturnValueNode(mc));
        if (rvn != null) {
            UnknownTypeNode utn = UnknownTypeNode.get((jq_Reference)mc.m.getReturnType());
            if (TRACE) out.println("Adding return mapping "+rvn+" to "+utn);
            OutsideNode on = rvn;
            while (on.skip != null) on = on.skip;
            addInclusionEdge(on, utn, mc.q);
        }
        ThrownExceptionNode ten = (ThrownExceptionNode)caller.nodes.get(new ThrownExceptionNode(mc));
        if (ten != null) {
            UnknownTypeNode utn = UnknownTypeNode.get((jq_Reference)Bootstrap.PrimordialClassLoader.getJavaLangObject());
            if (TRACE) out.println("Adding thrown mapping "+ten+" to "+utn);
            OutsideNode on = ten;
            while (on.skip != null) on = on.skip;
            addInclusionEdge(on, utn, mc.q);
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
            LinkedHashSet s = new LinkedHashSet();
            caller.getNodesThatCall(pp, s);
            //s.add(pn);
            if (TRACE) out.println("Adding parameter mapping "+pn+" to set "+s);
            OutsideNode on = pn;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s, mc.q);
        }
        ReturnValueNode rvn = (ReturnValueNode)caller.nodes.get(new ReturnValueNode(mc));
        if (rvn != null) {
            LinkedHashSet s = (LinkedHashSet)callee.returned.clone();
            //s.add(rvn);
            if (TRACE) out.println("Adding return mapping "+rvn+" to set "+s);
            OutsideNode on = rvn;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s, mc.q);
        }
        ThrownExceptionNode ten = (ThrownExceptionNode)caller.nodes.get(new ThrownExceptionNode(mc));
        if (ten != null) {
            LinkedHashSet s = (LinkedHashSet)callee.thrown.clone();
            //s.add(ten);
            if (TRACE) out.println("Adding thrown mapping "+ten+" to set "+s);
            OutsideNode on = ten;
            while (on.skip != null) on = on.skip;
            addInclusionEdges(on, s, mc.q);
        }
    }
    
    boolean addInclusionEdges(OutsideNode n, Set s, Quad q) {
        if (VerifyAssertions) jq.Assert(n.skip == null);
        Set s2 = (Set)nodeToInclusionEdges.get(n);
        if (s2 == null) {
            nodeToInclusionEdges.put(n, s);
            if ((TRACE_CHANGE && !this.change) || TRACE) {
                out.println("Changed! New set of inclusion edges for node "+n);
            }
            this.change = true;
            if (TRACK_CHANGES) {
                // we need to mark these edges so that they will be propagated
                // regardless of whether or not the target set has changed.
                if (cacheContains(n)) {
                    for (Iterator i=s.iterator(); i.hasNext(); ) {
                        Object o = i.next();
                        if (o instanceof OutsideNode) {
                            if (TRACE) out.println("Adding "+n+"->"+o+" as an unpropagated edge...");
                            recordUnpropagatedEdge(n, (OutsideNode)o);
                        }
                    }
                }
            }
            if (TRACK_SOURCE_QUADS) {
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    Object o = i.next();
                    edgesToQuads.put(Default.pair(n, o), q);
                }
            }
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
                    if ((TRACE_CHANGE && !this.change) || TRACE) {
                        out.println("Changed! New inclusion edge for node "+n+": "+o);
                    }
                    this.change = true;
                    if (TRACK_CHANGES) {
                        if (o instanceof OutsideNode) {
                            // we need to mark this edge so that it will be propagated
                            // regardless of whether or not the target set has changed.
                            if (cacheContains(n)) {
                                if (TRACE) out.println("Adding "+n+"->"+o+" as an unpropagated edge...");
                                recordUnpropagatedEdge(n, (OutsideNode)o);
                            }
                        }
                    }
                    if (TRACK_SOURCE_QUADS) {
                        edgesToQuads.put(Default.pair(n, o), q);
                    }
                }
            }
            return false;
        }
    }
    
    void addInclusionEdge(OutsideNode n, Node s, Quad q) {
        if (VerifyAssertions) jq.Assert(n.skip == null);
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
            nodeToInclusionEdges.put(n, s2 = new LinkedHashSet()); s2.add(s);
            if ((TRACE_CHANGE && !this.change) || TRACE) {
                out.println("Changed! New set of inclusion edges for node "+n);
            }
            this.change = true;
            if (TRACK_CHANGES) {
                if (s instanceof OutsideNode) {
                    // we need to mark this edge so that it will be propagated
                    // regardless of whether or not the target set has changed.
                    if (cacheContains(n)) {
                        if (TRACE) out.println("Adding "+n+"->"+s+" as an unpropagated edge...");
                        recordUnpropagatedEdge(n, (OutsideNode)s);
                    }
                }
            }
            if (TRACK_SOURCE_QUADS) {
                edgesToQuads.put(Default.pair(n, s), q);
            }
        } else if (s2.add(s)) {
            if ((TRACE_CHANGE && !this.change) || TRACE) {
                out.println("Changed! New inclusion edge for node "+n+": "+s);
            }
            this.change = true;
            if (TRACK_CHANGES) {
                if (s instanceof OutsideNode) {
                    // we need to mark this edge so that it will be propagated
                    // regardless of whether or not the target set has changed.
                    if (cacheContains(n)) {
                        if (TRACE) out.println("Adding "+n+"->"+s+" as an unpropagated edge...");
                        recordUnpropagatedEdge(n, (OutsideNode)s);
                    }
                }
            }
            if (TRACK_SOURCE_QUADS) {
                edgesToQuads.put(Default.pair(n, s), q);
            }
        }
    }
    
    Set getInclusionEdges(Node n) { return (Set)nodeToInclusionEdges.get(n); }
    
    LinkedHashSet getFromCache(OutsideNode n) {
        if (USE_SOFT_REFERENCES) {
            java.lang.ref.SoftReference r = (java.lang.ref.SoftReference)nodeToConcreteNodes.get(n);
            return r != null ? (LinkedHashSet)r.get() : null;
        } else {
            return (LinkedHashSet)nodeToConcreteNodes.get(n);
        }
    }
    
    void addToCache(OutsideNode n, LinkedHashSet s) {
        if (USE_SOFT_REFERENCES) {
            nodeToConcreteNodes.put(n, new java.lang.ref.SoftReference(s));
        } else {
            nodeToConcreteNodes.put(n, s);
        }
    }
    
    boolean cacheContains(OutsideNode n) {
        // this needs to return whether or not the cache EVER contained the given node.
        if (false) { // if (USE_SOFT_REFERENCES) {
            java.lang.ref.SoftReference r = (java.lang.ref.SoftReference)nodeToConcreteNodes.get(n);
            return r == null || r.get() == null;
        } else {
            return nodeToConcreteNodes.containsKey(n);
        }
    }
    
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
    void addEdgesFromConcreteNodes(Node from, jq_Field f, LinkedHashSet to) {
        Set s = getConcreteNodes(from);
        if (TRACE) out.println("Node "+from+" corresponds to concrete nodes "+s);
        for (Iterator i=s.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if (checkInvalidFieldAccess(n, f)) continue;
            if (TRACK_SOURCE_QUADS) {
                for (Iterator j=to.iterator(); j.hasNext(); ) {
                    Node o = (Node)j.next();
                    Quad q = TRACK_SOURCE_QUADS ? from.getSourceQuad(f, o) : (Quad)null;
                    if (n.addEdge(f, o, q)) {
                        if ((TRACE_CHANGE && !this.change) || TRACE) {
                            out.println("Changed! New edge for concrete node "+n+"."+f+": "+o);
                        }
                        if (TRACK_CHANGED_FIELDS) newChangedFields.add(f);
                        this.change = true;
                    }
                }
            } else {
                if (n.addEdges(f, to, null)) {
                    if ((TRACE_CHANGE && !this.change) || TRACE) {
                        out.println("Changed! New edges for concrete node "+n+"."+f+": "+to);
                    }
                    if (TRACK_CHANGED_FIELDS) newChangedFields.add(f);
                    this.change = true;
                }
            }
        }
    }
    
    // from.f = to
    void addEdgesFromConcreteNodes(Node from, jq_Field f, Node to) {
        Set s = getConcreteNodes(from);
        if (TRACE) out.println("Node "+from+" corresponds to concrete nodes "+s);
        Quad q = TRACK_SOURCE_QUADS ? from.getSourceQuad(f, to) : (Quad)null;
        for (Iterator i=s.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            if (checkInvalidFieldAccess(n, f)) continue;
            if (n.addEdge(f, to, q)) {
                if ((TRACE_CHANGE && !this.change) || TRACE) {
                    out.println("Changed! New edge for concrete node "+n+"."+f+": "+to);
                }
                if (TRACK_CHANGED_FIELDS) newChangedFields.add(f);
                this.change = true;
            }
        }
    }
    
    // from = global.f
    void addGlobalEdges(OutsideNode from, jq_Field f) {
        LinkedHashSet result = new LinkedHashSet();
        GlobalNode.GLOBAL.getEdges(f, result);
        while (from.skip != null) from = from.skip;
        addInclusionEdges(from, result, null);
    }
    
    // from = global.f
    void addGlobalEdges(LinkedHashSet from, jq_Field f) {
        LinkedHashSet result = new LinkedHashSet();
        GlobalNode.GLOBAL.getEdges(f, result);
        for (Iterator j=from.iterator(); j.hasNext(); ) {
            OutsideNode n2 = (OutsideNode)j.next();
            while (n2.skip != null) n2 = n2.skip;
            if (addInclusionEdges(n2, result, null)) result = (LinkedHashSet)result.clone();
        }
    }
    
    // from = base.f
    void addInclusionEdgesToConcreteNodes(LinkedHashSet from, Node base, jq_Field f) {
        Set s = getConcreteNodes(base);
        if (TRACE) out.println("Node "+base+" corresponds to concrete nodes "+s);
        LinkedHashSet result = new LinkedHashSet();
        for (Iterator j=s.iterator(); j.hasNext(); ) {
            Node n2 = (Node)j.next();
            n2.getEdges(f, result);
        }
        if (TRACE) out.println("Edges from "+base+"."+f+" : "+result);
        for (Iterator j=from.iterator(); j.hasNext(); ) {
            OutsideNode n2 = (OutsideNode)j.next();
            FieldNode fn = (FieldNode)n2;
            Quad q;
            if (fn.quads.isEmpty()) q = null;
            else q = (Quad)fn.quads.iterator().next();
            while (n2.skip != null) n2 = n2.skip;
            if (addInclusionEdges(n2, result, q)) result = (LinkedHashSet)result.clone();
        }
    }
    
    // from = base.f
    void addInclusionEdgesToConcreteNodes(OutsideNode from, Node base, jq_Field f) {
        Set s = getConcreteNodes(base);
        if (TRACE) out.println("Node "+base+" corresponds to concrete nodes "+s);
        LinkedHashSet result = new LinkedHashSet();
        for (Iterator j=s.iterator(); j.hasNext(); ) {
            Node n2 = (Node)j.next();
            n2.getEdges(f, result);
        }
        if (TRACE) out.println("Edges from "+base+"."+f+" : "+result);
        FieldNode fn = (FieldNode)from;
        Quad q;
        if (fn.quads.isEmpty()) q = null;
        else q = (Quad)fn.quads.iterator().next();
        while (from.skip != null) from = from.skip;
        addInclusionEdges(from, result, q);
    }
    
    Set getConcreteNodes(Node from) {
        if (from instanceof OutsideNode) 
            return getConcreteNodes((OutsideNode)from, null);
        else
            return Collections.singleton(from);
    }
    
    boolean temp_change;
    
    LinkedHashSet getConcreteNodes(OutsideNode from, Path p) {
        while (from.skip != null) {
            from = from.skip;
        }
        if (from.visited) {
            Path p2 = p;
            if (TRACE_CYCLES) out.println("cycle detected! node="+from+" path="+p);
            LinkedHashSet s = (LinkedHashSet)nodeToInclusionEdges.get(from);
            if (VerifyAssertions) jq.Assert(s != null);
            OutsideNode last = from;
            for (;; p = p.cdr()) {
                OutsideNode n = p.car();
                if (TRACK_CHANGES) markCollapsedNode(n);
                if (n == from) break;
                if (TRACE) out.println("next in path: "+n+", merging into: "+from);
                if (VerifyAssertions) jq.Assert(n.skip == null);
                n.skip = from;
                LinkedHashSet s2 = (LinkedHashSet)nodeToInclusionEdges.get(n);
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
        LinkedHashSet result = getFromCache(from);
        boolean brand_new = false;
        if (REUSE_CACHES) {
            if (result == null) {
                if (TRACE) out.println("No cache for "+from+" yet, creating.");
                addToCache(from, result = new LinkedHashSet());
                brand_new = true;
            } else {
                Object b = cacheIsCurrent.get(from);
                if (b != null) {
                    if (TRACE) out.println("Cache for "+from+" is current: "+result+" changed since last iteration: "+b);
                    if (TRACK_CHANGES) this.temp_change = ((Boolean)b).booleanValue();
                    return result;
                } else {
                    if (TRACE) out.println("Cache for "+from+" "+result+" is not current, updating.");
                }
            }
        } else {
            if (result != null) {
                if (TRACE) out.println("Using cached result for "+from+".");
                return result;
            }
            addToCache(from, result = new LinkedHashSet());
        }
        LinkedHashSet s = (LinkedHashSet)nodeToInclusionEdges.get(from);
        if (s == null) {
            if (TRACE) out.println("No inclusion edges for "+from+", returning.");
            if (TRACK_CHANGES) {
                cacheIsCurrent.put(from, Boolean.FALSE);
                this.temp_change = false;
            } else if (REUSE_CACHES) {
                cacheIsCurrent.put(from, from);
            }
            return result;
        }
        p = new Path(from, p);
        boolean local_change = false;
        for (;;) {
            Iterator i = s.iterator();
            for (;;) {
                if (!i.hasNext()) {
                    if (TRACE) out.println("Finishing exploring "+from+", change in cache="+local_change+", cache="+result);
                    if (REUSE_CACHES) {
                        if (TRACK_CHANGES) {
                            cacheIsCurrent.put(from, local_change?Boolean.TRUE:Boolean.FALSE);
                            this.temp_change = local_change;
                        } else {
                            cacheIsCurrent.put(from, from);
                        }
                    }
                    return result;
                }
                Node to = (Node)i.next();
                if (to instanceof OutsideNode) {
                    if (TRACE) out.println("Visiting inclusion edge "+from+" --> "+to+"...");
                    from.visited = true;
                    LinkedHashSet s2 = getConcreteNodes((OutsideNode)to, p);
                    from.visited = false;
                    if (from.skip != null) {
                        if (TRACE) out.println("Node "+from+" is skipped...");
                        return null;
                    }
                    if (s2 == null) {
                        if (TRACE) out.println("Nodes were merged into "+from);
                        if (TRACE) out.println("redoing iteration on "+s);
                        if (TRACK_CHANGES) brand_new = true; // we have new children, so always union them.
                        if (VerifyAssertions) jq.Assert(nodeToInclusionEdges.get(from) == s);
                        break;
                    } else {
                        if (TRACK_CHANGES) {
                            boolean b = removeUnpropagatedEdge(from, (OutsideNode)to);
                            if (!brand_new && !b && !this.temp_change) {
                                if (TRACE) out.println("No change in cache of target "+to+", skipping union operation");
                                if (VerifyAssertions) jq.Assert(result.containsAll(s2), from+" result "+result+" should contain all of "+to+" result "+s2);
                                continue;
                            }
                        }
                        if (result.addAll(s2)) {
                            if (TRACE) out.println("Unioning cache of target "+to+" changed our cache");
                            local_change = true;
                        }
                    }
                } else {
                    if (result.add(to)) {
                        if (TRACE) out.println("Adding concrete node "+to+" changed our cache");
                        local_change = true;
                    }
                }
            }
        }
    }
    
    void recordUnpropagatedEdge(OutsideNode from, OutsideNode to) {
        unpropagatedEdges.add(Default.pair(from, to));
    }
    boolean removeUnpropagatedEdge(OutsideNode from, OutsideNode to) {
        if (unpropagatedEdges.remove(getPair(from, to))) return true;
        HashSet s = (HashSet)collapsedNodes.get(to);
        if (s == null) return false;
        if (s.contains(from)) return false;
        s.add(from);
        return true;
    }
    void markCollapsedNode(OutsideNode n) {
        HashSet s = (HashSet)collapsedNodes.get(n);
        if (s == null) collapsedNodes.put(n, s = new HashSet());
        else s.clear();
    }
    
    final Default.PairList my_pair_list = new Default.PairList(null, null);
    public Default.PairList getPair(Object left, Object right) {
        my_pair_list.left = left; my_pair_list.right = right; return my_pair_list;
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

// CSPAResults.java, created Aug 7, 2003 12:34:24 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_LineNumberBC;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.FieldNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.PassedParameter;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ReturnValueNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ThrownExceptionNode;
import Compil3r.Analysis.IPA.CSPA.ThreadRootMap;
import Compil3r.Analysis.IPA.ProgramLocation.BCProgramLocation;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.BytecodeAnalysis.Bytecodes;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.LoadedCallGraph;
import Compil3r.Quad.Operator;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;
import Main.HostedVM;
import Util.Assert;
import Util.Strings;
import Util.Collections.GenericInvertibleMultiMap;
import Util.Collections.GenericMultiMap;
import Util.Collections.IndexMap;
import Util.Collections.InvertibleMultiMap;
import Util.Collections.MultiMap;
import Util.Collections.SortedArraySet;
import Util.Collections.UnmodifiableIterator;
import Util.Graphs.PathNumbering;
import Util.Graphs.SCComponent;
import Util.Graphs.Traversals;
import Util.Graphs.PathNumbering.Path;
import Util.IO.ByteSequence;

/**
 * Records results for context-sensitive pointer analysis.  The results can
 * be saved and reloaded.  This class also provides methods to query the results.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class CSPAResults {

    /** Call graph. */
    CallGraph cg;

    /** Path numbering for call graph. */
    PathNumbering pn;

    /** BDD factory object, to perform BDD operations. */
    BDDFactory bdd;

    /** Map between variables and indices in the V1o domain. */
    IndexMap variableIndexMap;
    /** Map between heap objects and indices in the H1o domain. */
    IndexMap heapobjIndexMap;

    /** BDD domain for context number of variable. */
    public BDDDomain V1c;
    /** BDD domain for variable number. */
    public BDDDomain V1o;
    /** BDD domain for context number of heap object. */
    public BDDDomain H1c;
    /** BDD domain for heap object number. */
    public BDDDomain H1o;
    /** BDD domain for field descriptor. */
    public BDDDomain FD;
    /** Extra BDD domain for context number of variable. */
    public BDDDomain V2c;
    /** Extra BDD domain for variable number. */
    public BDDDomain V2o;
    /** Extra BDD domain for context number of heap object. */
    public BDDDomain H2c;
    /** Extra BDD domain for heap object number. */
    public BDDDomain H2o;
    
    /** Points-to BDD: V1c x V1o x H1c x H1o.
     * This contains the result of the points-to analysis.
     * A relation (V,H) is in the BDD if variable V can point to heap object H.
     */
    BDD pointsTo;

    BDD fieldPt;
    
    /** Points-to BDD: V1o x H1o.
     * Just cached because it is used often.
     */
    BDD ci_pointsTo;

    /** Nodes that are returned from their methods. */
    Collection returned;
    /** Nodes that are thrown from their methods. */
    Collection thrown;
    /** Multi-map between passed parameters and nodes they operate on. */
    InvertibleMultiMap passedParams;

    public TypedBDD getPointsToSet(int var) {
        BDD result = ci_pointsTo.restrict(V1o.ithVar(var));
        return new TypedBDD(result, H1o);
    }

    public TypedBDD getAliasedLocations(int var) {
        BDD a = V1o.ithVar(var);
        BDD heapObjs = ci_pointsTo.restrict(a);
        a.free();
        TypedBDD result = new TypedBDD(ci_pointsTo.relprod(heapObjs, H1o.set()), V1o);
        heapObjs.free();
        return result;
    }
    
    public TypedBDD getAliased(int v1, int v2) {
        BDD a = V1o.ithVar(v1);
        BDD heapObjs1 = ci_pointsTo.restrict(a);
        a.free();
        BDD b = V1o.ithVar(v2);
        BDD heapObjs2 = ci_pointsTo.restrict(b);
        b.free();
        heapObjs1.andWith(heapObjs2);
        TypedBDD result = new TypedBDD(heapObjs1, H1o);
        return result;
    }
    
    public TypedBDD getAllHeapOfType(jq_Reference type) {
        int j=0;
        BDD result = bdd.zero();
        for (Iterator i=heapobjIndexMap.iterator(); i.hasNext(); ++j) {
            Node n = (Node) i.next();
            Assert._assert(getHeapIndex(n) == j);
            if (n != null && n.getDeclaredType() == type)
                result.orWith(H1o.ithVar(j));
        }
        return new TypedBDD(result, H1o);
        /*
        {
            int i = typeIndexMap.get(type);
            BDD a = T2.ithVar(i);
            BDD result = aC.restrict(a);
            a.free();
            return result;
        }
        */
    }
    
    /** Get a context-insensitive version of the points-to information.
     * It achieves this by merging all of the contexts together.  The
     * returned BDD is: V1o x H1o.
     */
    public TypedBDD getContextInsensitivePointsTo() {
        return new TypedBDD(ci_pointsTo.id(), V1o, H1o);
    }

    /** Load call graph from the given file name.
     */
    public void loadCallGraph(String fn) throws IOException {
        cg = new LoadedCallGraph(fn);
        pn = new PathNumbering();
        Map thread_map = new ThreadRootMap(findThreadRuns(cg));
        Number paths = pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator(), thread_map);
        System.out.println("Number of paths in call graph="+paths);
    }

    public boolean findAliasedParameters2(jq_Method m) {
        Collection s = methodToVariables.getValues(m);
        Collection paramNodes = new LinkedList();
        BDD vars = bdd.zero();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            Object o = j.next();
            if (o instanceof ParamNode || o instanceof FieldNode) {
            //if (!(o instanceof ThrownExceptionNode) && !(o instanceof GlobalNode))
                paramNodes.add(o);
                int v1 = getVariableIndex((Node) o);
                vars.orWith(V1o.ithVar(v1));
            }
        }
        BDD mpointsTo = pointsTo.and(vars);
        BDDPairing V1toV2 = bdd.makePair();
        V1toV2.set(new BDDDomain[] {V1c, V1o}, new BDDDomain[] {V2c, V2o});
        BDD v2_mpointsTo = mpointsTo.replace(V1toV2);
        BDD dom = H1c.set();
        dom.andWith(H1o.set());
        BDD result = mpointsTo.relprod(v2_mpointsTo, dom);
        v2_mpointsTo.free();
        dom.free();
        if (false) {
            TypedBDD r = new TypedBDD(result, V1c, V1o, V2c, V2o);
        }
        return !result.isZero();
    }

    public boolean findAliasedParameters(jq_Method m) {
        Collection s = methodToVariables.getValues(m);
        Collection paramNodes = new LinkedList();
        for (Iterator j = s.iterator(); j.hasNext(); ) {
            Object o = j.next();
            if (o instanceof ParamNode || o instanceof FieldNode)
            //if (!(o instanceof ThrownExceptionNode) && !(o instanceof GlobalNode))
                paramNodes.add(o);
        }
        boolean hasAliased = false;
        int n = 1;
        for (Iterator j = paramNodes.iterator(); j.hasNext(); ) {
            Node p1 = (Node) j.next();
            int v1 = getVariableIndex(p1);
            Iterator k = paramNodes.iterator();
            for (int a = 0; a < n; ++a) k.next();
            while (k.hasNext()) {
                Node p2 = (Node) k.next();
                Assert._assert(p1 != p2);
                int v2 = getVariableIndex(p2);
                TypedBDD result = getAliased(v1, v2);
                for (Iterator l = result.iterator(); l.hasNext(); ) {
                    int h = ((Integer)l.next()).intValue();
                    BDD relation = V1o.ithVar(v1);
                    relation.orWith(V1o.ithVar(v2));
                    relation.andWith(H1o.ithVar(h));
                    BDD c_result = pointsTo.relprod(relation, V1o.set().and(H1o.set()));
                    relation.free();
                    if (!c_result.isZero()) {
                        //System.out.println("Aliased: "+m+" "+p1+" "+p2+":");
                        //System.out.println(result);
                        //System.out.println("Under contexts: "+c_result.toStringWithDomains());
                        hasAliased = true;
                    }
                    c_result.free();
                }
                result.free();
            }
            ++n;
        }
        return hasAliased;
    }
    
    public void findAliasedParameters() {
        int noAlias = 0, hasAlias = 0;
        for (Iterator i = methodToVariables.keySet().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            boolean hasAliased = findAliasedParameters(m);
            if (hasAliased) hasAlias++;
            else noAlias++;
        }
        System.out.println("No aliased parameters: "+noAlias);
        System.out.println("Has aliased parameters: "+hasAlias);
    }

    public static boolean TRACE_ESCAPE = false;

    public Collection getTargetMethods(ProgramLocation callSite) {
        return cg.getTargetMethods(mapCall(callSite));
    }
    
    public boolean isReturned(Node n) {
        return returned.contains(n);
    }
    
    public boolean isThrown(Node n) {
        return thrown.contains(n);
    }
    
    public void escapeAnalysis() {
        
        BDD escapingLocations = bdd.zero();
        
        List order = Traversals.postOrder(cg.getNavigator(), cg.getRoots());
        Map methodToVarBDD = new HashMap();
        for (Iterator i = order.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            BDD m_vars;
            SCComponent scc = (SCComponent) pn.getSCC(m);
            if (scc.isLoop()) {
                m_vars = bdd.zero();
            } else {
                Collection m_nodes = methodToVariables.getValues(m);
                m_vars = bdd.zero();
                for (Iterator j = cg.getCallees(m).iterator(); j.hasNext(); ) {
                    jq_Method callee = (jq_Method) j.next();
                    BDD m_vars2 = (BDD) methodToVarBDD.get(callee);
                    if (m_vars2 == null) continue;
                    m_vars.orWith(m_vars2.id());
                }
            }
            methodToVarBDD.put(m, m_vars);
            Collection m_nodes = methodToVariables.getValues(m);
            HashMap concreteNodes = new HashMap();
            for (Iterator j = m_nodes.iterator(); j.hasNext(); ) {
                Node o = (Node) j.next();
                if (o instanceof ConcreteTypeNode) {
                    ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                    ProgramLocation pl = ctn.getLocation();
                    pl = mapCall(pl);
                    concreteNodes.put(pl, ctn);
                }
                boolean bad = false;
                if (o.getEscapes()) {
                    if (TRACE_ESCAPE) System.out.println(o+" escapes, bad");
                    bad = true;
                } else if (cg.getRoots().contains(m) && isThrown(o)) {
                    if (TRACE_ESCAPE) System.out.println(o+" is thrown from root set, bad");
                    bad = true;
                } else {
                    Set passedParams = o.getPassedParameters();
                    if (passedParams != null) {
                        outer:
                        for (Iterator k = passedParams.iterator(); k.hasNext(); ) {
                            PassedParameter pp = (PassedParameter) k.next();
                            ProgramLocation mc = pp.getCall();
                            for (Iterator a = getTargetMethods(mc).iterator(); a.hasNext(); ) {
                                jq_Method m2 = (jq_Method) a.next();
                                if (m2.getBytecode() == null) {
                                    if (TRACE_ESCAPE) System.out.println(o+" is passed into a native method, bad");
                                    bad = true;
                                    break outer;
                                }
                            }
                        }
                    }
                }
                if (!bad) {
                    int v_i = getVariableIndex(o);
                    m_vars.orWith(V1o.ithVar(v_i));
                    //if (TRACE_ESCAPE) System.out.println("Var "+v_i+" is good: "+m_vars.toStringWithDomains());
                }
            }
            if (TRACE_ESCAPE) System.out.println("Non-escaping locations for "+m+" = "+m_vars.toStringWithDomains());
            ControlFlowGraph cfg = CodeCache.getCode(m);
            boolean trivial = false;
            for (QuadIterator j = new QuadIterator(cfg); j.hasNext(); ) {
                Quad q = j.nextQuad();
                if (q.getOperator() instanceof Operator.New ||
                    q.getOperator() instanceof Operator.NewArray) {
                    ProgramLocation pl = new QuadProgramLocation(m, q);
                    pl = mapCall(pl);
                    ConcreteTypeNode ctn = (ConcreteTypeNode) concreteNodes.get(pl);
                    if (ctn == null) {
                        //trivial = true;
                        trivial = q.getOperator() instanceof Operator.New;
                        System.out.println(cfg.getMethod()+": "+q+" trivially doesn't escape.");
                    } else {
                        int v_i = getVariableIndex(ctn);
                        BDD h = ci_pointsTo.restrict(V1o.ithVar(v_i));
                        Assert._assert(h.satCount(H1o.set()) == 1.0);
                        if (TRACE_ESCAPE) {
                            System.out.println("Heap location: "+h.toStringWithDomains()+" = "+ctn);
                            System.out.println("Pointed to by: "+ci_pointsTo.restrict(h).toStringWithDomains());
                        }
                        h.andWith(m_vars.not());
                        escapingLocations.orWith(h);
                    }
                }
            }
            if (trivial) {
                System.out.println(cfg.fullDump());
            }
        }
        for (Iterator i = methodToVarBDD.values().iterator(); i.hasNext(); ) {
            BDD b = (BDD) i.next();
            b.free();
        }

        BDD escapingHeap = escapingLocations.relprod(ci_pointsTo, V1o.set());
        escapingLocations.free();
        System.out.println("Escaping heap: "+escapingHeap.satCount(H1o.set()));
        //System.out.println("Escaping heap: "+escapingHeap.toStringWithDomains());
        BDD capturedHeap = escapingHeap.not();
        capturedHeap.andWith(H1o.varRange(0, heapobjIndexMap.size()-1));
        System.out.println("Captured heap: "+capturedHeap.satCount(H1o.set()));
        
        int capturedSites = 0;
        int escapedSites = 0;
        long capturedSize = 0L;
        long escapedSize = 0L;
        
        for (Iterator i=heapobjIndexMap.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            int ndex = getHeapIndex(n);
            if (n instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) n;
                jq_Reference t = (jq_Reference) ctn.getDeclaredType();
                if (t == null) continue;
                int size = 0;
                t.prepare();
                if (t instanceof jq_Class)
                    size = ((jq_Class) t).getInstanceSize();
                else
                    continue;
                BDD bdd = capturedHeap.and(H1o.ithVar(ndex));
                if (capturedHeap.and(H1o.ithVar(ndex)).isZero()) {
                    // not captured.
                    if (TRACE_ESCAPE) System.out.println("Escaped: "+n);
                    escapedSites ++;
                    escapedSize += size;
                } else {
                    // captured.
                    if (TRACE_ESCAPE) System.out.println("Captured: "+n);
                    capturedSites ++;
                    capturedSize += size;
                }
            }
        }
        System.out.println("Captured sites = "+capturedSites+", "+capturedSize+" bytes.");
        System.out.println("Escaped sites = "+escapedSites+", "+escapedSize+" bytes.");
    }

    public static Set findThreadRuns(CallGraph cg) {
        Set thread_runs = new HashSet();
        for (Iterator i=cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            if (m.getNameAndDesc().equals(CSPA.run_method)) {
                jq_Class k = m.getDeclaringClass();
                k.prepare();
                PrimordialClassLoader.getJavaLangThread().prepare();
                jq_Class jlr = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Runnable;");
                jlr.prepare();
                if (k.isSubtypeOf(PrimordialClassLoader.getJavaLangThread()) ||
                    k.isSubtypeOf(jlr)) {
                    System.out.println("Thread run method found: "+m);
                    thread_runs.add(m);
                }
            }
        }
        return thread_runs;
    }

    /** Load points-to results from the given file name prefix.
     */
    public void load(String fn) throws IOException {
        FileInputStream fis;
        DataInputStream di;
        
        di = new DataInputStream(fis = new FileInputStream(fn+".config"));
        readConfig(di);
        di.close();
        
        System.out.print("Loading BDDs...");
        this.pointsTo = bdd.load(fn+".bdd");
        System.out.print("pointsTo "+this.pointsTo.nodeCount()+" nodes");
        this.fieldPt = bdd.load(fn+".bdd2");
        System.out.print(", fieldPt "+this.fieldPt.nodeCount()+" nodes, ");
        System.out.println("done.");
        
        this.returned = new HashSet();
        this.thrown = new HashSet();
        this.passedParams = new GenericInvertibleMultiMap();
        
        System.out.print("Loading maps...");
        di = new DataInputStream(new FileInputStream(fn+".vars"));
        variableIndexMap = readIndexMap("Variable", di);
        di.close();
        
        di = new DataInputStream(new FileInputStream(fn+".heap"));
        heapobjIndexMap = readIndexMap("Heap", di);
        di.close();
        System.out.println("done.");

        buildContextInsensitive();

        initializeMethodMap();
    }

    private void buildContextInsensitive() {
        BDD context = V1c.set();
        context.andWith(H1c.set());
        this.ci_pointsTo = pointsTo.exist(context);
        context.free();
    }

    private void readConfig(DataInput in) throws IOException {
        String s = in.readLine();
        StringTokenizer st = new StringTokenizer(s);
        int VARBITS = Integer.parseInt(st.nextToken());
        int HEAPBITS = Integer.parseInt(st.nextToken());
        int FIELDBITS = Integer.parseInt(st.nextToken());
        int CLASSBITS = Integer.parseInt(st.nextToken());
        int CONTEXTBITS = Integer.parseInt(st.nextToken());
        int[] domainBits;
        int[] domainSpos;
        BDDDomain[] bdd_domains;
        domainBits = new int[] {VARBITS, CONTEXTBITS,
                                VARBITS, CONTEXTBITS,
                                FIELDBITS,
                                HEAPBITS, CONTEXTBITS,
                                HEAPBITS, CONTEXTBITS};
        domainSpos = new int[domainBits.length];
        
        long[] domains = new long[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1L << domainBits[i]);
        }
        bdd_domains = bdd.extDomain(domains);
        for (int i=0; i<domainBits.length; ++i) {
            Assert._assert(bdd_domains[i].varNum() == domainBits[i], "Domain "+i+" bits "+bdd_domains[i].varNum());
        }
        V1o = bdd_domains[0];
        V1c = bdd_domains[1];
        V2o = bdd_domains[2];
        V2c = bdd_domains[3];
        FD = bdd_domains[4];
        H1o = bdd_domains[5];
        H1c = bdd_domains[6];
        H2o = bdd_domains[7];
        H2c = bdd_domains[8];
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "FD_H2cxH2o_V2cxV1cxV2oxV1o_H1cxH1o");
        
        int[] varorder = CSPA.makeVarOrdering(bdd, domainBits, domainSpos,
                                              reverseLocal, ordering);
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
    }
    
    private IndexMap readIndexMap(String name, DataInput in) throws IOException {
        int size = Integer.parseInt(in.readLine());
        IndexMap m = new IndexMap(name, size);
        for (int i=0; i<size; ++i) {
            String s = in.readLine();
            StringTokenizer st = new StringTokenizer(s);
            Node n = MethodSummary.readNode(st);
            if (n == null && i != 0) {
                System.out.println("Cannot find node: "+s);
                n = new GlobalNode();
            }
            int j = m.get(n);
            //System.out.println(i+" = "+n);
            Assert._assert(i == j);
            while (st.hasMoreTokens()) {
                String t = st.nextToken();
                if (t.equals("returned"))
                    returned.add(n);
                else if (t.equals("thrown"))
                    thrown.add(n);
                else if (t.equals("passed")) {
                    PassedParameter pp = PassedParameter.read(st);
                    passedParams.add(pp, n);
                    n.recordPassedParameter(pp);
                }
            }
        }
        return m;
    }
    
    public static void main(String[] args) throws IOException {
        CSPAResults r = runAnalysis(args, null);
        r.interactive();
    }
    
    public static CSPAResults runAnalysis(String[] args, String addToClasspath) throws IOException {
        // We use bytecode maps.
        CodeCache.AlwaysMap = true;
        HostedVM.initialize();
        
        if (addToClasspath != null)
            PrimordialClassLoader.loader.addToClasspath(addToClasspath);
        
        String prefix = "";
        String sep = System.getProperty("file.separator");
        if (args.length > 0) {
            prefix = args[0];
            if (!prefix.endsWith(sep))
                prefix += sep;
        }
        
        int nodeCount = 500000;
        int cacheSize = 50000;
        BDDFactory bdd = BDDFactory.init(nodeCount, cacheSize);
        bdd.setMaxIncrease(nodeCount/4);
        CSPAResults r = new CSPAResults(bdd);
        r.loadCallGraph(prefix+"callgraph");
        r.load(prefix+System.getProperty("bddresults", "cspa"));
        return r;
    }

    public static String domainName(BDDDomain d) {
        switch (d.getIndex()) {
            case 0: return "V1o";
            case 1: return "V1c";
            case 2: return "V2o";
            case 3: return "V2c";
            case 4: return "FD";
            case 5: return "H1o";
            case 6: return "H1c";
            case 7: return "H2o";
            case 8: return "H2c";
            default: return "???";
        }
    }

    public String elementToString(BDDDomain d, int i) {
        StringBuffer sb = new StringBuffer();
        sb.append(domainName(d)+"("+i+")");
        Node n = null;
        if (d == V1o) {
            n = (Node) getVariableNode(i);
        } else if (d == V2o) {
            n = (Node) getVariableNode(i);
        } else if (d == H1o) {
            n = (Node) getHeapNode(i);
        } else if (d == H2o) {
            n = (Node) getHeapNode(i);
        }
        if (n != null) {
            sb.append(": ");
            sb.append(n.toString_short());
        }
        return sb.toString();
    }

    public static String domainNames(Set dom) {
        StringBuffer sb = new StringBuffer();
        for (Iterator i=dom.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            sb.append(domainName(d));
            if (i.hasNext()) sb.append(',');
        }
        return sb.toString();
    }
    
    public static final Comparator domain_comparator = new Comparator() {

        public int compare(Object arg0, Object arg1) {
            BDDDomain d1 = (BDDDomain) arg0;
            BDDDomain d2 = (BDDDomain) arg1;
            if (d1.getIndex() < d2.getIndex()) return -1;
            else if (d1.getIndex() > d2.getIndex()) return 1;
            else return 0;
        }
        
    };
    
    public static final boolean USE_BC_LOCATION = false;
    
    public static ProgramLocation getLoadLocation(jq_Class klass, int lineNum) {
        if (USE_BC_LOCATION)
            return getBCProgramLocation(klass, lineNum, Bytecodes.LoadInstruction.class, 0);
        else {
            ProgramLocation pl;
            pl = getQuadProgramLocation(klass, lineNum, Operator.ALoad.ALOAD_A.class, 0);
            if (pl != null) return pl;
            pl = getQuadProgramLocation(klass, lineNum, Operator.ALoad.ALOAD_P.class, 0);
            if (pl != null) return pl;
            pl = getQuadProgramLocation(klass, lineNum, Operator.Getfield.GETFIELD_A.class, 0);
            if (pl != null) return pl;
            return getQuadProgramLocation(klass, lineNum, Operator.Getfield.GETFIELD_P.class, 0);
        }
    }
    
    public static ProgramLocation getAllocLocation(jq_Class klass, int lineNum) {
        if (USE_BC_LOCATION)
            return getBCProgramLocation(klass, lineNum, Bytecodes.AllocationInstruction.class, 0);
        else {
            ProgramLocation pl;
            pl = getQuadProgramLocation(klass, lineNum, Operator.New.class, 0);
            if (pl != null) return pl;
            return getQuadProgramLocation(klass, lineNum, Operator.NewArray.class, 0);
        }
    }
    
    public static ProgramLocation getConstLocation(jq_Class klass, int lineNum) {
        if (USE_BC_LOCATION) {
            ProgramLocation pl = getBCProgramLocation(klass, lineNum, Bytecodes.LDC.class, 0);
            if (pl != null) return pl;
            return getBCProgramLocation(klass, lineNum, Bytecodes.LDC2_W.class, 0);
        } else {
            return getQuadProgramLocation(klass, lineNum, Operator.Move.class, 0);
        }
    }
    
    public static ProgramLocation getInvokeLocation(jq_Class klass, int lineNum) {
        if (USE_BC_LOCATION)
            return getBCProgramLocation(klass, lineNum, Bytecodes.InvokeInstruction.class, 0);
        else {
            return getQuadProgramLocation(klass, lineNum, Operator.Invoke.class, 0);
        }
    }
    
    public static ProgramLocation getBCProgramLocation(jq_Class klass, int lineNum, Class instructionType, int k) {
        klass.load();
        jq_Method m = klass.getMethodContainingLine((char) lineNum);
        if (m == null) return null;
        jq_LineNumberBC[] ln = m.getLineNumberTable();
        if (ln == null) return null;
        int i = 0;
        for ( ; i<ln.length; ++i) {
            if (ln[i].getLineNum() == lineNum) break;
        }
        if (i == ln.length) return null;
        int loIndex = ln[i].getStartPC();
        int hiIndex = m.getBytecode().length;
        if (i < ln.length-1) hiIndex = ln[i+1].getStartPC();
        ByteSequence bs = new ByteSequence(m.getBytecode(), loIndex, hiIndex-loIndex);
        try {
            while (bs.available() > 0) {
                int off = bs.getIndex();
                Bytecodes.Instruction in = Bytecodes.Instruction.readInstruction(klass.getCP(), bs);
                if (instructionType.isInstance(in)) {
                    if (k == 0)
                        return new BCProgramLocation(m, off);
                    --k;
                }
            }
        } catch (IOException x) {
            Assert.UNREACHABLE();
        }
        return null;
    }
    
    public static ProgramLocation getQuadProgramLocation(jq_Class klass, int lineNum, Class instructionType, int k) {
        klass.load();
        jq_Method m = klass.getMethodContainingLine((char) lineNum);
        if (m == null) return null;
        jq_LineNumberBC[] ln = m.getLineNumberTable();
        if (ln == null) return null;
        int i = 0;
        for ( ; i<ln.length; ++i) {
            if (ln[i].getLineNum() == lineNum) break;
        }
        if (i == ln.length) return null;
        int loIndex = ln[i].getStartPC();
        int hiIndex = m.getBytecode().length;
        if (i < ln.length-1) hiIndex = ln[i+1].getStartPC();
        Map bc_map = CodeCache.getBCMap(m);
        for (Iterator j = bc_map.entrySet().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            Quad q = (Quad) e.getKey();
            if (!instructionType.isInstance(q.getOperator()))
                continue;
            int index = ((Integer) e.getValue()).intValue();
            if (index >= loIndex && index < hiIndex)
                return new QuadProgramLocation(m, q);
        }
        return null;
    }
    
    public static ProgramLocation mapCall(ProgramLocation callSite) {
        if (USE_BC_LOCATION && callSite instanceof ProgramLocation.QuadProgramLocation) {
            jq_Method m = (jq_Method) callSite.getMethod();
            Map map = CodeCache.getBCMap(m);
            Quad q = ((ProgramLocation.QuadProgramLocation) callSite).getQuad();
            if (q == null) {
                Assert.UNREACHABLE("Error: cannot find call site "+callSite);
            }
            Integer i = (Integer) map.get(q);
            if (i == null) {
                Assert.UNREACHABLE("Error: no mapping for quad "+q);
            }
            int bcIndex = i.intValue();
            callSite = new ProgramLocation.BCProgramLocation(m, bcIndex);
        }
        return callSite;
    }
    
    /** ProgramLocation is the location of the method invocation that you want the return value of. */
    public int getReturnValueIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ReturnValueNode) {
                ReturnValueNode ctn = (ReturnValueNode) o;
                ProgramLocation pl2 = ctn.getMethodCall();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl2 != null && pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    public int getThrownExceptionIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ThrownExceptionNode) {
                ThrownExceptionNode ctn = (ThrownExceptionNode) o;
                ProgramLocation pl2 = ctn.getMethodCall();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl2 != null && pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    int[] getIndices(Collection c) {
        if (c == null) return null;
        int s = c.size();
        if (s == 0) return null;
        int[] r = new int[s];
        int j = -1;
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node n = (Node) i.next();
            r[++j] = getVariableIndex(n);
        }
        Assert._assert(j == r.length-1);
        return r;
    }
    
    public int[] getInvokeParamIndices(ProgramLocation pl, int k) {
        return getInvokeParamIndices(new PassedParameter(pl, k));
    }
    public int[] getInvokeParamIndices(PassedParameter pp) {
        Collection c = passedParams.getValues(pp);
        return getIndices(c);
    }
    
    public int[] getReturnValueIndices(jq_Method m) {
        Collection c = methodToVariables.getValues(m);
        LinkedList result = new LinkedList();
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (returned.contains(o))
                result.add(o);
        }
        return getIndices(result);
    }
    
    public int[] getThrownValueIndices(jq_Method m) {
        Collection c = methodToVariables.getValues(m);
        LinkedList result = new LinkedList();
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (thrown.contains(o))
                result.add(o);
        }
        return getIndices(result);
    }
    
    /** Multimap between methods and their variables. */ 
    MultiMap methodToVariables;
    
    public void initializeMethodMap() {
        methodToVariables = new GenericMultiMap();
        for (Iterator i = variableIndexMap.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o != null)
                methodToVariables.add(o.getDefiningMethod(), o);
        }
    }
    
    public int getLoadIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof FieldNode) {
                FieldNode ctn = (FieldNode) o;
                if (ctn.getLocations().contains(pl))
                    return getVariableIndex(o);
            }
        }
        return -1;
    }
    
    public int getHeapVariableIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl2 != null && pl.equals(pl2)) {
                    return getVariableIndex(o);
                }
            }
        }
        return -1;
    }
    
    public int getHeapObjectIndex(ProgramLocation pl) {
        Collection c = methodToVariables.getValues(pl.getMethod());
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Node o = (Node) i.next();
            if (o instanceof ConcreteTypeNode) {
                ConcreteTypeNode ctn = (ConcreteTypeNode) o;
                ProgramLocation pl2 = ctn.getLocation();
                Assert._assert(pl2.getMethod() == pl.getMethod());
                Assert._assert(pl.getClass() == pl2.getClass());
                if (pl2 != null && pl.equals(pl2)) {
                    return getHeapIndex(o);
                }
            }
        }
        return -1;
    }
    
    public Node getVariableNode(int v) {
        Node n = (Node) variableIndexMap.get(v);
        return n;
    }
    
    public int getVariableIndex(Node n) {
        int size = variableIndexMap.size();
        int v = variableIndexMap.get(n);
        Assert._assert(size == variableIndexMap.size());
        return v;
    }

    public Node getHeapNode(int v) {
        Node n = (Node) heapobjIndexMap.get(v);
        return n;
    }
    
    public int getHeapIndex(Node n) {
        int size = heapobjIndexMap.size();
        int v = heapobjIndexMap.get(n);
        Assert._assert(size == heapobjIndexMap.size());
        return v;
    }

    public ProgramLocation getHeapProgramLocation(int v) {
        Node n = (Node) getHeapNode(v);
        if (n instanceof ConcreteTypeNode)
            return ((ConcreteTypeNode) n).getLocation();
        return null;
    }

    public static jq_Class getClass(String classname) {
        jq_Class klass = (jq_Class) jq_Type.parseType(classname);
        return klass;
    }

    public static jq_Method getMethod(String classname, String name, String desc) {
        jq_Class klass = (jq_Class) jq_Type.parseType(classname);
        if (klass == null) return null;
        jq_Method m = (jq_Method) klass.getDeclaredMember(name, desc);
        return m;
    }
    
    public int getParameterIndex(jq_Method m, int k) {
        Collection c = methodToVariables.getValues(m);
        for (Iterator i = c.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof ParamNode) {
                ParamNode pn = (ParamNode) o;
                if (pn.getMethod() == m && k == pn.getIndex())
                    return getVariableIndex(pn);
            }
        }
        return -1;
    }
    
    public class TypedBDD {
        private static final int DEFAULT_NUM_TO_PRINT = 6;
        private static final int PRINT_ALL = -1;
        BDD bdd;
        Set dom;
        
        /**
         * @param pointsTo
         * @param domains
         */
        public TypedBDD(BDD bdd, BDDDomain[] domains) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.addAll(Arrays.asList(domains));
        }
        
        public TypedBDD(BDD bdd, Set domains) {
            this.bdd = bdd;
            this.dom = domains;
        }
            
        public TypedBDD(BDD bdd, BDDDomain d) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3, BDDDomain d4) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
            this.dom.add(d4);
        }
        
        public TypedBDD(BDD bdd, BDDDomain d1, BDDDomain d2, BDDDomain d3, BDDDomain d4, BDDDomain d5) {
            this.bdd = bdd;
            this.dom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            this.dom.add(d1);
            this.dom.add(d2);
            this.dom.add(d3);
            this.dom.add(d4);
            this.dom.add(d5);
        }
        
        public TypedBDD relprod(TypedBDD bdd1, TypedBDD set) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            newDom.addAll(bdd1.dom);
            if (!newDom.containsAll(set.dom)) {
                System.err.println("Warning! Quantifying domain that doesn't exist: "+domainNames(set.dom));
            }
            newDom.removeAll(set.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.relprod(bdd1.bdd, set.bdd), newDom);
        }
        
        public TypedBDD restrict(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.containsAll(bdd1.dom)) {
                System.err.println("Warning! Restricting domain that doesn't exist: "+domainNames(bdd1.dom));
            }
            if (bdd1.satCount() > 1.0) {
                System.err.println("Warning! Using restrict with more than one value");
            }
            newDom.removeAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.restrict(bdd1.bdd), newDom);
        }
        
        public TypedBDD exist(TypedBDD set) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.containsAll(set.dom)) {
                System.err.println("Warning! Quantifying domain that doesn't exist: "+domainNames(set.dom));
            }
            newDom.removeAll(set.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.exist(set.bdd), newDom);
        }
        
        public TypedBDD and(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            newDom.addAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.and(bdd1.bdd), newDom);
        }
        
        public TypedBDD or(TypedBDD bdd1) {
            Set newDom = SortedArraySet.FACTORY.makeSet(domain_comparator);
            newDom.addAll(dom);
            if (!newDom.equals(bdd1.dom)) {
                System.err.println("Warning! Or'ing BDD with different domains: "+domainNames(bdd1.dom));
            }
            newDom.addAll(bdd1.dom);
            System.out.println("Resulting domains: "+domainNames(newDom));
            return new TypedBDD(bdd.or(bdd1.bdd), newDom);
        }
        
        public boolean isZero() {
            return bdd.isZero();
        }
        
        public String getDomainNames() {
            return domainNames(dom);
        }
        
        private BDD getDomains() {
            BDDFactory f = bdd.getFactory();
            BDD r = f.one();
            for (Iterator i=dom.iterator(); i.hasNext(); ) {
                BDDDomain d = (BDDDomain) i.next();
                r.andWith(d.set());
            }
            return r;
        }
        
        public double satCount() {
            return bdd.satCount(getDomains());
        }
        
        public String toString() {
            return toString(DEFAULT_NUM_TO_PRINT);
        }
        
        public String toStringAll() {
            return toString(PRINT_ALL);
        }
        
        public String toString(int numToPrint) {
            BDD dset = getDomains();
            double s = bdd.satCount(dset);
            if (s == 0.) return "<empty>";
            BDD b = bdd.id();
            StringBuffer sb = new StringBuffer();
            int j = 0;
            while (!b.isZero()) {
                if (numToPrint != PRINT_ALL && j > numToPrint - 1) {
                    sb.append("\tand "+(long)b.satCount(dset)+" others.");
                    sb.append(Strings.lineSep);
                    break;
                }
                int[] val = b.scanAllVar();
                sb.append("\t(");
                BDD temp = b.getFactory().one();
                for (Iterator i=dom.iterator(); i.hasNext(); ) {
                    BDDDomain d = (BDDDomain) i.next();
                    int e = val[d.getIndex()];
                    sb.append(elementToString(d, e));
                    if (i.hasNext()) sb.append(' ');
                    temp.andWith(d.ithVar(e));
                }
                sb.append(')');
                b.applyWith(temp, BDDFactory.diff);
                ++j;
                sb.append(Strings.lineSep);
            }
            //sb.append(bdd.toStringWithDomains());
            return sb.toString();
        }

        public Iterator iterator() {
            Assert._assert(dom.size() == 1);
            final BDD t = this.bdd.id();
            final BDDDomain d = (BDDDomain) this.dom.iterator().next();
            return new UnmodifiableIterator() {

                public boolean hasNext() {
                    return !t.isZero();
                }

                public int nextInt() {
                    int v = t.scanVar(d);
                    if (v == -1)
                        throw new NoSuchElementException();
                    t.applyWith(d.ithVar(v), BDDFactory.diff);
                    return v;
                }

                public Object next() {
                    return new Integer(nextInt());
                }
            };
        }

        public void free() {
            bdd.free(); bdd = null;
        }
    }

    TypedBDD parseBDD(List a, String s) {
        if (s.equals("pointsTo")) {
            return new TypedBDD(pointsTo, V1c, V1o, H1c, H1o );
        }
        if (s.equals("fieldPt")) {
            return new TypedBDD(fieldPt, H1c, H1o, FD, H2c, H2o );
        }
        if (s.startsWith("V1o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V1o.ithVar(x), V1o);
        }
        if (s.startsWith("V1c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V1c.ithVar(x), V1c);
        }
        if (s.startsWith("V2o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V2o.ithVar(x), V2o);
        }
        if (s.startsWith("V2c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V2c.ithVar(x), V2c);
        }
        if (s.startsWith("H1o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H1o.ithVar(x), H1o);
        }
        if (s.startsWith("H1c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H1c.ithVar(x), H1c);
        }
        if (s.startsWith("H2o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H2o.ithVar(x), H2o);
        }
        if (s.startsWith("H2c(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H2c.ithVar(x), H2c);
        }
        if (s.startsWith("FD(")) {
            int x = Integer.parseInt(s.substring(3, s.length()-1));
            return new TypedBDD(FD.ithVar(x), FD);
        }
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return new TypedBDD(d.domain(), d);
        }
        int i = Integer.parseInt(s)-1;
        return (TypedBDD) a.get(i);
    }

    TypedBDD parseBDDset(List a, String s) {
        if (s.equals("V1")) {
            BDD b = V1o.set(); b.andWith(V1c.set());
            return new TypedBDD(b, V1c, V1o);
        }
        if (s.equals("H1")) {
            BDD b = H1o.set(); b.andWith(H1c.set());
            return new TypedBDD(b, H1c, H1o);
        }
        if (s.equals("C")) {
            BDD b = V1c.set(); b.andWith(H1c.set());
            return new TypedBDD(b, V1c, H1c);
        }
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return new TypedBDD(d.set(), d);
        }
        int i = Integer.parseInt(s)-1;
        return (TypedBDD) a.get(i);
    }

    BDDDomain parseDomain(String dom) {
        if (dom.equals("V1c")) return V1c;
        if (dom.equals("V1o")) return V1o;
        if (dom.equals("V2c")) return V2c;
        if (dom.equals("V2o")) return V2o;
        if (dom.equals("FD")) return FD;
        if (dom.equals("H1c")) return H1c;
        if (dom.equals("H1o")) return H1o;
        if (dom.equals("H2c")) return H2c;
        if (dom.equals("H2o")) return H2o;
        return null;
    }

    void interactive() {
        int i = 1;
        List results = new ArrayList();
        DataInput in = new DataInputStream(System.in);
        for (;;) {
            boolean increaseCount = true;
            boolean listAll = false;
            
            try {
                System.out.print(i+"> ");
                String s = in.readLine();
                if (s == null) return;
                StringTokenizer st = new StringTokenizer(s);
                if (!st.hasMoreElements()) continue;
                String command = st.nextToken();
                if (command.equals("relprod")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = bdd1.relprod(bdd2, set);
                    results.add(r);
                } else if (command.equals("restrict")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.restrict(bdd2);
                    results.add(r);
                } else if (command.equals("exist")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = bdd1.exist(set);
                    results.add(r);
                } else if (command.equals("and")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.and(bdd2);
                    results.add(r);
                } else if (command.equals("or")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = bdd1.or(bdd2);
                    results.add(r);
                } else if (command.equals("var")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = new TypedBDD(V1o.ithVar(z), V1o);
                    results.add(r);
                } else if (command.equals("heap")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = new TypedBDD(H1o.ithVar(z), H1o);
                    results.add(r);
                } else if (command.equals("quit") || command.equals("exit")) {
                    break;
                } else if (command.equals("aliased")) {
                    int z = Integer.parseInt(st.nextToken());
                    TypedBDD r = getAliasedLocations(z);
                    results.add(r);
                } else if (command.equals("heapType")) {
                    jq_Reference typeRef = (jq_Reference) jq_Type.parseType(st.nextToken());
                    if (typeRef != null) {
                        TypedBDD r = getAllHeapOfType(typeRef);
                        results.add(r);
                    }
                } else if (command.equals("list")) {
                    TypedBDD r = parseBDD(results, st.nextToken());
                    results.add(r);
                    listAll = true;
                    System.out.println("Domains: " + r.getDomainNames());
                } else if (command.equals("contextvar")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = getVariableNode (varNum);
                    jq_Method m = n.getDefiningMethod();
                    Number c = new BigInteger(st.nextToken(), 10);
                    if (m == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        Path trace = pn.getPath(m, c);
                        System.out.println(m+" context "+c+":\n"+trace);
                    }
                    increaseCount = false;
                } else if (command.equals("contextheap")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = (Node) getHeapNode(varNum);
                    jq_Method m = n.getDefiningMethod();
                    Number c = new BigInteger(st.nextToken(), 10);
                    if (m == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        Path trace = pn.getPath(m, c);
                        System.out.println(m+" context "+c+": "+trace);
                    }
                    increaseCount = false;
                } else if (command.equals("aliasedparams")) {
                    findAliasedParameters();
                    increaseCount = false;
                } else if (command.equals("escape")) {
                    escapeAnalysis();
                    increaseCount = false;
                } else {
                    System.err.println("Unrecognized command");
                    increaseCount = false;
                    //results.add(new TypedBDD(bdd.zero(), Collections.EMPTY_SET));
                }
            } catch (IOException e) {
                System.err.println("Error: IOException");
                increaseCount = false;
            } catch (NumberFormatException e) {
                System.err.println("Parse error: NumberFormatException");
                increaseCount = false;
            } catch (NoSuchElementException e) {
                System.err.println("Parse error: NoSuchElementException");
                increaseCount = false;
            } catch (IndexOutOfBoundsException e) {
                System.err.println("Parse error: IndexOutOfBoundsException");
                increaseCount = false;
            }

            if (increaseCount) {
                TypedBDD r = (TypedBDD) results.get(i-1);
                if (listAll) {
                    System.out.println(i+" -> "+r.toStringAll());
                } 
                else {
                    System.out.println(i+" -> "+r);
                }
                Assert._assert(i == results.size());
                ++i;
            }
        }
    }

    public CSPAResults(BDDFactory bdd) {
        this.bdd = bdd;
    }

}

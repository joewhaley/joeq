// PAResults.java, created Nov 3, 2003 12:34:24 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.LoadedCallGraph;
import Main.HostedVM;
import Util.Assert;
import Util.Strings;
import Util.Collections.HashWorklist;
import Util.Graphs.PathNumbering.Path;

/**
 * Records results for pointer analysis.  The results can be saved and reloaded.
 * This class also provides methods to query the results.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class PAResults {

    PA r;
    
    CallGraph cg;
    
    public PAResults(PA pa) {
        r = pa;
    }
    
    public static void main(String[] args) throws IOException {
        initialize(null);
        PAResults r = loadResults(args, null);
        r.interactive();
    }
    
    public static void initialize(String addToClasspath) {
        // We use bytecode maps.
        CodeCache.AlwaysMap = true;
        HostedVM.initialize();
        
        if (addToClasspath != null)
            PrimordialClassLoader.loader.addToClasspath(addToClasspath);
    }
    
    public static PAResults loadResults(String[] args, String addToClasspath) throws IOException {
        String prefix;
        if (args.length > 0) {
            prefix = args[0];
            String sep = System.getProperty("file.separator");
            if (!prefix.endsWith(sep))
                prefix += sep;
        } else {
            prefix = "";
        }
        String fileName = System.getProperty("bddresults", "pa");
        String bddfactory = "typed";
        PAResults r = loadResults(bddfactory, prefix, fileName);
        return r;
    }
    
    public static PAResults loadResults(String bddfactory,
                                        String prefix,
                                        String fileName) throws IOException {
        PA pa = PA.loadResults(bddfactory, prefix, fileName);
        PAResults r = new PAResults(pa);
        r.loadCallGraph(prefix+"callgraph");
        // todo: load path numbering instead of renumbering.
        pa.numberPaths(r.cg, false);
        return r;
    }
    
    /** Load call graph from the given file name.
     */
    public void loadCallGraph(String fn) throws IOException {
        cg = new LoadedCallGraph(fn);
    }
    
    public void interactive() {
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
                if (command.equals("quit") || command.equals("exit")) {
                    break;
                } else if (command.equals("relprod")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = (TypedBDD) bdd1.relprod(bdd2, set);
                    results.add(r);
                } else if (command.equals("restrict")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = (TypedBDD) bdd1.restrict(bdd2);
                    results.add(r);
                } else if (command.equals("exist")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD set = parseBDDset(results, st.nextToken());
                    TypedBDD r = (TypedBDD) bdd1.exist(set);
                    results.add(r);
                } else if (command.equals("and")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = (TypedBDD) bdd1.and(bdd2);
                    results.add(r);
                } else if (command.equals("or")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD bdd2 = parseBDD(results, st.nextToken());
                    TypedBDD r = (TypedBDD) bdd1.or(bdd2);
                    results.add(r);
                } else if (command.equals("list")) {
                    TypedBDD r = parseBDD(results, st.nextToken());
                    results.add(r);
                    listAll = true;
                    System.out.println("Domains: " + r.getDomainSet());
                } else if (command.equals("contextvar")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = getVariableNode(varNum);
                    if (n == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        jq_Method m = n.getDefiningMethod();
                        Number c = new BigInteger(st.nextToken(), 10);
                        if (m == null) {
                            System.out.println("No method for node "+n);
                        } else {
                            Path trace = r.vCnumbering.getPath(m, c);
                            System.out.println(m+" context "+c+":\n"+trace);
                        }
                    }
                    increaseCount = false;
                } else if (command.equals("contextheap")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = (Node) getHeapNode(varNum);
                    if (n == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        jq_Method m = n.getDefiningMethod();
                        Number c = new BigInteger(st.nextToken(), 10);
                        if (m == null) {
                            System.out.println("No method for node "+n);
                        } else {
                            Path trace = r.hCnumbering.getPath(m, c);
                            System.out.println(m+" context "+c+": "+trace);
                        }
                    }
                    increaseCount = false;
                } else if (command.equals("method")) {
                    jq_Class c = (jq_Class) jq_Type.parseType(st.nextToken());
                    if (c == null) {
                        System.out.println("Cannot find class");
                        increaseCount = false;
                    } else {
                        jq_Method m = c.getDeclaredMethod(st.nextToken());
                        if (m == null) {
                            System.out.println("Cannot find method");
                            increaseCount = false;
                        } else {
                            System.out.println("Method: "+m);
                            int k = getMethodIndex(m);
                            results.add(r.M.ithVar(k));
                        }
                    }
                } else if (command.equals("thread")) {
                    int k = Integer.parseInt(st.nextToken());
                    jq_Method run = null;
                    Iterator j = PA.thread_runs.keySet().iterator();
                    while (--k >= 0) {
                        run = (jq_Method) j.next();
                    }
                    System.out.println(k+": "+run);
                    int x = getMethodIndex(run);
                    results.add(r.M.ithVar(x));
                } else if (command.equals("threadlocal")) {
                    TypedBDD r = (TypedBDD) getThreadLocalObjects();
                    results.add(r);
                } else if (command.equals("reachable")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    TypedBDD r = (TypedBDD) getReachableVars(bdd1);
                    results.add(r);
                } else if (command.equals("usedef")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    BDD r = calculateUseDef(bdd1);
                    results.add(r);
                } else if (command.equals("printusedef")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    printUseDefChain(bdd1);
                    increaseCount = false;
                } else if (command.equals("dumpusedef")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    DataOutput out = new DataOutputStream(new FileOutputStream("usedef.dot"));
                    this.defUseGraph(bdd1, false, out);
                    increaseCount = false;
                } else if (command.equals("defuse")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    BDD r = calculateDefUse(bdd1);
                    results.add(r);
                } else if (command.equals("dumpdefuse")) {
                    TypedBDD bdd1 = parseBDD(results, st.nextToken());
                    DataOutput out = new DataOutputStream(new FileOutputStream("defuse.dot"));
                    this.defUseGraph(bdd1, true, out);
                    increaseCount = false;
                } else if (command.equals("help")) {
                    printHelp();
                    increaseCount = false;
                } else {
                    System.err.println("Unrecognized command");
                    increaseCount = false;
                    //results.add(new TypedBDD(bdd.zero(), Collections.EMPTY_SET));
                }
            } catch (Exception e) {
                System.err.println("Error: "+e);
                e.printStackTrace();
                increaseCount = false;
            }

            if (increaseCount) {
                TypedBDD r = (TypedBDD) results.get(i-1);
                if (listAll) {
                    System.out.println(i+" -> "+toString(r, -1));
                } else {
                    System.out.println(i+" -> "+toString(r, DEFAULT_NUM_TO_PRINT));
                }
                Assert._assert(i == results.size());
                ++i;
            }
        }
    }
    
    public static void printHelp() {
        System.out.println("dumpconnect # <fn>:   dump heap connectivity graph for heap object # to file fn");
        System.out.println("dumpallconnect <fn>:  dump entire heap connectivity graph to file fn");
        System.out.println("escape:               run escape analysis");
        System.out.println("findequiv:            find equivalent objects");
        System.out.println("relprod b1 b2 bs:     relational product of b1 and b2 w.r.t. set bs");
    }

    public Node getVariableNode(int v) {
        if (v < 0 || v >= r.Vmap.size())
            return null;
        Node n = (Node) r.Vmap.get(v);
        return n;
    }
    
    public int getVariableIndex(Node n) {
        if (!r.Vmap.contains(n)) return -1;
        int v = r.Vmap.get(n);
        return v;
    }

    public ProgramLocation getInvoke(int v) {
        if (v < 0 || v >= r.Imap.size())
            return null;
        ProgramLocation n = (ProgramLocation) r.Imap.get(v);
        return n;
    }
    
    public int getInvokeIndex(ProgramLocation n) {
        n = LoadedCallGraph.mapCall(n);
        if (!r.Imap.contains(n)) return -1;
        int v = r.Imap.get(n);
        return v;
    }
    
    public Node getHeapNode(int v) {
        if (v < 0 || v >= r.Hmap.size())
            return null;
        Node n = (Node) r.Hmap.get(v);
        return n;
    }
    
    public int getHeapIndex(Node n) {
        if (!r.Hmap.contains(n)) return -1;
        int v = r.Hmap.get(n);
        return v;
    }

    public jq_Field getField(int v) {
        if (v < 0 || v >= r.Fmap.size())
            return null;
        jq_Field n = (jq_Field) r.Fmap.get(v);
        return n;
    }
    
    public int getFieldIndex(jq_Field m) {
        if (!r.Fmap.contains(m)) return -1;
        int v = r.Fmap.get(m);
        return v;
    }
    
    public jq_Reference getType(int v) {
        if (v < 0 || v >= r.Tmap.size())
            return null;
        jq_Reference n = (jq_Reference) r.Tmap.get(v);
        return n;
    }
    
    public int getTypeIndex(jq_Type m) {
        if (!r.Tmap.contains(m)) return -1;
        int v = r.Tmap.get(m);
        return v;
    }
    
    public jq_Method getName(int v) {
        if (v < 0 || v >= r.Nmap.size())
            return null;
        jq_Method n = (jq_Method) r.Nmap.get(v);
        return n;
    }
    
    public int getNameIndex(jq_Method m) {
        if (!r.Nmap.contains(m)) return -1;
        int v = r.Nmap.get(m);
        return v;
    }
    
    public jq_Method getMethod(int v) {
        if (v < 0 || v >= r.Mmap.size())
            return null;
        jq_Method n = (jq_Method) r.Mmap.get(v);
        return n;
    }
    
    public int getMethodIndex(jq_Method m) {
        if (!r.Mmap.contains(m)) return -1;
        int v = r.Mmap.get(m);
        return v;
    }
    
    BDDDomain parseDomain(String dom) {
        for (int i = 0; i < r.bdd.numberOfDomains(); ++i) {
            if (dom.equals(r.bdd.getDomain(i).getName()))
                return r.bdd.getDomain(i);
        }
        return null;
    }
    
    TypedBDD parseBDD(List a, String s) {
        int paren_index = s.indexOf('(');
        if (paren_index > 0) {
            int close_index = s.indexOf(')');
            if (close_index <= paren_index) return null;
            String domainName = s.substring(0, paren_index);
            BDDDomain dom = parseDomain(domainName);
            if (dom == null) return null;
            long index = Long.parseLong(s.substring(paren_index+1, close_index));
            return (TypedBDD) dom.ithVar(index);
        }
        try {
            Field f = PA.class.getDeclaredField(s);
            if (f.getType() == BDD.class) {
                return (TypedBDD) f.get(r);
            }
        } catch (NoSuchFieldException e) {
        } catch (IllegalArgumentException e) {
        } catch (IllegalAccessException e) {
        }
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return (TypedBDD) d.domain();
        }
        try {
            int num = Integer.parseInt(s) - 1;
            if (num >= 0 && num < a.size()) {
                return (TypedBDD) a.get(num);
            }
        } catch (NumberFormatException e) { }
        return null;
    }

    TypedBDD parseBDDset(List a, String s) {
        BDDDomain d = parseDomain(s);
        if (d != null) {
            return (TypedBDD) d.set();
        }
        return parseBDD(a, s);
    }

    public static final int DEFAULT_NUM_TO_PRINT = 10;
    
    public String toString(TypedBDD b, int numToPrint) {
        if (b.isZero()) return "<empty>";
        TypedBDD dset = (TypedBDD) b.getFactory().one();
        Set domains = b.getDomainSet();
        for (Iterator i = domains.iterator(); i.hasNext(); ) {
            BDDDomain d = (BDDDomain) i.next();
            dset.andWith(d.set());
        }
        StringBuffer sb = new StringBuffer();
        int j = 0;
        for (Iterator i = b.iterator(); i.hasNext(); ++j) {
            if (numToPrint >= 0 && j > numToPrint - 1) {
                sb.append("\tand "+(long)b.satCount(dset)+" others.");
                sb.append(Strings.lineSep);
                break;
            }
            TypedBDD b1 = (TypedBDD) i.next();
            sb.append("\t(");
            sb.append(b1.toStringWithDomains(r.TS));
            sb.append(')');
            sb.append(Strings.lineSep);
        }
        return sb.toString();
    }

    /***** COOL OPERATIONS BELOW *****/
    
    /** Given a set of uses (V1xV1c), calculate the set of definitions (V1xV1c) that
     * reach that set of uses.  Only does one step at a time; you'll have to iterate
     * to get the transitive closure.
     */
    public BDD calculateUseDef(BDD r_v1) {
        // A: v2=v1;
        BDD b = r.A.relprod(r_v1, r.V1set); // V2
        b.replaceWith(r.V2toV1);
        //System.out.println("Arguments/Return Values = "+b.satCount(r.V2set));
        BDD r_v2 = r_v1.replace(r.V1toV2);
        // L: v2=v1.f;
        BDD c = r.L.relprod(r_v2, r.V2set); // V1xF
        r_v2.free();
        BDD d = r.vP.relprod(c, r.V1set); // H1xF
        c.free();
        BDD e = r.hP.relprod(d, r.H1Fset); // H2
        d.free();
        e.replaceWith(r.H2toH1);
        BDD f = r.vP.relprod(e, r.H1set); // V1
        //System.out.println("Loads/Stores = "+f.satCount(r.V1set));
        e.free();
        f.orWith(b);
        return f;
    }
    
    /** Given a set of definitions (V1xV1c), calculate the set of uses (V1xV1c) that
     * reach that set of definitions.  Only does one step at a time; you'll have to
     * iterate to get the transitive closure.
     */
    public BDD calculateDefUse(BDD r_v1) {
        BDD r_v2 = r_v1.replace(r.V1toV2);
        // A: v2=v1;
        BDD b = r.A.relprod(r_v2, r.V2set);
        System.out.println("Arguments/Return Values = "+b.satCount(r.V1set));
        // S: v1.f=v2;
        BDD c = r.S.relprod(r_v2, r.V2set); // V1xF
        r_v2.free();
        BDD d = r.vP.relprod(c, r.V1set); // H1xF
        c.free();
        BDD e = r.vP.relprod(d, r.H1set); // V1xF
        d.free();
        // L: v2=v1.f;
        BDD f = r.L.relprod(e, r.V1Fset); // V2
        f.replaceWith(r.V2toV1);
        System.out.println("Loads/Stores = "+f.satCount(r.V1set));
        f.orWith(b);
        return f;
    }

    /** Output def-use or use-def graph in dot format.
     */
    public void defUseGraph(BDD vPrelation, boolean direction, DataOutput out) throws IOException {
        out.writeBytes("digraph \");");
        if (direction) out.writeBytes("DefUse");
        else out.writeBytes("UseDef");
        out.writeBytes("\" {\n");
        HashWorklist w = new HashWorklist(true);
        BDD c = vPrelation.id();
        int k = -1;
        Node n = null;
        for (;;) {
            while (!c.isZero()) {
                int k2 = (int) c.scanVar(r.V1);
                Node n2 = getVariableNode(k2);
                if (w.add(n2)) {
                    String name = n2.toString();
                    out.writeBytes("n"+k2+" [label=\""+name+"\"];\n");
                }
                if (n != null) {
                    if (direction) {
                        out.writeBytes("n"+k+
                                       " -> n"+k2+";\n");
                    } else {
                        out.writeBytes("n"+k2+
                                       " -> n"+k+";\n");
                    }
                }
                BDD q = r.V1.ithVar(k2);
                q.andWith(r.V1c.domain());
                c.applyWith(q, BDDFactory.diff);
            }
            if (w.isEmpty()) break;
            n = (Node) w.pull();
            k = getVariableIndex(n);
            BDD b = r.V1.ithVar(k);
            b.andWith(r.V1c.domain());
            c = direction?calculateDefUse(b):calculateUseDef(b);
        }
        out.writeBytes("}\n");
    }
    
    /** Prints out the chain of use-defs, starting from the given uses (V1xV1c).
     */
    public void printUseDefChain(BDD vPrelation) {
        BDD visited = r.bdd.zero();
        vPrelation = vPrelation.id();
        for (int k = 1; !vPrelation.isZero(); ++k) {
            System.out.println("Step "+k+":");
            System.out.println(vPrelation.toStringWithDomains(r.TS));
            visited.orWith(vPrelation.id());
            // A: v2=v1;
            BDD b = r.A.relprod(vPrelation, r.V1set);
            System.out.println("Arguments/Return Values = "+b.satCount(r.V2set));
            // L: v2=v1.f;
            vPrelation.replaceWith(r.V1toV2);
            BDD c = r.L.relprod(vPrelation, r.V2set); // V1xF
            vPrelation.free();
            BDD d = r.vP.relprod(c, r.V1set); // H1xF
            c.free();
            BDD e = r.hP.relprod(d, r.H1Fset); // H2
            d.free();
            e.replaceWith(r.H2toH1);
            BDD f = r.vP.relprod(e, r.H1set); // V1
            System.out.println("Loads/Stores = "+f.satCount(r.V1set));
            e.free();
            vPrelation = b;
            vPrelation.replaceWith(r.V2toV1);
            vPrelation.orWith(f);
            vPrelation.applyWith(visited.id(), BDDFactory.diff);
        }
    }
    
    /** Starting from a method with a context (MxV1c), calculate the set of
     * transitively-reachable variables (V1xV1c).
     */
    public BDD getReachableVars(BDD method_plus_context0) {
        System.out.println("Method = "+method_plus_context0.toStringWithDomains());
        BDD result = r.bdd.zero();
        BDD allInvokes = r.mI.exist(r.Nset);
        BDD new_m = method_plus_context0.id();
        BDD V2cIset = r.Iset.and(r.V2c.set());
        for (int k=1; ; ++k) {
            System.out.println("Iteration "+k);
            BDD vars = new_m.relprod(r.mV, r.Mset); // V1cxM x MxV1 = V1cxV1
            result.orWith(vars);
            BDD invokes = new_m.relprod(allInvokes, r.Mset); // V1cxM x MxI = V1cxI
            invokes.replaceWith(r.V1ctoV2c); // V2cxI
            BDD methods = invokes.relprod(r.IEc, V2cIset); // V2cxI x V2cxIxV1cxM = V1cxM
            new_m.orWith(methods);
            new_m.applyWith(method_plus_context0.id(), BDDFactory.diff);
            if (new_m.isZero()) break;
            method_plus_context0.orWith(new_m.id());
        }
        return result;
    }
    
    /** Return the set of thread-local objects (H1xH1c).
     */
    public BDD getThreadLocalObjects() {
        jq_NameAndDesc main_nd = new jq_NameAndDesc("main", "([Ljava/lang/String;)V");
        jq_Method main = null;
        for (Iterator i = r.Mmap.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (main_nd.equals(m.getNameAndDesc())) {
                main = m;
                System.out.println("Using main() method: "+main);
                break;
            }
        }
        BDD allObjects = r.bdd.zero();
        BDD sharedObjects = r.bdd.zero();
        if (main != null) {
            int M_i = r.Mmap.get(main);
            BDD m = r.M.ithVar(M_i);
            m.andWith(r.V1c.ithVar(0));
            System.out.println("Main: "+m.toStringWithDomains());
            BDD b = getReachableVars(m);
            m.free();
            System.out.println("Reachable vars: "+b.satCount(r.V1set));
            BDD b2 = b.relprod(r.vP, r.V1set);
            b.free();
            System.out.println("Reachable objects: "+b2.satCount(r.H1set));
            allObjects.orWith(b2);
        }
        for (Iterator i = PA.thread_runs.keySet().iterator(); i.hasNext(); ) {
            jq_Method run = (jq_Method) i.next();
            int M_i = r.Mmap.get(run);
            Set t_runs = (Set) PA.thread_runs.get(run);
            if (t_runs == null) {
                System.out.println("Unknown run() method: "+run);
                continue;
            }
            Iterator k = t_runs.iterator();
            for (int j = 0; k.hasNext(); ++j) {
                Node q = (Node) k.next();
                BDD m = r.M.ithVar(M_i);
                m.andWith(r.V1c.ithVar(j));
                System.out.println("Thread: "+m.toStringWithDomains()+" Object: "+q);
                BDD b = getReachableVars(m);
                m.free();
                System.out.println("Reachable vars: "+b.satCount(r.V1set));
                BDD b2 = b.relprod(r.vP, r.V1set);
                b.free();
                System.out.println("Reachable objects: "+b2.satCount(r.H1set));
                BDD b3 = allObjects.and(b2);
                System.out.println("Shared objects: "+b3.satCount(r.H1set));
                sharedObjects.orWith(b3);
                allObjects.orWith(b2);
            }
        }
        
        System.out.println("All shared objects: "+sharedObjects.satCount(r.H1set));
        allObjects.applyWith(sharedObjects, BDDFactory.diff);
        System.out.println("All local objects: "+allObjects.satCount(r.H1set));
        
        return allObjects;
    }
    
}

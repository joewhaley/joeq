// CSPAResults.java, created Aug 7, 2003 12:34:24 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.*;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.*;

import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.LoadedCallGraph;
import Main.HostedVM;
import Util.Assert;
import Util.Strings;
import Util.Collections.IndexMap;
import Util.Collections.SortedArraySet;
import Util.Graphs.PathNumbering;
import Util.Graphs.PathNumbering.Path;

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
    BDDDomain V1c;
    /** BDD domain for variable number. */
    BDDDomain V1o;
    /** BDD domain for context number of heap object. */
    BDDDomain H1c;
    /** BDD domain for heap object number. */
    BDDDomain H1o;
    
    /** Points-to BDD: V1c x V1o x H1c x H1o.
     * This contains the result of the points-to analysis.
     * A relation (V,H) is in the BDD if variable V can point to heap object H.
     */
    BDD pointsTo;

    BDD getAliasedLocations(Node variable) {
        BDD context = V1c.set();
        context.andWith(H1c.set());
        BDD ci_pointsTo = pointsTo.exist(context);
        context.free();
        int i = variableIndexMap.get(variable);
        BDD a = V1o.ithVar(i);
        BDD heapObjs = ci_pointsTo.restrict(a);
        a.free();
        BDD result = ci_pointsTo.relprod(heapObjs, H1o.set());
        heapObjs.free();
        return result;
    }
    
    BDD getAllHeapOfType(jq_Reference type) {
        int j=0;
        BDD result = bdd.zero();
        for (Iterator i=heapobjIndexMap.iterator(); i.hasNext(); ++j) {
            Node n = (Node) i.next();
            Assert._assert(this.heapobjIndexMap.get(n) == j);
            if (n != null && n.getDeclaredType() == type)
                result.orWith(H1o.ithVar(j));
        }
        return result;
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
    public BDD getContextInsensitivePointsTo() {
        BDD context = V1c.set();
        context.andWith(H1c.set());
        return pointsTo.exist(context);
    }

    /** Load call graph from the given file name.
     */
    public void loadCallGraph(String fn) throws IOException {
        cg = new LoadedCallGraph(fn);
        pn = new PathNumbering();
        Number paths = pn.countPaths(cg.getRoots(), cg.getCallSiteNavigator());
        System.out.println("Number of paths in call graph="+paths);
    }

    /** Load points-to results from the given file name prefix.
     */
    public void load(String fn) throws IOException {
        DataInput di;
        
        di = new DataInputStream(new FileInputStream(fn+".config"));
        readConfig(di);
        
        this.pointsTo = bdd.load(fn+".bdd");
        
        di = new DataInputStream(new FileInputStream(fn+".vars"));
        variableIndexMap = readIndexMap("Variable", di);
        
        di = new DataInputStream(new FileInputStream(fn+".heap"));
        heapobjIndexMap = readIndexMap("Heap", di);
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
        V1o = bdd_domains[0];
        V1c = bdd_domains[1];
        H1o = bdd_domains[5];
        H1c = bdd_domains[6];
        
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
            int j = m.get(n);
            //System.out.println(i+" = "+n);
            Assert._assert(i == j);
        }
        return m;
    }

    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        
        int nodeCount = 1000000;
        int cacheSize = 100000;
        BDDFactory bdd = BDDFactory.init(nodeCount, cacheSize);
        bdd.setMaxIncrease(nodeCount/4);
        CSPAResults r = new CSPAResults(bdd);
        r.loadCallGraph("callgraph");
        r.load("cspa");
        r.interactive();
    }

    public static String domainName(BDDDomain d) {
        switch (d.getIndex()) {
            case 0: return "V1o";
            case 1: return "V1c";
            case 5: return "H1o";
            case 6: return "H1c";
            default: return "???";
        }
    }

    public String elementToString(BDDDomain d, int i) {
        StringBuffer sb = new StringBuffer();
        Node n = null;
        if (d == V1o) {
            sb.append("V1o("+i+"): ");
            n = (Node) variableIndexMap.get(i);
        } else if (d == H1o) {
            sb.append("H1o("+i+"): ");
            n = (Node) heapobjIndexMap.get(i);
        }
        if (n != null) {
            sb.append(n.toString_short());
        } else {
            sb.append(domainName(d)+"("+i+")");
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
        
        public String getDomainNames() {
            return domainNames(dom);
        }
        
        BDD getDomains() {
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
    }

    TypedBDD parseBDD(List a, String s) {
        if (s.equals("pointsTo")) {
            return new TypedBDD(pointsTo,
                                new BDDDomain[] { V1c, V1o, H1c, H1o });
        }
        if (s.startsWith("V1o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(V1o.ithVar(x), V1o);
        }
        if (s.startsWith("H1o(")) {
            int x = Integer.parseInt(s.substring(4, s.length()-1));
            return new TypedBDD(H1o.ithVar(x), H1o);
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
        if (dom.equals("H1c")) return H1c;
        if (dom.equals("H1o")) return H1o;
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
                    Node node = (Node) variableIndexMap.get(z);
                    TypedBDD r = new TypedBDD(getAliasedLocations(node), V1o);
                    results.add(r);
                } else if (command.equals("heapType")) {
                    jq_Reference typeRef = (jq_Reference) jq_Type.parseType(st.nextToken());
                    if (typeRef != null) {
                        TypedBDD r = new TypedBDD(getAllHeapOfType(typeRef), H1o);
                        results.add(r);
                    }
                } else if (command.equals("list")) {
                    TypedBDD r = parseBDD(results, st.nextToken());
                    results.add(r);
                    listAll = true;
                    System.out.println("Domains: " + r.getDomainNames());
                } else if (command.equals("contextvar")) {
                    int varNum = Integer.parseInt(st.nextToken());
                    Node n = (Node) variableIndexMap.get(varNum);
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
                    Node n = (Node) heapobjIndexMap.get(varNum);
                    jq_Method m = n.getDefiningMethod();
                    Number c = new BigInteger(st.nextToken(), 10);
                    if (m == null) {
                        System.out.println("No method for node "+n);
                    } else {
                        Path trace = pn.getPath(m, c);
                        System.out.println(m+" context "+c+": "+trace);
                    }
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

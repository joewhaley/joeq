// PathNumbering.java, created Aug 16, 2003 1:49:33 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Graphs;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.math.BigInteger;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.StringTokenizer;
import java.util.TreeSet;

import Util.Assert;
import Util.Strings;
import Util.Collections.IndexMap;
import Util.Collections.Pair;
import Util.Collections.UnmodifiableIterator;
import Util.IO.Textualizable;
import Util.IO.Textualizer;

/**
 * PathNumbering
 * 
 * @author John Whaley
 * @version $Id$
 */
public class PathNumbering {

    public static final boolean PRINT_BIGGEST = false;
    public static final boolean TRACE_NUMBERING = false;
    public static final boolean TRACE_PATH = false;
    public static final boolean VERIFY_ASSERTIONS = false;

    public static class Range implements Textualizable {
        public Number low, high;
        public Range(Number l, Number h) {
            this.low = l; this.high = h;
        }
        public Range(Number l, BigInteger h) {
            this.low = l; this.high = fromBigInt(h);
        }
        public Range(BigInteger l, Number h) {
            this.low = fromBigInt(l); this.high = h;
        }
        public Range(BigInteger l, BigInteger h) {
            this.low = fromBigInt(l); this.high = fromBigInt(h);
        }
        public String toString() {
            return "<"+low+','+high+'>';
        }
        public boolean equals(Range r) {
            return low.equals(r.low) && high.equals(r.high);
        }
        public boolean equals(Object o) {
            try {
                return equals((Range) o);
            } catch (ClassCastException x) {
                return false;
            }
        }
        public int hashCode() {
            return low.hashCode() ^ high.hashCode();
        }
        public void write(Textualizer t) throws IOException {
            t.writeBytes(low+" "+high);
        }
        public void writeEdges(Textualizer t) throws IOException { }
        public void addEdge(String s, Textualizable t) { }
        public static Range read(StringTokenizer st) {
            long lo = Long.parseLong(st.nextToken());
            long hi = Long.parseLong(st.nextToken());
            return new Range(new Long(lo), new Long(hi));
        }
    }
    
    public PathNumbering() {}
    
    public PathNumbering(Selector s) {
        this.selector = s;
    }
    
    public PathNumbering(Graph g) {
        countPaths(g);
    }
    
    /** Navigator for the graph. */
    Navigator navigator;
    
    /** Cache of topologically-sorted SCC graph. */
    SCCTopSortedGraph graph;

    /** Map from a node to its SCC. */
    Map nodeToScc = new HashMap();
    
    /** Map from SCCs to Ranges. */
    Map sccNumbering = new HashMap();
    
    /** Map from pairs of SCCs to the set of edges between them. */
    Map sccEdges = new HashMap();
    
    /** Map from edges to ranges. */
    Map edgeNumbering = new HashMap();
    
    public interface Selector {
        /**
         * Return true if the edge scc1->scc2 is important.
         */
        boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num);
    }
    
    /** Select important edges. */
    Selector selector;
    
    /** Converts the given Number to BigInteger representation. */
    public static BigInteger toBigInt(Number n) {
        if (n instanceof BigInteger) return (BigInteger) n;
        else return BigInteger.valueOf(n.longValue());
    }

    /** Converts the given BigInteger to a potentially smaller Number representation. */
    public static Number fromBigInt(BigInteger n) {
        int bits = n.bitLength();
        if (bits < 32) return new Integer(n.intValue());
        if (bits < 64) return new Long(n.longValue());
        return n;
    }

    /** Counts the number of paths in the given graph. */
    public Number countPaths(Graph graph) {
        return countPaths(graph.getRoots(), graph.getNavigator(), null);
    }
    
    Set unimportant = new HashSet();
    
    /** Counts the number of paths from the given root set, using the given graph navigator. */
    public Number countPaths(Collection roots, Navigator navigator, Map initialMap) {
        BigInteger max_paths = BigInteger.ZERO;
        
        int max_scc = 0;
        int num_scc = 0;
        
        if (TRACE_NUMBERING) System.out.print("Building and sorting SCCs...");
        Set sccs = SCComponent.buildSCC(roots, navigator);
        this.navigator = navigator;
        graph = SCCTopSortedGraph.topSort(sccs);
        if (TRACE_NUMBERING) System.out.print("done.");
        
        SCComponent scc = graph.getFirst();
        while (scc != null) {
            initializeSccMap(scc);
            max_scc = Math.max(scc.getId(), max_scc);
            scc = scc.nextTopSort();
            ++num_scc;
        }
        if (TRACE_NUMBERING) System.out.println("Max SCC="+max_scc+", Num SCC="+num_scc);
        
        /* Walk through SCCs in forward order. */
        scc = graph.getFirst();
        if (VERIFY_ASSERTIONS) Assert._assert(scc.prevLength() == 0);
        while (scc != null) {
            /* Assign a number for each SCC. */
            if (TRACE_NUMBERING)
                System.out.println("Visiting SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)"));
            BigInteger total = BigInteger.ZERO;
            if (initialMap != null && !scc.isLoop())
                total = toBigInt((Number) initialMap.get(scc.nodes()[0]));
            recordEdgesFromSCC(scc);
            for (Iterator i=Arrays.asList(scc.prev()).iterator(); i.hasNext(); ) {
                SCComponent pred = (SCComponent) i.next();
                Pair edge = new Pair(pred, scc);
                //System.out.println("Visiting edge SCC"+pred.getId()+" to SCC"+scc.getId());
                int nedges = ((Collection) sccEdges.get(edge)).size();
                Range r = (Range) sccNumbering.get(pred);
                // t1 = r.high+1;
                BigInteger t1 = toBigInt(r.high).add(BigInteger.ONE);
                // newtotal = total + t1*nedges
                BigInteger newtotal = total.add(t1.multiply(BigInteger.valueOf(nedges)));
                if (isImportant(pred, scc, newtotal)) {
                    total = newtotal;
                } else {
                    unimportant.add(edge);
                    if (total.compareTo(t1) < 0) total = t1;
                }
            }
            if (scc.prevLength() == 0)
                total = total.add(BigInteger.ONE);
            Range r = new Range(fromBigInt(total), fromBigInt(total.subtract(BigInteger.ONE)));
            if (TRACE_NUMBERING ||
                (PRINT_BIGGEST && total.compareTo(max_paths) == 1))
                System.out.println("Paths to SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)")+"="+total);
            sccNumbering.put(scc, r);
            max_paths = max_paths.max(total);
            scc = scc.nextTopSort();
        }
        
        scc = graph.getFirst();
        while (scc != null) {
            addEdges(scc);
            scc = scc.nextTopSort();
        }
        
        scc = graph.getFirst();
        while (scc != null) {
            Range r = (Range) sccNumbering.get(scc);
            if (TRACE_NUMBERING) System.out.println("Range for SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)")+"="+r);
            if (r.low.longValue() != 0L) {
                r.low = new Integer(0);
            }
            scc = scc.nextTopSort();
        }
        
        return max_paths;
    }

    /** Initialize the mapping from nodes to their SCCs. */
    private void initializeSccMap(SCComponent scc1) {
        Object[] nodes1 = scc1.nodes();
        for (int i=0; i<nodes1.length; ++i) {
            Object node = nodes1[i];
            nodeToScc.put(node, scc1);
        }
    }
    
    /** Record the outgoing edges between nodes in the given SCC. */
    private void recordEdgesFromSCC(SCComponent scc1) {
        Object[] nodes = scc1.nodes();
        for (int i=0; i<nodes.length; ++i) {
            Object exit = nodes[i];
            Collection targets = navigator.next(exit);
            for (Iterator j=targets.iterator(); j.hasNext(); ) {
                Object target = j.next();
                SCComponent scc2 = (SCComponent) nodeToScc.get(target);
                Pair edge = new Pair(scc1, scc2);
                if (TRACE_NUMBERING) System.out.println("Edge SCC"+scc1.getId()+" to SCC"+scc2.getId()+": "+target);
                Collection value = (Collection) sccEdges.get(edge);
                if (value == null) sccEdges.put(edge, value = new LinkedList());
                value.add(new Pair(exit, target));
            }
        }
    }
    
    private void addEdges(SCComponent scc1) {
        if (TRACE_NUMBERING) System.out.println("Adding edges SCC"+scc1.getId());
        Range r1 = (Range) sccNumbering.get(scc1);
        if (scc1.prevLength() == 0) {
            if (TRACE_NUMBERING) System.out.println("SCC"+scc1.getId()+" is in the root set");
            if (VERIFY_ASSERTIONS) Assert._assert(r1.low.longValue() == 1L && r1.high.longValue() == 0L);
            r1.low = new Integer(0);
        }
        if (scc1.isLoop()) {
            Collection internalEdges = (Collection) sccEdges.get(new Pair(scc1, scc1));
            for (Iterator i = internalEdges.iterator(); i.hasNext(); ) {
                Pair edge = (Pair) i.next();
                if (TRACE_NUMBERING) System.out.println("Range for "+edge+" = "+r1+" "+Strings.hex(r1));
                edgeNumbering.put(edge, r1);
            }
        }
        for (Iterator i=Arrays.asList(scc1.next()).iterator(); i.hasNext(); ) {
            SCComponent scc2 = (SCComponent) i.next();
            Pair e = new Pair(scc1, scc2);
            boolean important = !unimportant.contains(e);
            Range r2 = (Range) sccNumbering.get(scc2);
            Collection calls = (Collection) sccEdges.get(e);
            for (Iterator k=calls.iterator(); k.hasNext(); ) {
                Pair edge = (Pair) k.next();
                Range r3;
                if (important) {
                    // external call. update internal object and make new object.
                    BigInteger newlow = toBigInt(r2.low).subtract(toBigInt(r1.high).add(BigInteger.ONE));
                    Assert._assert(newlow.signum() != -1);
                    r2.low = fromBigInt(newlow);
                    if (TRACE_NUMBERING) System.out.println("External edge!  New range for SCC"+scc2.getId()+" = "+r2);
                    r3 = new Range(newlow, newlow.add(toBigInt(r1.high)));
                } else {
                    // unimportant external call. don't update internal object.
                    BigInteger newlow = BigInteger.ZERO;
                    r3 = new Range(newlow, newlow.add(toBigInt(r1.high)));
                }
                if (TRACE_NUMBERING) System.out.println("Range for "+edge+" = "+r3+" "+Strings.hex(r3));
                edgeNumbering.put(edge, r3);
            }
        }
    }
    
    public boolean isImportant(SCComponent scc1, SCComponent scc2, BigInteger num) {
        if (selector == null) return true;
        return selector.isImportant(scc1, scc2, num);
    }
    
    public Range getRange(Object o) {
        SCComponent scc = (SCComponent) nodeToScc.get(o);
        return getSCCRange(scc);
    }
    
    public Range getSCCRange(SCComponent scc) {
        Range r = (Range) sccNumbering.get(scc);
        return r;
    }
    
    public Number numberOfPathsTo(Object o) {
        SCComponent scc = (SCComponent) nodeToScc.get(o);
        return numberOfPathsToSCC(scc);
    }
    
    public Number numberOfPathsToSCC(SCComponent scc) {
        Range r = (Range) sccNumbering.get(scc);
        Number n = fromBigInt(toBigInt(r.high).add(BigInteger.ONE));
        return n;
    }
    
    public Range getEdge(Object from, Object to) {
        return getEdge(new Pair(from, to));
    }
    
    public Range getEdge(Pair edge) {
        Range r = (Range) edgeNumbering.get(edge);
        return r;
    }
    
    public Collection/*<Range>*/ getSCCEdges(SCComponent from, SCComponent to) {
        return getSCCEdges(new Pair(from, to));
    }
    
    public Collection/*<Range>*/ getSCCEdges(Pair edge) {
        Collection c = (Collection) sccEdges.get(edge);
        return c;
    }
    
    public SCComponent getSCC(Object node) {
        return (SCComponent) nodeToScc.get(node);
    }
    
    /** Comparator used to put nodes in post order according to the SCC post order. */
    static class PostOrderComparator implements Comparator {

        private final Map nodeToScc;
        private final IndexMap postOrderNumberingOfSccs;
        
        /** Construct a post-order comparator with the given node-to-scc mapping and
         * topologically-sorted SCC graph.
         */
        public PostOrderComparator(Map nodeToScc, SCCTopSortedGraph g) {
            this.nodeToScc = nodeToScc;
            List list = g.list();
            postOrderNumberingOfSccs = new IndexMap("PostOrderNumbering", list.size());
            postOrderNumberingOfSccs.addAll(list);
        }
        
        /* (non-Javadoc)
         * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
         */
        public int compare(Object arg0, Object arg1) {
            if (arg0.equals(arg1)) return 0;
            SCComponent scc0 = (SCComponent) nodeToScc.get(arg0);
            SCComponent scc1 = (SCComponent) nodeToScc.get(arg1);
            int a = postOrderNumberingOfSccs.get(scc0);
            int b = postOrderNumberingOfSccs.get(scc1);
            if (a < b) return -1;
            if (a > b) return 1;
            // in the same SCC.
            a = Arrays.asList(scc0.nodes()).indexOf(arg0);
            b = Arrays.asList(scc1.nodes()).indexOf(arg1);
            if (a < b) return -1;
            if (VERIFY_ASSERTIONS) Assert._assert(a != b);
            return 1;
        }
    }
    
    /** Represents a path through the graph as an immutable linked structure. */
    public static class Path extends AbstractList {

        private final Object o;
        private final int length;
        private final Path next;

        /** Construct a path with exactly one element: the given one. */
        public Path(Object o) {
            this.o = o;
            this.length = 1;
            this.next = null;
        }

        /** Construct a path by prepending an element to an existing path. */
        public Path(Object o, Path next) {
            this.o = o;
            this.length = next.length+1;
            this.next = next;
        }
        
        /** Return the length of this path. */
        public int size() {
            return this.length;
        }
        
        /** Return a certain element of this path. */
        public Object get(int i) {
            Path p = this;
            for (;;) {
                if (i == 0) return p.o;
                p = p.next;
                --i;
            }
        }
        
        /* (non-Javadoc)
         * @see java.util.Collection#iterator()
         */
        public Iterator iterator() {
            return new UnmodifiableIterator() {
                Path p = Path.this;

                public boolean hasNext() {
                    return p != null;
                }

                public Object next() {
                    Object o = p.o;
                    p = p.next;
                    return o;
                }
            };
        }
        
        /** Return a string representation of this path. */
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("\t<");
            sb.append(o);
            Path p = next;
            while (p != null) {
                sb.append(',');
                sb.append(Strings.lineSep);
                sb.append('\t');
                sb.append(p.o);
                p = p.next;
            }
            sb.append('>');
            return sb.toString();
        }
    }
    
    public Path getPath(Object callee, Number context) {
        BigInteger c = toBigInt(context);
        Range range = (Range) sccNumbering.get(nodeToScc.get(callee));
        if (c.compareTo(toBigInt(range.high)) > 0) {
            if (TRACE_PATH) System.out.println("Out of range (high)");
            return null;
        }
        
        Path result = new Path(callee);
        
        // visit SCCs in post order.
        PostOrderComparator po_comparator = new PostOrderComparator(nodeToScc, graph);
        SortedSet worklist = new TreeSet(po_comparator);
        Map contexts = new HashMap();
        Map results = new HashMap();
        
        worklist.add(callee);
        contexts.put(callee, c);
        results.put(callee, result);
        
        while (!worklist.isEmpty()) {
            callee = worklist.last(); worklist.remove(callee);
            c = (BigInteger) contexts.get(callee);
            result = (Path) results.get(callee);
            if (TRACE_PATH) System.out.println("Getting context "+c+" at "+callee);
            for (Iterator i=navigator.prev(callee).iterator(); i.hasNext(); ) {
                Object caller = i.next();
                Range r = getEdge(caller, callee);
                if (TRACE_PATH) System.out.println("Edge "+caller+" to "+callee+": "+r);
                if (c.compareTo(toBigInt(r.high)) > 0) {
                    if (TRACE_PATH) System.out.println("Out of range (high)");
                    continue;
                }
                BigInteger c2 = c.subtract(toBigInt(r.low));
                if (c2.signum() < 0) {
                    if (TRACE_PATH) System.out.println("Out of range (low)");
                    continue;
                }
                if (contexts.containsKey(caller)) {
                    // recursive cycle.
                    if (TRACE_PATH) System.out.println("Recursive cycle");
                } else {
                    if (TRACE_PATH) System.out.println("Matches! Adding to worklist.");
                    worklist.add(caller);
                    contexts.put(caller, c2);
                    results.put(caller, new Path(caller, result));
                }
            }
        }
        return result;
    }
    
    public SCCTopSortedGraph getSCCGraph() {
        return graph;
    }
    
    public static PathNumbering read(DataInput in) throws IOException {
        HashMap idToSCC = new HashMap();
        PathNumbering p = new PathNumbering();
        String s;
        StringTokenizer st;
        s = in.readLine();
        Textualizer t = new Textualizer.Map(in);
        int nodeToScc_size = Integer.parseInt(s);
        for (int i = 0; i < nodeToScc_size; ++i) {
            s = in.readLine();
            st = new StringTokenizer(s);
            Textualizable t2 = t.readObject();
            s = st.nextToken();
            Assert._assert(s.equals("->"));
            s = st.nextToken();
            Assert._assert(s.startsWith("SCC"));
            int id = Integer.parseInt(s.substring(3));
            Integer key = new Integer(id);
            SCComponent scc = (SCComponent) idToSCC.get(key);
            if (scc == null) { idToSCC.put(key, scc = new SCComponent()); }
            Object result = p.nodeToScc.put(t2, scc);
            Assert._assert(result == null);
        }
        
        s = in.readLine();
        int sccNumbering_size = Integer.parseInt(s);
        for (int i = 0; i < sccNumbering_size; ++i) {
            s = in.readLine();
            st = new StringTokenizer(s);
            s = st.nextToken();
            Assert._assert(s.startsWith("SCC"));
            int id = Integer.parseInt(s.substring(3));
            Integer key = new Integer(id);
            SCComponent scc = (SCComponent) idToSCC.get(key);
            Assert._assert(scc != null);
            s = st.nextToken();
            Assert._assert(s.equals("Range"));
            Range r = Range.read(st);
            Object result = p.sccNumbering.put(scc, r);
            Assert._assert(result == null);
        }
        
        s = in.readLine();
        int sccEdges_size = Integer.parseInt(s);
        for (int i = 0; i < sccEdges_size; ++i) {
            s = in.readLine();
            st = new StringTokenizer(s);
            s = st.nextToken();
            Assert._assert(s.startsWith("SCC"));
            int id1 = Integer.parseInt(s.substring(3));
            Integer key1 = new Integer(id1);
            SCComponent scc1 = (SCComponent) idToSCC.get(key1);
            Assert._assert(scc1 != null);
            s = st.nextToken();
            Assert._assert(s.startsWith("SCC"));
            int id2 = Integer.parseInt(s.substring(3));
            Integer key2 = new Integer(id2);
            SCComponent scc2 = (SCComponent) idToSCC.get(key2);
            Assert._assert(scc2 != null);
            Pair p1 = new Pair(scc1, scc2);
            s = st.nextToken();
            int edges_size = Integer.parseInt(s);
            HashSet set = new HashSet();
            for (int j = 0; j < edges_size; ++j) {
                Textualizable t1 = t.readObject();
                Textualizable t2 = t.readObject();
                Pair p2 = new Pair(t1, t2);
                set.add(p2);
            }
            p.sccEdges.put(p1, set);
        }
        
        s = in.readLine();
        int edgeNumbering_size = Integer.parseInt(s);
        for (int i = 0; i < edgeNumbering_size; ++i) {
            s = in.readLine();
            st = new StringTokenizer(s);
            Textualizable t1 = t.readObject();
            Textualizable t2 = t.readObject();
            Pair p1 = new Pair(t1, t2);
            Range r = Range.read(st);
            p.edgeNumbering.put(p1, r);
        }
        return p;
    }
    
    public void dump(DataOutput ou) throws IOException {
        IndexMap sccs = new IndexMap("SCCs");
        sccs.addAll(nodeToScc.values());
        Textualizer t = new Textualizer.Map(ou, sccs);
        t.writeBytes(nodeToScc.size()+"\n");
        for (Iterator i = nodeToScc.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            Textualizable o = (Textualizable) e.getKey();
            o.write(t);
            SCComponent scc = (SCComponent) e.getValue();
            t.writeBytes(" -> SCC"+scc.getId()+"\n");
        }
        t.writeBytes(sccNumbering.size()+"\n");
        for (Iterator i = sccNumbering.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            SCComponent scc = (SCComponent) e.getKey();
            Range r = (Range) e.getValue();
            t.writeBytes("SCC"+scc.getId()+" Range ");
            r.write(t);
            t.writeBytes("\n");
        }
        t.writeBytes(sccEdges.size()+"\n");
        for (Iterator i = sccEdges.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            Pair p = (Pair) e.getKey();
            SCComponent scc1 = (SCComponent) p.left;
            SCComponent scc2 = (SCComponent) p.right;
            Collection edges = (Collection) e.getValue();
            t.writeBytes("SCC"+scc1.getId()+" SCC"+scc2.getId()+" "+edges.size());
            for (Iterator j = edges.iterator(); j.hasNext(); ) {
                Pair p2 = (Pair) j.next();
                t.writeBytes(" ");
                Textualizable t1 = (Textualizable) p2.left;
                t1.write(t);
                t.writeBytes(" ");
                Textualizable t2 = (Textualizable) p2.right;
                t2.write(t);
            }
            t.writeBytes("\n");
        }
        t.writeBytes(edgeNumbering.size()+"\n");
        for (Iterator i = sccEdges.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            Pair p = (Pair) e.getKey();
            Textualizable t1 = (Textualizable) p.left;
            t1.write(t);
            t.writeBytes(" ");
            Textualizable t2 = (Textualizable) p.right;
            t2.write(t);
            t.writeBytes(" ");
            Range r = (Range) e.getValue();
            r.write(t);
            t.writeBytes("\n");
        }
    }
    
    public void dotGraph(DataOutput out) throws IOException {
        out.writeBytes("digraph \"PathNumbering\" {\n");
        IndexMap m = new IndexMap("NodeMap");
        for (Iterator i=nodeToScc.keySet().iterator(); i.hasNext(); ) {
            Object o = i.next();
            int j = m.get(o);
            out.writeBytes("n"+j+" [label=\""+o+"\"];\n");
        }
        for (Iterator i=edgeNumbering.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            Pair edge = (Pair) e.getKey();
            Range r = (Range) e.getValue();
            int fromIndex = m.get(edge.left);
            int toIndex = m.get(edge.right);
            out.writeBytes("n"+fromIndex+" -> n"+toIndex+" [label=\""+r+"\"];\n");
        }
        out.writeBytes("}\n");
    }
    
}

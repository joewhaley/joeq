// PathNumbering.java, created Aug 16, 2003 1:49:33 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Graphs;

import java.io.DataOutput;
import java.io.IOException;
import java.math.BigInteger;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import Util.Assert;
import Util.Strings;
import Util.Collections.IndexMap;
import Util.Collections.Pair;
import Util.Collections.UnmodifiableIterator;

/**
 * PathNumbering
 * 
 * @author John Whaley
 * @version $Id$
 */
public class PathNumbering {

    public static final boolean TRACE_NUMBERING = false;
    public static final boolean TRACE_PATH = false;

    public static class Range {
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
    }
    
    public PathNumbering() {}
    
    public PathNumbering(Graph g) {
        countPaths(g);
    }
    
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

    public Number countPaths(Graph graph) {
        return countPaths(graph.getRoots(), graph.getNavigator());
    }
    
    public Number countPaths(Collection roots, Navigator navigator) {
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
        Assert._assert(scc.prevLength() == 0);
        while (scc != null) {
            /* Assign a number for each SCC. */
            if (TRACE_NUMBERING)
                System.out.println("Visiting SCC"+scc.getId()+(scc.isLoop()?" (loop)":" (non-loop)"));
            BigInteger total = BigInteger.ZERO;
            recordEdgesFromSCC(scc);
            for (Iterator i=Arrays.asList(scc.prev()).iterator(); i.hasNext(); ) {
                SCComponent pred = (SCComponent) i.next();
                Pair edge = new Pair(pred, scc);
                //System.out.println("Visiting edge SCC"+pred.getId()+" to SCC"+scc.getId());
                int nedges = ((Collection)sccEdges.get(edge)).size();
                Range r = (Range) sccNumbering.get(pred);
                BigInteger t1 = toBigInt(r.high).add(BigInteger.ONE);
                BigInteger newtotal = total.add(t1.multiply(BigInteger.valueOf(nedges)));
                total = newtotal;
            }
            if (total.equals(BigInteger.ZERO)) total = BigInteger.ONE;
            Range r = new Range(fromBigInt(total), fromBigInt(total.subtract(BigInteger.ONE)));
            if (TRACE_NUMBERING || total.compareTo(max_paths) == 1)
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
                Assert.UNREACHABLE("SCC"+scc.getId()+" Range="+r);
            }
            scc = scc.nextTopSort();
        }
        
        return max_paths;
    }

    private void initializeSccMap(SCComponent scc1) {
        Object[] nodes1 = scc1.nodes();
        for (int i=0; i<nodes1.length; ++i) {
            Object node = nodes1[i];
            nodeToScc.put(node, scc1);
        }
    }
    
    private void recordEdgesFromSCC(SCComponent scc1) {
        Object[] nodes = scc1.nodes();
        int total = 0;
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
        Object[] nodes1 = scc1.nodes();
        Range r1 = (Range) sccNumbering.get(scc1);
        if (scc1.prevLength() == 0) {
            if (TRACE_NUMBERING) System.out.println("SCC"+scc1.getId()+" is in the root set");
            Assert._assert(r1.low.longValue() == 1L && r1.high.longValue() == 0L);
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
            Range r2 = (Range) sccNumbering.get(scc2);
            Collection calls = (Collection) sccEdges.get(new Pair(scc1, scc2));
            for (Iterator k=calls.iterator(); k.hasNext(); ) {
                Pair edge = (Pair) k.next();
                // external call. update internal object and make new object.
                BigInteger newlow = toBigInt(r2.low).subtract(toBigInt(r1.high).add(BigInteger.ONE));
                r2.low = fromBigInt(newlow);
                if (TRACE_NUMBERING) System.out.println("External edge!  New range for SCC"+scc2.getId()+" = "+r2);
                Range r3 = new Range(newlow, newlow.add(toBigInt(r1.high)));
                if (TRACE_NUMBERING) System.out.println("Range for "+edge+" = "+r3+" "+Strings.hex(r3));
                edgeNumbering.put(edge, r3);
            }
        }
    }
    
    public Range getRange(Object o) {
        SCComponent scc = (SCComponent) nodeToScc.get(o);
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
    
    public SCComponent getSCC(Object node) {
        return (SCComponent) nodeToScc.get(node);
    }
    
    public static class PostOrderComparator implements Comparator {

        final Map nodeToScc;
        final IndexMap postOrderNumbering;
        
        public PostOrderComparator(Map nodeToScc, SCCTopSortedGraph g) {
            this.nodeToScc = nodeToScc;
            List list = g.list();
            postOrderNumbering = new IndexMap("PostOrderNumbering", list.size());
            postOrderNumbering.addAll(list);
        }
        
        /* (non-Javadoc)
         * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
         */
        public int compare(Object arg0, Object arg1) {
            if (arg0.equals(arg1)) return 0;
            SCComponent scc0 = (SCComponent) nodeToScc.get(arg0);
            SCComponent scc1 = (SCComponent) nodeToScc.get(arg1);
            int a = postOrderNumbering.get(scc0);
            int b = postOrderNumbering.get(scc1);
            if (a < b) return -1;
            if (a > b) return 1;
            // in the same SCC.
            a = Arrays.asList(scc0.nodes()).indexOf(arg0);
            b = Arrays.asList(scc1.nodes()).indexOf(arg1);
            if (a < b) return -1;
            Assert._assert(a != b);
            return 1;
        }
    }
    
    public static class Path extends AbstractList {

        final Object o;
        final int length;
        final Path next;

        public Path(Object o) {
            this.o = o;
            this.length = 1;
            this.next = null;
        }

        public Path(Object o, Path next) {
            this.o = o;
            this.length = next.length+1;
            this.next = next;
        }
        
        public int size() {
            return this.length;
        }
        
        public Object get(int i) {
            Path p = this;
            for (;;) {
                if (i == 0) return p.o;
                p = p.next;
                --i;
            }
        }
        
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
            boolean found = false, any = false;
            for (Iterator i=navigator.prev(callee).iterator(); i.hasNext(); ) {
                Object caller = i.next();
                Range r = getEdge(caller, callee);
                any = true;
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
                    found = true;
                }
            }
        }
        return result;
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

// GlobalPathNumbering.java, created Aug 4, 2004 8:56:01 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Graphs;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.math.BigInteger;
import joeq.Util.Assert;
import joeq.Util.Collections.Pair;

/**
 * GlobalPathNumbering
 * 
 * @author jwhaley
 * @version $Id$
 */
public class GlobalPathNumbering extends PathNumbering {
    
    /** Navigator for the graph. */
    Navigator navigator;
    
    /** Map from nodes to numbers. */
    Map nodeNumbering = new HashMap();
    
    /** Map from edges to ranges. */
    Map edgeNumbering = new HashMap();
    
    /* (non-Javadoc)
     * @see joeq.Util.Graphs.PathNumbering#countPaths(java.util.Collection, joeq.Util.Graphs.Navigator, java.util.Map)
     */
    public BigInteger countPaths(Collection roots, Navigator navigator, Map initialMap) {
        for (Iterator i = roots.iterator(); i.hasNext(); ) {
            Object o = i.next();
            BigInteger total = BigInteger.ONE;
            if (initialMap != null)
                total = toBigInt((Number) initialMap.get(o));
            nodeNumbering.put(o, total);
        }
        BigInteger max = BigInteger.ZERO;
        Iterator rpo = Traversals.reversePostOrder(navigator, roots).iterator();
        while (rpo.hasNext()) {
            Object o = rpo.next();
            BigInteger val = (BigInteger) nodeNumbering.get(o);
            Collection prev = navigator.prev(o);
            if (prev.size() == 0 && val == null) {
                Assert.UNREACHABLE("Missing root! "+o);
            }
            if (val == null) val = BigInteger.ONE;
            for (Iterator i = prev.iterator(); i.hasNext(); ) {
                Object p = i.next();
                BigInteger val2 = (BigInteger) nodeNumbering.get(p);
                if (val2 == null) {
                    //System.out.println("Loop edge: "+p+" -> "+o);
                    nodeNumbering.put(p, val2 = BigInteger.ONE);
                }
                BigInteger val3 = val.add(val2);
                Object edge = new Pair(p, o);
                Range range = new Range(val, val3.subtract(BigInteger.ONE));
                edgeNumbering.put(edge, range);
                //System.out.println("Putting Edge ("+edge+") = "+range);
                val = val3;
            }
            nodeNumbering.put(o, val);
            if (val.compareTo(max) > 0) max = val;
        }
        return max;
    }

    /* (non-Javadoc)
     * @see joeq.Util.Graphs.PathNumbering#getRange(java.lang.Object)
     */
    public Range getRange(Object o) {
        BigInteger b = (BigInteger) nodeNumbering.get(o);
        //System.out.println("Node ("+o+") = "+b);
        //if (b == null) b = BigInteger.ZERO;
        return new Range(BigInteger.ZERO, b.subtract(BigInteger.ONE));
    }

    /* (non-Javadoc)
     * @see joeq.Util.Graphs.PathNumbering#getEdge(java.lang.Object, java.lang.Object)
     */
    public Range getEdge(Object from, Object to) {
        Object key = new Pair(from, to);
        Range r = (Range) edgeNumbering.get(key);
        //System.out.println("Edge ("+key+") = "+r);
        return r;
    }
    
}
// Pair.java, created Wed Mar  5  0:26:26 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Collections;

import java.io.IOException;
import java.io.Serializable;
import java.util.AbstractList;

import joeq.Util.IO.Textualizable;
import joeq.Util.IO.Textualizer;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Pair extends AbstractList implements Serializable, Textualizable {
    public Object left, right;
    public Pair(Object left, Object right) {
        this.left = left; this.right = right;
    }
    public int size() { return 2; }
    public Object get(int index) {
        switch(index) {
        case 0: return this.left;
        case 1: return this.right;
        default: throw new IndexOutOfBoundsException();
        }
    }
    public Object set(int index, Object element) {
        Object prev;
        switch(index) {
        case 0: prev=this.left; this.left=element; return prev;
        case 1: prev=this.right; this.right=element; return prev;
        default: throw new IndexOutOfBoundsException();
        }
    }
    /* (non-Javadoc)
     * @see Util.IO.Textualizable#write(Util.IO.Textualizer)
     */
    public void write(Textualizer t) throws IOException {
    }
    /* (non-Javadoc)
     * @see Util.IO.Textualizable#writeEdges(Util.IO.Textualizer)
     */
    public void writeEdges(Textualizer t) throws IOException {
        t.writeEdge("left", (Textualizable) left);
        t.writeEdge("right", (Textualizable) right);
    }
    /* (non-Javadoc)
     * @see Util.IO.Textualizable#addEdge(java.lang.String, Util.IO.Textualizable)
     */
    public void addEdge(String edge, Textualizable t) {
        if (edge.equals("left"))
            this.left = t;
        else if (edge.equals("right"))
            this.right = t;
        else
            throw new InternalError();
    }
}

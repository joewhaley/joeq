// IndexMap.java, created Jun 15, 2003 2:04:05 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Collections;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;

import Util.IO.Textualizable;
import Util.IO.Textualizer;

/**
 * IndexMap
 * 
 * @author John Whaley
 * @version $Id$
 */
public class IndexMap implements IndexedMap {
    private final String name;
    private final HashMap hash;
    private final ArrayList list;
    private final boolean trace;
    
    public IndexMap(String name) {
        this.name = name;
        hash = new HashMap();
        list = new ArrayList();
        trace = false;
    }
    
    public IndexMap(String name, int size) {
        this.name = name;
        hash = new HashMap(size);
        list = new ArrayList(size);
        trace = false;
    }
    
    public IndexMap(String name, int size, boolean t) {
        this.name = name;
        hash = new HashMap(size);
        list = new ArrayList(size);
        trace = t;
    }
    
    public int get(Object o) {
        Integer i = (Integer) hash.get(o);
        if (i == null) {
            hash.put(o, i = new Integer(list.size()));
            list.add(o);
            if (trace) System.out.println(this+"["+i+"] = "+o);
        }
        return i.intValue();
    }
        
    public Object get(int i) {
        return list.get(i);
    }
        
    public boolean contains(Object o) {
        return hash.containsKey(o);
    }
        
    public int size() {
        return list.size();
    }
        
    public String toString() {
        return name;
    }
    
    public Iterator iterator() {
        return list.iterator();
    }
    
    public boolean addAll(Collection c) {
        int before = size();
        for (Iterator i=c.iterator(); i.hasNext(); ) {
            get(i.next());
        }
        return before != size();
    }
    
    public void dump(final DataOutput out) throws IOException {
        Textualizer t = new Textualizer.Map(out, this);
        t.writeBytes(size()+"\n");
        for (int j = 0; j < size(); ++j) {
            Textualizable o = (Textualizable) get(j);
            if (o == null) {
                t.writeBytes("null");
            } else {
                t.writeObject(o);
            }
            t.writeBytes("\n");
            ++j;
        }
    }
}

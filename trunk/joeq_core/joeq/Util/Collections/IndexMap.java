// IndexMap.java, created Jun 15, 2003 2:04:05 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Collections;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

/**
 * IndexMap
 * 
 * @author John Whaley
 * @version $Id$
 */
public class IndexMap {
    private final String name;
    private final HashMap hash;
    private final ArrayList list;
    
    public IndexMap(String name) {
        this.name = name;
        hash = new HashMap();
        list = new ArrayList();
    }
    
    public IndexMap(String name, int size) {
        this.name = name;
        hash = new HashMap(size);
        list = new ArrayList(size);
    }
    
    public int get(Object o) {
        Integer i = (Integer) hash.get(o);
        if (i == null) {
            hash.put(o, i = new Integer(list.size()));
            list.add(o);
            if (false) System.out.println(this+"["+i+"] = "+o);
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
    
}

// IndexMap.java, created Sep 20, 2003 2:04:05 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Collections;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

/**
 * Interface for an indexed map.  An indexed map provides a mapping
 * between elements and (integer) indices.
 * 
 * @author jwhaley
 * @version $Id$
 */
public interface IndexedMap {

    int get(Object o);
    Object get(int i);
    boolean contains(Object o);
    Iterator iterator();
    int size();
    void dump(final DataOutput out) throws IOException;
    
}
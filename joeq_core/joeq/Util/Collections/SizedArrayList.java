// SizedArrayList.java, created Aug 3, 2004 8:44:46 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Collections;

import java.util.AbstractList;

/**
 * SizedArrayList
 * 
 * @author jwhaley
 * @version $Id$
 */
public class SizedArrayList extends AbstractList {
    
    Object[] array;
    int size;
    
    public SizedArrayList(Object[] array, int size) {
        this.array = array;
        this.size = size;
    }

    /* (non-Javadoc)
     * @see java.util.AbstractList#get(int)
     */
    public Object get(int index) {
        if (index >= size)
            throw new IndexOutOfBoundsException(index+" >= "+size);
        return array[index];
    }

    /* (non-Javadoc)
     * @see java.util.AbstractCollection#size()
     */
    public int size() {
        return size;
    }
    
}

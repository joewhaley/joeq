// MemUnitComparator.java, created Mon Nov 25  9:05:34 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Allocator;

import java.util.Comparator;

/**
 * MemUnitComparator
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
public class MemUnitComparator implements Comparator {
    public int compare(Object o1, Object o2) {
        if(!(o1 instanceof MemUnit && o2 instanceof MemUnit)) {
            throw new ClassCastException();
        } else {
            return (((MemUnit)o1).getByteLength() - ((MemUnit)o2).getByteLength());
        }
    }
}

// WorstFitStrategy.java, created Mon Mar 17  2:03:41 2003 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Allocator;

import java.util.Collection;
import java.util.TreeSet;

/**
 * Worst Fit Strategy
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
public class WorstFitStrategy implements FreeMemStrategy {
    private TreeSet freePool;

    public WorstFitStrategy() {
        freePool = new TreeSet(new MemUnitComparator());
    }

    public void addFreeMem(MemUnit unit) {
        freePool.add(unit);
    }

    public void addCollection(Collection c) {
        freePool.addAll(c);
    }

    public MemUnit getFreeMem(int size) {
        return (MemUnit) (freePool.last());
    }
}

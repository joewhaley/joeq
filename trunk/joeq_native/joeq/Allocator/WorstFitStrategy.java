/**
 * Worst Fit Strategy
 *
 * Created on Nov 26, 2002, 11:03:57 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package Allocator;

import java.util.Collection;
import java.util.TreeSet;

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

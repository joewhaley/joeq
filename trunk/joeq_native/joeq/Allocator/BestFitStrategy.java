/**
 * Best Fit Strategy
 *
 * Created on Nov 26, 2002, 11:03:57 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package Allocator;

import java.util.Collection;
import java.util.TreeSet;

public class BestFitStrategy implements FreeMemStrategy {
    private TreeSet freePool;

    public BestFitStrategy() {
        freePool = new TreeSet(new MemUnitComparator());
    }

    public void addFreeMem(MemUnit unit) {
        freePool.add(unit);
    }

    public void addCollection(Collection c) {
        freePool.addAll(c);
    }

    public MemUnit getFreeMem(int size) {
        if (false) {
            MemUnit target = new MemUnit(null, size);
            return (MemUnit) (freePool.tailSet(target).first());
        } else {
            // FIXME: circular dependency. the allocation of MemUnit above
            // calls the allocator which calls getFreeMem.
            return null;
        }
    }
}

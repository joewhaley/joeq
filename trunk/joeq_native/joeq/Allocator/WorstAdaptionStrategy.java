/**
 * BestAdaptionStrategy
 *
 * Created on Nov 26, 2002, 11:03:57 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.TreeSet;
import java.util.Collection;

public class WorstAdaptionStrategy implements FreeMemStrategy {
    private TreeSet freePool;

    public WorstAdaptionStrategy() {
        freePool = new TreeSet(new MemUnitComparator());
    }

    public MemUnit next(int size) {
        MemUnit target = new MemUnit(null, size);
        return (MemUnit)freePool.last();
    }

    public void addCollection(Collection c) {
        freePool.addAll(c);
    }
}

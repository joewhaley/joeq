/**
 * BestAdaptionStrategy
 *
 * Created on Nov 26, 2002, 11:03:57 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.Collection;
import java.util.TreeSet;

public class BestAdaptionStrategy implements FreeMemStrategy {
    private TreeSet freePool;

    public BestAdaptionStrategy() {
        freePool = new TreeSet(new MemUnitComparator());
    }

    public MemUnit next(int size) {
        MemUnit target = new MemUnit(null, size);
        return (MemUnit)freePool.tailSet(target).first();
    }

    public void addCollection(Collection c) {
        freePool.addAll(c);
    }
}

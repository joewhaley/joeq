/**
 * FreeMemManager
 *
 * Created on Nov 26, 2002, 10:07:32 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.Comparator;
import java.util.TreeSet;

import GC.GCBitsManager.SweepUnit;
import GC.GCBitsManager.SweepUnitComparator;

public class FreeMemManager {
    private TreeSet freePool;
    private FreeMemStrategy stg;

    public FreeMemManager(FreeMemStrategy stg) {
        this.stg = stg;
        freePool = new TreeSet(new SweepUnitComparator());
    }

    public FreeMemManager() {
        this(new FirstAdaptionStrategy());
    }
}

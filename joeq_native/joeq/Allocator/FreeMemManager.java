/**
 * FreeMemManager
 *
 * Created on Nov 26, 2002, 10:07:32 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.SortedSet;
import java.util.Comparator;
import GC.GCBitsManager.SweepUnit;

public class FreeMemManager implements Comparator {
    private SortedSet freePool;
    private FreeMemStrategy stg;

    public FreeMemManager(FreeMemStrategy stg) {
        this stg = stg;
        freePool
    }

    public int compare(Object o1, Object o2) {
        if(!o1.instanceof(SweepUnit) || !o2.instanceof(SweepUnit)) {
            throws new ClassCastException();
        } else {
            return ((SweepUnit)o1).getHead().difference(((SweepUnit)o2).getHead());
        }
    }

}

/**
 * FreeMemManager
 *
 * Created on Nov 26, 2002, 10:07:32 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.Collection;
import java.util.HashSet;

import Memory.HeapAddress;

public class FreeMemManager {
    private static FreeMemStrategy defaultStrategy = new BestAdaptionStrategy();

    private Collection freePool = new HashSet();
    private FreeMemStrategy stg;

    public FreeMemManager(FreeMemStrategy stg) {
        this.stg = stg;
        stg.addCollection(freePool);
    }

    public FreeMemManager() {
        this(defaultStrategy);
    }

    public void addFreeMem(MemUnit unit) {
        freePool.add(unit);
    }

    public HeapAddress getFreeMem(int size) {
        MemUnit unit = stg.next(size);
        if(unit == null) {
            return null;
        } else {
            HeapAddress addr = unit.getHead();
            unit.setHead((HeapAddress)addr.offset(size));
            return addr;
        }
    }
}

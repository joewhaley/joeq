/**
 * FreeMemManager
 *
 * Created on Nov 26, 2002, 10:07:32 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package Allocator;

import Memory.Address;

public class FreeMemManager {
    private static FreeMemStrategy defaultStrategy = new BestFitStrategy();
    private static FreeMemStrategy strategy = defaultStrategy;

    public static void setFreeMemStrategy(FreeMemStrategy stg) {
        strategy = stg;
    }

    public static void addFreeMem(MemUnit unit) {
        strategy.addFreeMem(unit);
    }

    public static Address getFreeMem(int size) {
        MemUnit unit = strategy.getFreeMem(size);
        if (unit == null) {
            return null;
        } else {
            Address addr = unit.getHead().offset(size);
            int byteLength = unit.getByteLength() - size;
            if (byteLength > 0) {
                strategy.addFreeMem(new MemUnit(addr, byteLength));
            }
            return addr;
        }
    }
}
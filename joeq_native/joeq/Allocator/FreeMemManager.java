// FreeMemManager.java, created Mon Nov 25  6:16:24 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import joeq.Memory.Address;

/**
 * FreeMemManager
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
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

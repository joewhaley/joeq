/**
 * FreeMemStrategy
 *
 * Created on Nov 26, 2002, 10:47:51 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import GC.GCBitsManager.SweepUnit;

public interface FreeMemStrategy {
    public SweepUnit next();
}

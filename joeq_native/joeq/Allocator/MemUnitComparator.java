/**
 * MemUnitComparator
 *
 * Created on Nov 27, 2002, 12:50:52 AM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import java.util.Comparator;

public class MemUnitComparator implements Comparator {
    public int compare(Object o1, Object o2) {
        if(!(o1 instanceof MemUnit && o2 instanceof MemUnit)) {
            throw new ClassCastException();
        } else {
            return (((MemUnit)o1).getByteLength() - ((MemUnit)o2).getByteLength());
        }
    }
}

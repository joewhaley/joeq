/**
 * FreeMemStrategy
 *
 * Created on Nov 26, 2002, 10:47:51 PM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package Allocator;

import java.util.Collection;

public interface FreeMemStrategy {
    public void addCollection(Collection c);
    public void addFreeMem(MemUnit unit);
    public MemUnit getFreeMem(int size);
}

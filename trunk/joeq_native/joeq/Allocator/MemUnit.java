/**
 * MemUnit
 *
 * Created on Nov 27, 2002, 12:49:38 AM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Allocator;

import Memory.HeapAddress;
import Memory.Address;

public class MemUnit {
    private HeapAddress head;
    private int byteLength;

    public MemUnit(HeapAddress head, int byteLength) {
        this.head = head;
        this.byteLength = byteLength;
    }

    public MemUnit(HeapAddress head, Address end) {
        this(head, end.difference(head));
    }

    public HeapAddress getHead() {
        return head;
    }

    public void setHead(HeapAddress head) {
        this.head = head;
    }

    public int getByteLength() {
        return byteLength;
    }

    public void setByteLength(int byteLength) {
        this.byteLength = byteLength;
    }
}

/**
 * MemUnit
 *
 * Created on Nov 27, 2002, 12:49:38 AM
 *
 * @author laudney <laudney@acm.org>
 * @version 0.1
 */
package Allocator;

import Memory.Address;

public class MemUnit {
    private Address head;
    private int byteLength;

    public MemUnit(Address head, int byteLength) {
        this.head = head;
        this.byteLength = byteLength;
    }

    public MemUnit(Address head, Address end) {
        this(head, end.difference(head));
    }

    public Address getHead() {
        return head;
    }

    public void setHead(Address head) {
        this.head = head;
    }

    public int getByteLength() {
        return byteLength;
    }

    public void setByteLength(int byteLength) {
        this.byteLength = byteLength;
    }
}

// MemUnit.java, created Mon Nov 25  9:15:28 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Allocator;

import Memory.Address;

/**
 * MemUnit
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
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

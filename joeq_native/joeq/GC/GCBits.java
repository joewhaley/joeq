// GCBits.java, created Wed Sep 25 20:04:21 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package GC;

import java.util.HashSet;

import Allocator.MemUnit;
import Allocator.SimpleAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Memory.HeapAddress;
import Util.BitString;

/**
 * GCBits
 *
 * PRE-REQUISITE: All the objects and arrays are allocated align8
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
public class GCBits {

    protected static final int bitLength = SimpleAllocator.BLOCK_SIZE / 8;
    protected static final int byteLength = SimpleAllocator.BLOCK_SIZE / 64;

    protected HeapAddress blockHead, blockEnd;

    /**
     * Each bit in allocbits corresponds to 8 bytes on the heap. When an object is
     * allocated and aligned on 8 byte boundary, the bit in allocbits corresponding
     * to the starting 8 bytes (including HEADER) of the object is set.
     */
    private final BitString allocbits = new BitString(bitLength);

    /**
     * Each bit in markbits corresponds to 8 bytes on the heap. When an object
     * is reachable from TraceRootSet, the bit in markbits corresponding to the
     * starting 8 bytes (including HEADER) of the object is set.
     */
    private final BitString markbits = new BitString(bitLength);

    // Never created normally.  Always created with allocateObject_nogc.
    public GCBits(HeapAddress blockHead, HeapAddress blockEnd) {
        this.blockHead = blockHead;
        this.blockEnd = blockEnd;
        GCBitsManager.register(this);
    }

    public void set(HeapAddress addr) {
        allocbits.set((addr.difference(blockHead)) / 8);
    }

    public void unset(HeapAddress addr) {
        allocbits.clear((addr.difference(blockHead)) / 8);
    }

    public boolean isSet(HeapAddress addr) {
        return allocbits.get((addr.difference(blockHead)) / 8);
    }

    public void mark(HeapAddress addr) {
        markbits.set((addr.difference(blockHead)) / 8);
    }

    public void unmark(HeapAddress addr) {
        markbits.clear((addr.difference(blockHead)) / 8);
    }

    public boolean isMarked(HeapAddress addr) {
        return markbits.get((addr.difference(blockHead)) / 8);
    }

    public HashSet diff() {
        BitString sweepbits = (BitString) markbits.clone();
        sweepbits.xor(allocbits);
        BitString.ForwardBitStringIterator iter = sweepbits.iterator();
        HashSet units = new HashSet();
        int i, j;
        while (iter.hasNext()) {
            i = iter.nextIndex(); // head of a sweepable object
            j = allocbits.firstSet(i); // end of a sweepable object
            if (j != -1) {
                units.add(new MemUnit((HeapAddress)blockHead.offset(i * 8), (j - i) * 8));
            } else {
                units.add(new MemUnit((HeapAddress)blockHead.offset(i * 8), blockEnd));
            }
        }
        return units;
    }

    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LGC/GCBits;");
}

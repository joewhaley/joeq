/**
 * GCBits
 *
 * Created on Sep 26, 2002, 9:41:32 AM
 *
 * PRE-REQUISITE: All the objects and arrays are allocated align8
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package GC;

import Util.BitString;
import Allocator.SimpleAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Memory.HeapAddress;

public class GCBits {

    protected static final int bitLength = SimpleAllocator.BLOCK_SIZE / 8;
    protected static final int byteLength = SimpleAllocator.BLOCK_SIZE / 64;

    protected HeapAddress blockHead, blockEnd;

    /**
     * Each bit in allocbits corresponds to 8 bytes on the heap. When an object
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

    public static final jq_Class _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LGC/GCBits;");
}

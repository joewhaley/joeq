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
import Memory.HeapAddress;

public class GCBits {

    protected static final int bitLength = SimpleAllocator.BLOCK_SIZE / 8;
    protected static final int byteLength = SimpleAllocator.BLOCK_SIZE / 64;

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

    private HeapAddress blockHead, blockEnd;

    public GCBits(HeapAddress blockHead, HeapAddress blockEnd) {
        this.blockHead = blockHead;
        this.blockEnd = blockEnd;
    }

    public void set(HeapAddress objectAddr) {
        allocbits.set((objectAddr.difference(blockHead)) / 8);
    }

    public void unset(HeapAddress objectAddr) {
        allocbits.clear((objectAddr.difference(blockHead)) / 8);
    }

    public void mark(HeapAddress objectAddr) {
        markbits.set((objectAddr.difference(blockHead)) / 8);
    }

    public void unmark(HeapAddress objectAddr) {
        markbits.clear((objectAddr.difference(blockHead)) / 8);
    }
}

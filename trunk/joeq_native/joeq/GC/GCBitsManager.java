/**
 * GCBitsManager
 *
 * Created on Sep 27, 2002, 1:46:11 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package GC;

import Allocator.HeapAllocator.HeapPointer;
import Memory.StackAddress;
import Memory.HeapAddress;

import java.util.Comparator;
import java.util.TreeMap;
import java.util.SortedMap;

public class GCBitsManager {

    private static TreeMap pool = new TreeMap();

    public static void register(GCBits newcomer) {
        pool.put(new HeapPointer(newcomer.blockEnd), newcomer);
    }

    // In order for addr to be valid, totally 3 conditions must be met
    public static boolean isValidHeapAddr(HeapAddress addr) {
        // First Condition: addr is within a memory block
        HeapPointer key = (HeapPointer) pool.tailMap(addr).firstKey();
        GCBits value = (GCBits) pool.get(key);
        int difference;
        if ((difference = addr.difference(value.blockHead)) < 0) {
            return false;
        }

        // Second Condition: addr must be 8-byte aligned

        // NOTES: the size of memory allocated for an object
        // is the sum of header size and object size. That
        // sum is involved in 8-byte alignment.
        // However, the passed addr here excludes the header
        // size. Currently, OBJECT_HEADER_SIZE = 8 bytes and
        // ARRAY_HEADER_SIZE = 12. Therefore, special attention
        //should be paid to arrays only.
        if (difference % 8 != 0 && difference % 8 != 4) {
            return false;
        }

        // Third Condition: addr must be set correspondingly in allocbits
        if (difference % 8 == 4) { // Maybe an array
            return value.isSet((HeapAddress) (addr.offset(-12)));
        } else { // Maybe an object
            return value.isSet((HeapAddress) (addr.offset(-8)));
        }
    }
}

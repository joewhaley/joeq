// LargeHeap.java, created Tue Dec 10 14:02:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Heap;

import Allocator.HeapAllocator;
import Memory.HeapAddress;
import Run_Time.Debug;
import Run_Time.SystemInterface;
import Util.Assert;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class LargeHeap extends Heap {

    // Internal management
    private ImmortalHeap immortal; // place where we allocate metadata
    private final int pageSize = 4096; // large space allocated in 4K chunks
    private final int GC_LARGE_SIZES = 20; // for statistics  
    private final int GC_INITIAL_LARGE_SPACE_PAGES = 200;
    // for early allocation of large objs
    private int usedPages = 0;
    private int largeSpacePages;
    private int large_last_allocated; // where to start search for free space
    private short[] largeSpaceAlloc; // used to allocate in large space
    private short[] largeSpaceMark; // used to mark large objects
    private int[] countLargeAlloc; //  - count sizes of large objects alloc'ed

    /**
     * Initialize for boot image - called from init of various collectors
     */
    public LargeHeap(ImmortalHeap imm) {
        super("Large Object Heap");
        immortal = imm;
        large_last_allocated = 0;
        largeSpacePages = GC_INITIAL_LARGE_SPACE_PAGES;
        countLargeAlloc = new int[GC_LARGE_SIZES];
    }

    public void init() {
        int size = GC_INITIAL_LARGE_SPACE_PAGES * pageSize;
        start = (HeapAddress) SystemInterface.syscalloc(size);
        if (start.isNull()) {
            Debug.writeln("Panic!  Cannot allocate ", size, "bytes.");
            Assert.UNREACHABLE();
        }
        end = (HeapAddress) start.offset(size);
        largeSpaceAlloc = new short[GC_INITIAL_LARGE_SPACE_PAGES];
        largeSpaceMark = new short[GC_INITIAL_LARGE_SPACE_PAGES];
    }

    /**
     * Get total amount of memory used by large space.
     *
     * @return the number of bytes
     */
    public int totalMemory() {
        return getSize();
    }

    /**
     * Allocate size bytes of zeroed memory.
     * Size is a multiple of wordsize, and the returned memory must be word aligned
     * 
     * @param size Number of bytes to allocate
     * @return Address of allocated storage
     */
    protected HeapAddress allocateZeroedMemory(int size) {
        int count = 0;
        int num_pages;
        int first_free;
outerLoop:
        while (true) {
            num_pages = (size + (pageSize - 1)) / pageSize;
            // Number of pages needed
            int last_possible = largeSpacePages - num_pages;

            synchronized(this) {
                
                while (largeSpaceAlloc[large_last_allocated] != 0) {
                    large_last_allocated += largeSpaceAlloc[large_last_allocated];
                }
    
                first_free = large_last_allocated;
                while (first_free <= last_possible) {
                    // Now find contiguous pages for this object
                    // first find the first available page
                    // i points to an available page: remember it
                    int i;
                    for (i = first_free + 1; i < first_free + num_pages; i++) {
                        if (largeSpaceAlloc[i] != 0)
                            break;
                    }
                    if (i == (first_free + num_pages)) {
                        // successful: found num_pages contiguous pages
                        // mark the newly allocated pages
                        // mark the beginning of the range with num_pages
                        // mark the end of the range with -num_pages
                        // so that when marking (ref is input) will know which extreme 
                        // of the range the ref identifies, and then can find the other
    
                        largeSpaceAlloc[first_free + num_pages - 1] =
                            (short) (-num_pages);
                        largeSpaceAlloc[first_free] = (short) (num_pages);
                        break outerLoop; // release lock *and synch changes*
                    } else {
                        // free area did not contain enough contig. pages
                        first_free = i + largeSpaceAlloc[i];
                        while (largeSpaceAlloc[first_free] != 0)
                            first_free += largeSpaceAlloc[first_free];
                    }
                }
            }
            //release lock: won't keep change to large_last_alloc'd

            // Couldn't find space; inform allocator (which will either trigger GC or 
            // throw out of memory exception)
            HeapAllocator.heapExhausted(this, size, count++);
        }
        int pageSize = 1 << HeapAddress.pageAlign();
        HeapAddress target = (HeapAddress) start.offset(pageSize * first_free);
        SystemInterface.mem_set(target, size, (byte)0);
        // zero space before return
        usedPages += num_pages;
        return target;
    }

    /**
     * Hook to allow heap to perform post-allocation processing of the object.
     * For example, setting the GC state bits in the object header.
     */
    protected void postAllocationProcessing(Object newObj) {
    }

    public void startCollect() {
        usedPages = 0;
        HeapAddress a = HeapAddress.addressOf(largeSpaceMark);
        SystemInterface.mem_set(a, largeSpaceMark.length * 2, (byte)0);
    }

    public void endCollect() {
        short[] temp = largeSpaceAlloc;
        largeSpaceAlloc = largeSpaceMark;
        largeSpaceMark = temp;
        large_last_allocated = 0;
    }

    public boolean isLive(HeapAddress ref) {
        HeapAddress addr = ref;
        int page_num = addr.difference(start) >> 12;
        return (largeSpaceMark[page_num] != 0);
    }

    public boolean mark(HeapAddress ref) {
        HeapAddress tref = ref;

        int ij;
        int page_num = tref.difference(start) >>> 12;
        boolean result = (largeSpaceMark[page_num] != 0);
        if (result)
            return false; // fast, no synch case

        synchronized (this) { // get sysLock for large objects
            result = (largeSpaceMark[page_num] != 0);
            if (result) { // need to recheck
                return false;
            }
            int temp = largeSpaceAlloc[page_num];
            usedPages += (temp > 0) ? temp : -temp;
            if (temp == 1) {
                largeSpaceMark[page_num] = 1;
            } else {
                // mark entries for both ends of the range of allocated pages
                if (temp > 0) {
                    ij = page_num + temp - 1;
                    largeSpaceMark[ij] = (short) - temp;
                } else {
                    ij = page_num + temp + 1;
                    largeSpaceMark[ij] = (short) - temp;
                }
                largeSpaceMark[page_num] = (short) temp;
            }
        } // unlock INCLUDES sync()

        return true;
    }

    private void countObjects() {
        int i, num_pages, countLargeOld;
        int contiguousFreePages, maxContiguousFreePages;

        for (i = 0; i < GC_LARGE_SIZES; i++)
            countLargeAlloc[i] = 0;
        countLargeOld = contiguousFreePages = maxContiguousFreePages = 0;

        for (i = 0; i < largeSpacePages;) {
            num_pages = largeSpaceAlloc[i];
            if (num_pages == 0) { // no large object found here
                countLargeAlloc[0]++; // count free pages in entry[0]
                contiguousFreePages++;
                i++;
            } else { // at beginning of a large object
                if (num_pages < GC_LARGE_SIZES - 1)
                    countLargeAlloc[num_pages]++;
                else
                    countLargeAlloc[GC_LARGE_SIZES - 1]++;
                if (contiguousFreePages > maxContiguousFreePages)
                    maxContiguousFreePages = contiguousFreePages;
                contiguousFreePages = 0;
                i = i + num_pages; // skip to next object or free page
            }
        }
        if (contiguousFreePages > maxContiguousFreePages)
            maxContiguousFreePages = contiguousFreePages;

        Debug.write("Large Objects Allocated - by num pages\n");
        for (i = 0; i < GC_LARGE_SIZES - 1; i++) {
            Debug.write("pages ");
            Debug.write(i);
            Debug.write(" count ");
            Debug.write(countLargeAlloc[i]);
            Debug.write("\n");
        }
        Debug.write(countLargeAlloc[GC_LARGE_SIZES - 1]);
        Debug.write(" large objects ");
        Debug.write(GC_LARGE_SIZES - 1);
        Debug.write(" pages or more.\n");
        Debug.write(countLargeAlloc[0]);
        Debug.write(" Large Object Space pages are free.\n");
        Debug.write(maxContiguousFreePages);
        Debug.write(" is largest block of contiguous free pages.\n");
        Debug.write(countLargeOld);
        Debug.write(" large objects are old.\n");

    } // countLargeObjects()

    public int freeSpace() {
        return usedPages * pageSize;
    }

}

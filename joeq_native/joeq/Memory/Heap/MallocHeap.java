// MallocHeap.java, created Tue Dec 10 14:02:00 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Heap;

import joeq.Allocator.ObjectLayout;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Memory.HeapAddress;
import joeq.Runtime.Debug;
import joeq.Runtime.SystemInterface;
import joeq.Runtime.Unsafe;
import joeq.Util.Assert;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class MallocHeap extends Heap {

    /**
     * Initialize for boot image - called from init of various collectors
     */
    public MallocHeap() {
        super("Malloc Heap");
    }

    void init() {
    }

    /**
     * Get total amount of memory used by malloc space.
     *
     * @return the number of bytes
     */
    public int totalMemory() {
        return getSize();
    }

    /**
     * Allocate a scalar object. Fills in the header for the object,
     * and set all data fields to zero. Assumes that type is already initialized.
     * Disables thread switching during allocation. 
     * 
     * @param type  jq_Class of type to be instantiated
     *
     * @return the reference for the allocated object
     */
    public Object atomicAllocateObject(jq_Class type) {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = allocateObject(type);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }

    /**
     * Allocate an array object. Fills in the header for the object,
     * sets the array length to the specified length, and sets
     * all data fields to zero.  Assumes that type is already initialized.
     * Disables thread switching during allocation. 
     *
     * @param type  jq_Array of type to be instantiated
     * @param numElements  number of array elements
     *
     * @return the reference for the allocated array object 
     */
    public Object atomicAllocateArray(jq_Array type, int numElements) {
        Unsafe.getThreadBlock().disableThreadSwitch();
        Object o = allocateArray(type, numElements);
        Unsafe.getThreadBlock().enableThreadSwitch();
        return o;
    }

    /**
     * Atomically free an array object.
     * @param o the object to free
     */
    public void atomicFreeArray(Object o) {
        Unsafe.getThreadBlock().disableThreadSwitch();
        HeapAddress start = (HeapAddress) HeapAddress.addressOf(o).offset(-ObjectLayout.ARRAY_HEADER_SIZE);
        free(start);
        Unsafe.getThreadBlock().enableThreadSwitch();
    }

    /**
     * Free a memory region.
     * @param addr the pointer to free
     */
    public void free(HeapAddress addr) {
        SystemInterface.sysfree(addr);
        // Cannot correctly change start/end here
    }

    /**
     * Allocate size bytes of zeroed memory.
     * Size is a multiple of wordsize, and the returned memory must be word aligned.
     * 
     * @param size Number of bytes to allocate
     * @return Address of allocated storage
     */
    protected synchronized HeapAddress allocateZeroedMemory(int size) {
        HeapAddress region = (HeapAddress) SystemInterface.syscalloc(size);
        if (region.isNull()) {
            Debug.writeln("Panic!  Cannot allocate ", size, "bytes.");
            Assert.UNREACHABLE();
        }
        HeapAddress regionEnd = (HeapAddress) region.offset(size);

        if (start.isNull() || region.difference(start) < 0)
            start = region;
        if (regionEnd.difference(end) > 0)
            end = regionEnd;

        return region;
    }

    /**
     * Hook to allow heap to perform post-allocation processing of the object.
     * For example, setting the GC state bits in the object header.
     */
    protected void postAllocationProcessing(Object newObj) {
        // nothing to do in this heap since the GC subsystem
        // ignores objects in the malloc heap.
    }

}

// ImmortalHeap.java, created Tue Dec 10 14:02:03 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Heap;

import joeq.Allocator.ObjectLayout;
import joeq.Allocator.ObjectLayoutMethods;
import joeq.Clazz.jq_Array;
import joeq.Memory.Address;
import joeq.Memory.HeapAddress;
import joeq.Run_Time.Debug;
import joeq.Run_Time.SystemInterface;
import joeq.Util.Assert;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ImmortalHeap extends Heap {

    private HeapAddress allocationCursor;
    private int markValue;

    /**
     * Initialize for boot image - called from init of various collectors or spaces
     */
    public ImmortalHeap() {
        super("Immortal Heap");
    }

    public void init(int size) {
        start = (HeapAddress) SystemInterface.syscalloc(size);
        if (start.isNull()) {
            Debug.writeln("Panic!  Cannot allocate ", size, "bytes.");
            Assert.UNREACHABLE();
        }
        end = (HeapAddress) start.offset(size);
        allocationCursor = start;
    }

    /**
     * Get total amount of memory used by immortal space.
     *
     * @return the number of bytes
     */
    public int totalMemory() {
        return end.difference(start);
    }

    /**
     * Get the total amount of memory available in immortal space.
     * @return the number of bytes available
     */
    public int freeMemory() {
        return end.difference(allocationCursor);
    }

    /**
     * Mark an object in the boot heap
     * @param ref the object reference to mark
     * @return whether or not the object was already marked
     */
    public boolean mark(HeapAddress ref) {
        Object obj = ref.asObject();
        return ObjectLayoutMethods.testAndMark(obj, markValue);
    }

    /**
     * Is the object reference live?
     */
    public boolean isLive(HeapAddress ref) {
        Object obj = ref.asObject();
        return ObjectLayoutMethods.testMarkBit(obj, markValue);
    }

    /**
     * Work to do before collection starts.
     */
    public void startCollect() {
        // flip the sense of the mark bit.
        markValue = markValue ^ ObjectLayout.GC_BIT;
    }

    /**
     * Allocate an array object whose pointer is N bit aligned.
     * 
     * @param type  jq_Array of type to be instantiated
     * @param numElements  number of array elements
     * @param alignment 
     *
     * @return the reference for the allocated array object 
     */
    public Object allocateAlignedArray(jq_Array type, int numElements, int alignment) {
        int size = type.getInstanceSize(numElements);
        size = Address.alignInt(size, HeapAddress.logSize());
        Object tib = type.getVTable();
        int offset = ObjectLayout.ARRAY_HEADER_SIZE;
        HeapAddress region = allocateZeroedMemory(size, alignment, offset);
        Object newObj = ObjectLayoutMethods.initializeArray(region, tib, numElements, size);
        postAllocationProcessing(newObj);
        return newObj;
    }

    /**
     * Allocate a chunk of memory of a given size.
     * 
     *   @param size Number of bytes to allocate
     *   @return Address of allocated storage
     */
    protected HeapAddress allocateZeroedMemory(int size) {
        return allocateZeroedMemory(size, 2, 0);
    }

    /**
     * Allocate a chunk of memory of a given size.
     * 
     *   @param size Number of bytes to allocate
     *   @param alignment Alignment specifier; must be a power of two
     *   @return Address of allocated storage
     */
    protected HeapAddress allocateZeroedMemory(int size, int alignment) {
        return allocateZeroedMemory(size, alignment, 0);
    }

    /**
     * Allocate a chunk of memory of a given size.
     * 
     *   @param size Number of bytes to allocate
     *   @param alignment Alignment specifier; must be a power of two
     *   @param offset Offset within the object that must be aligned
     *   @return Address of allocated storage
     */
    protected HeapAddress allocateZeroedMemory(int size, int alignment, int offset) {
        HeapAddress region = allocateInternal(size, alignment, offset);
        SystemInterface.mem_set(region, size, (byte)0);
        return region;
    }

    private synchronized HeapAddress allocateInternal(int size, int alignment, int offset) {
        // reserve space for offset bytes
        allocationCursor = (HeapAddress) allocationCursor.offset(offset);
        // align the interior portion of the requested space
        allocationCursor = (HeapAddress) allocationCursor.align(alignment);
        HeapAddress result = allocationCursor;
        // allocate remaining space 
        allocationCursor = (HeapAddress) allocationCursor.offset(size - offset);
        if (allocationCursor.difference(end) > 0)
            Assert.UNREACHABLE("Immortal heap space exhausted");
        // subtract back offset bytes
        HeapAddress a = (HeapAddress) result.offset(-offset);
        if (false) {
            Debug.write("Allocated ", size);
            Debug.write(" bytes, alignment=", alignment);
            Debug.write(", offset=", offset);
            Debug.writeln(" address=", a);
        }
        return a;
    }

    /**
     * Hook to allow heap to perform post-allocation processing of the object.
     * For example, setting the GC state bits in the object header.
     */
    protected void postAllocationProcessing(Object newObj) {
        ObjectLayoutMethods.writeMarkBit(newObj, markValue);
    }

}

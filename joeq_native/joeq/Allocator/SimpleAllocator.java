// SimpleAllocator.java, created Mon Feb  5 23:23:19 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import java.lang.reflect.Array;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_InstanceMethod;
import joeq.Class.jq_Primitive;
import joeq.Class.jq_Reference;
import joeq.Class.jq_Type;
import joeq.Memory.Address;
import joeq.Memory.HeapAddress;
import joeq.Runtime.SystemInterface;
import joeq.Runtime.Unsafe;
import joeq.Scheduler.jq_NativeThread;
import joeq.Scheduler.jq_Thread;
import joeq.Util.Assert;

/**
 * SimpleAllocator
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class SimpleAllocator extends HeapAllocator {

    /**
     * Size of blocks allocated from the OS.
     */
    public static final int BLOCK_SIZE = 2097152;

    /**
     * Maximum memory, in bytes, to be allocated from the OS.
     */
    public static /*final*/ int MAX_MEMORY = 67108864;

    /**
     * Threshold for direct OS allocation.  When an array overflows the current block
     * and is larger than this size, it is allocated directly from the OS.
     */
    public static final int LARGE_THRESHOLD = 262144;

    /**
     * Pointers to the start, current, and end of the heap.
     */
    private HeapAddress heapFirst, heapCurrent, heapEnd;

    /**
     * Simple work queue for GC.
     */
    private SimpleGCWorkQueue gcWorkQueue = new SimpleGCWorkQueue();
    
    /**
     * Pointer to the start of the free list.
     */
    private HeapAddress firstFreeBlock;
    
    /**
     * Pointer to the start and current of the large object list.
     */
    private HeapAddress firstLarge, currLarge;
    
    /**
     * Are we currently doing a GC?  For debugging purposes.
     */
    private boolean inGC;

    /**
     * Are we currently doing an allocation?  For debugging purposes.
     */
    private boolean inAlloc;
    
    /**
     * Perform initialization for this allocator.  This will be called before any other methods.
     * This allocates an initial block of memory from the OS and sets up relevant pointers.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    public void init() throws OutOfMemoryError {
        Assert._assert(!inGC);
        Assert._assert(!inAlloc);
        heapCurrent = heapFirst = (HeapAddress) SystemInterface.syscalloc(BLOCK_SIZE);
        if (heapCurrent.isNull())
            HeapAllocator.outOfMemory();
        // At end of memory block:
        //  - one word for pointer to first free memory in this block.
        //  - one word for pointer to start of next memory block.
        heapEnd = (HeapAddress) heapFirst.offset(BLOCK_SIZE - 2 * HeapAddress.size());
    }

    /**
     * Allocates a new block of memory from the OS, sets the current block to
     * point to it, and makes the new block the current block.
     *
     * @throws OutOfMemoryError if there is not enough memory for initialization
     */
    private void allocateNewBlock() {
        if (totalMemory() >= MAX_MEMORY) HeapAllocator.outOfMemory();
        heapCurrent = (HeapAddress) SystemInterface.syscalloc(BLOCK_SIZE);
        if (heapCurrent.isNull())
            HeapAllocator.outOfMemory();
        heapEnd.offset(HeapAddress.size()).poke(heapCurrent);
        // At end of memory block:
        //  - one word for pointer to first free memory in this block.
        //  - one word for pointer to start of next memory block.
        heapEnd = (HeapAddress) heapCurrent.offset(BLOCK_SIZE - 2 * HeapAddress.size());
    }

    /**
     * Returns an estimate of the amount of free memory available.
     *
     * @return bytes of free memory
     */
    public int freeMemory() {
        return heapEnd.difference(heapCurrent);
    }

    /**
     * Returns an estimate of the total memory allocated (both used and unused).
     *
     * @return bytes of memory allocated
     */
    public int totalMemory() {
        int total = 0;
        HeapAddress ptr = heapFirst;
        while (!ptr.isNull()) {
            total += BLOCK_SIZE;
            ptr = (HeapAddress) ptr.offset(BLOCK_SIZE - HeapAddress.size()).peek();
        }
        return total;
    }

    /**
     * Allocate an object with the default alignment.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateObject(int size, Object vtable) throws OutOfMemoryError {
        Assert._assert(!inGC);
        Assert._assert(!inAlloc); inAlloc = true;
        if (size < ObjectLayout.OBJ_HEADER_SIZE) {
            // size overflow! become minus!
            inAlloc = false;
            HeapAllocator.outOfMemory();
        }
        //jq.Assert((size & 0x3) == 0);
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            Object o = allocObjectFromFreeList(size, vtable);
            if (o != null) {
                inAlloc = false;
                return o;
            }
            // not enough space on free list, allocate another memory block.
            allocateNewBlock();
            addr = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE);
        }
        // fast path
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        inAlloc = false;
        return addr.asObject();
    }

    /**
     * Smallest object size allowed in free list.
     */
    public static final int MIN_SIZE = ObjectLayout.OBJ_HEADER_SIZE;
    
    private HeapAddress allocFromFreeList(int size) {
        // Search free list to find if there is an area that will fit the object.
        HeapAddress prev_p = HeapAddress.getNull();
        HeapAddress curr_p = firstFreeBlock;
        while (!curr_p.isNull()) {
            HeapAddress next_p = (HeapAddress) curr_p.peek();
            int areaSize = curr_p.offset(HeapAddress.size()).peek4();
            if (areaSize >= size) {
                // This area fits!
                HeapAddress addr = (HeapAddress) curr_p.offset(ObjectLayout.OBJ_HEADER_SIZE);
                // Zero out the memory.
                SystemInterface.mem_set(curr_p, areaSize, (byte) 0);
                // Fix up free list.
                int newSize = areaSize - size;
                HeapAddress new_next_p;
                if (newSize > MIN_SIZE) {
                    // Still some space left here.
                    Assert._assert(newSize >= HeapAddress.size() * 2);
                    new_next_p = (HeapAddress) curr_p.offset(size);
                    new_next_p.poke(next_p);
                    new_next_p.offset(HeapAddress.size()).poke4(newSize);
                } else {
                    // Remainder is too small, skip it.
                    new_next_p = next_p;
                }
                if (prev_p.isNull()) {
                    // New start of free list.
                    firstFreeBlock = new_next_p;
                } else {
                    // Patch previous in free list to point to new location.
                    prev_p.poke(new_next_p);
                }
                return addr;
            }
            prev_p = curr_p;
            curr_p = next_p;
        }
        // Nothing in the free list is big enough!
        return HeapAddress.getNull();
    }
    
    /**
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocObjectFromFreeList(int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = allocFromFreeList(size);
        if (addr.isNull()) {
            // Not enough space in free list, try a GC.
            collect();
            addr = allocFromFreeList(size);
            if (addr.isNull()) {
                // need to allocate new block of memory.
                return null;
            }
        }
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    /**
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocArrayFromFreeList(int length, int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = allocFromFreeList(size);
        if (addr.isNull()) {
            // Not enough space in free list, try a GC.
            collect();
            addr = allocFromFreeList(size);
            if (addr.isNull()) {
                // need to allocate new block of memory.
                return null;
            }
        }
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    public static final int getObjectSize(Object o) {
        jq_Reference t = jq_Reference.getTypeOf(o);
        int size;
        if (t.isArrayType()) {
            jq_Array a = (jq_Array) t;
            int length = Array.getLength(o);
            size = a.getInstanceSize(length);
        } else {
            jq_Class c = (jq_Class) t;
            size = c.getInstanceSize();
        }
        size = (size + 3) & ~3; // align size
        return size;
    }
    
    /**
     * @param length
     * @param size
     * @param vtable
     * @return
     * @throws OutOfMemoryError
     */
    private Object allocLargeArray(int length, int size, Object vtable) throws OutOfMemoryError {
        HeapAddress addr = (HeapAddress) SystemInterface.syscalloc(size + HeapAddress.size());
        if (addr.isNull())
            outOfMemory();
        if (firstLarge.isNull()) {
            firstLarge = currLarge = addr;
        } else {
            Object o = ((HeapAddress) currLarge.offset(ObjectLayout.ARRAY_HEADER_SIZE)).asObject();
            int currLargeSize = getObjectSize(o);
            currLarge.offset(currLargeSize).poke(addr);
            currLarge = addr;
        }
        addr = (HeapAddress) addr.offset(ObjectLayout.ARRAY_HEADER_SIZE);
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    /**
     * Allocate an object such that the first field is 8-byte aligned.
     * If the object cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param size size of object to allocate (including object header), in bytes
     * @param vtable vtable pointer for new object
     * @return new uninitialized object
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateObjectAlign8(int size, Object vtable) throws OutOfMemoryError {
        heapCurrent = (HeapAddress) heapCurrent.offset(ObjectLayout.OBJ_HEADER_SIZE).align(3).offset(-ObjectLayout.OBJ_HEADER_SIZE);
        return allocateObject(size, vtable);
    }

    /**
     * Allocate an array with the default alignment.
     * If length is negative, throws NegativeArraySizeException.
     * If the array cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param length length of new array
     * @param size size of array to allocate (including array header), in bytes
     * @param vtable vtable pointer for new array
     * @return new array
     * @throws NegativeArraySizeException if length is negative
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateArray(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        Assert._assert(!inGC);
        Assert._assert(!inAlloc); inAlloc = true;
        if (length < 0) throw new NegativeArraySizeException(length + " < 0");
        if (size < ObjectLayout.ARRAY_HEADER_SIZE) {
            // size overflow!
            inAlloc = false;
            HeapAllocator.outOfMemory();
        }
        size = (size + 3) & ~3; // align size
        HeapAddress addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
        heapCurrent = (HeapAddress) heapCurrent.offset(size);
        if (heapEnd.difference(heapCurrent) < 0) {
            // not enough space (rare path)
            heapCurrent = (HeapAddress) heapCurrent.offset(-size);
            if (size > LARGE_THRESHOLD) {
                // special large-object allocation
                return allocLargeArray(length, size, vtable);
            } else {
                Object o = allocArrayFromFreeList(length, size, vtable);
                if (o != null) {
                    inAlloc = false;
                    return o;
                }
                // not enough space on free list, allocate another memory block.
                allocateNewBlock();
                addr = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE);
            }
        }
        // fast path
        addr.offset(ObjectLayout.ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(ObjectLayout.VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        inAlloc = false;
        return addr.asObject();
    }

    /**
     * Allocate an array such that the elements are 8-byte aligned.
     * If length is negative, throws NegativeArraySizeException.
     * If the array cannot be allocated due to lack of memory, throws OutOfMemoryError.
     *
     * @param length length of new array
     * @param size size of array to allocate (including array header), in bytes
     * @param vtable vtable pointer for new array
     * @return new array
     * @throws NegativeArraySizeException if length is negative
     * @throws OutOfMemoryError if there is insufficient memory to perform the operation
     */
    public Object allocateArrayAlign8(int length, int size, Object vtable) throws OutOfMemoryError, NegativeArraySizeException {
        heapCurrent = (HeapAddress) heapCurrent.offset(ObjectLayout.ARRAY_HEADER_SIZE).align(3).offset(-ObjectLayout.ARRAY_HEADER_SIZE);
        return allocateArray(length, size, vtable);
    }

    public void collect() {
        jq_Thread t = Unsafe.getThreadBlock();
        t.disableThreadSwitch();
        inGC = true;
        jq_NativeThread.suspendAllThreads();
        
        SemiConservative.scanRoots(true);
        scanGCQueue(true);
        updateFreeList();
        updateLargeObjectList();
        SemiConservative.scanRoots(false);
        scanGCQueue(false);        
        
        inGC = false;
        jq_NativeThread.resumeAllThreads();
        t.enableThreadSwitch();
    }

    void scanGCQueue(boolean b) {
        for (;;) {
            Object o = gcWorkQueue.pull();
            if (o == null) break;
            scanObject(o, b);
        }
    }
    
    void updateLargeObjectList() {
        HeapAddress prev_p = HeapAddress.getNull();
        int prevSize = 0;
        HeapAddress curr_p = firstLarge;
        int currSize;
        while (!curr_p.isNull()) {
            Object o = ((HeapAddress) curr_p.offset(ObjectLayout.ARRAY_HEADER_SIZE)).asObject();
            currSize = getObjectSize(o);
            HeapAddress next_p = (HeapAddress) curr_p.offset(currSize).peek();
            int status = HeapAddress.addressOf(o).offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
            if ((status & ObjectLayout.GC_BIT) == 0) {
                if (prev_p.isNull()) {
                    firstLarge = next_p;
                } else {
                    prev_p.offset(prevSize).poke(next_p);
                }
                if (curr_p.difference(currLarge) == 0) {
                    currLarge = prev_p;
                }
            }
            prev_p = curr_p;
            prevSize = currSize;
            curr_p = next_p;
        }
    }
    
    void updateFreeList() {
        // Scan forward through heap, finding unmarked objects and merging free spaces.
        HeapAddress currFree, prevFree;
        int prevFreeSize;
        prevFree = HeapAddress.getNull();
        prevFreeSize = 0;
        currFree = firstFreeBlock;
        HeapAddress currBlock = heapFirst;
        while (!currBlock.isNull()) {
            HeapAddress currBlockEnd = (HeapAddress) currBlock.offset(BLOCK_SIZE - 2 * HeapAddress.size());
            HeapAddress p = currBlock;
            while (currBlockEnd.difference(p) > 0) {
                
                if (p.difference(currFree) == 0) {
                    // Skip over free chunk.
                    int size = currFree.offset(HeapAddress.size()).peek4();
                    p = (HeapAddress) p.offset(size);
                    prevFree = currFree;
                    prevFreeSize = size;
                    currFree = (HeapAddress) currFree.peek();
                    continue;
                }
                
                HeapAddress obj;
                
                // This is complicated by the fact that there may be a one-word
                // gap between objects due to 8-byte alignment.
                HeapAddress p1 = (HeapAddress) p.offset(ObjectLayout.OBJ_HEADER_SIZE);
                HeapAddress p3 = (HeapAddress) p.offset(ObjectLayout.ARRAY_HEADER_SIZE);
                
                Address vt1 = (Address) p1.offset(ObjectLayout.VTABLE_OFFSET).peek();
                Address vt3 = (Address) p3.offset(ObjectLayout.VTABLE_OFFSET).peek();
                
                boolean b1 = isValidVTable(vt1);
                boolean b3 = isValidArrayVTable(vt3);
                
                boolean skipWord;
                
                if (b1) {
                    // Looks like a scalar object at p1.
                    if (b3) {
                        // p1 could also be status word of array at p3.
                        obj = p3;
                        skipWord = false;
                    } else {
                        // todo: p1 could also possibly be status word of aligned object.
                        obj = p1;
                        skipWord = false;
                    }
                } else {
                    HeapAddress p4 = (HeapAddress) p.offset(4+ObjectLayout.ARRAY_HEADER_SIZE);
                    Address vt4 = (Address) p4.offset(ObjectLayout.VTABLE_OFFSET).peek();
                    boolean b4 = isValidArrayVTable(vt4);
                    if (b3) {
                        // Looks like array object at p3.
                        // todo: p3 could also possibly be status word of aligned array.
                        obj = p3;
                        skipWord = false;
                    } else if (b4) {
                        // Looks like aligned array at p4.
                        obj = p4;
                        skipWord = true;
                    } else {
                        // Heap is corrupted or we reached the end of this block.
                        break;
                    }
                }
                
                Object o = obj.asObject();
                int size = getObjectSize(o);
                
                HeapAddress next_p = (HeapAddress) p.offset(size + (skipWord?4:0)).align(2);
                
                int status = obj.offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
                if ((status & ObjectLayout.GC_BIT) == 0) {
                    HeapAddress prevFreeEnd = (HeapAddress) prevFree.offset(prevFreeSize);
                    if (p.difference(prevFreeEnd) == 0) {
                        // Just extend size of this free area.
                        prevFreeSize = p.difference(prevFree);
                        prevFree.offset(HeapAddress.size()).poke4(prevFreeSize);
                    } else {
                        // Insert into free list.
                        p.poke(currFree);
                        p.offset(HeapAddress.size()).poke4(next_p.difference(p));
                        if (prevFree.isNull()) {
                            firstFreeBlock = p;
                        } else {
                            prevFree.poke(p);
                        }
                    }
                }
                
                p = next_p;
            }
            currBlock = (HeapAddress) currBlock.offset(BLOCK_SIZE - 2 * HeapAddress.size()).peek();
        }
    }
    
    void scanObject(Object obj, boolean b) {
        jq_Reference type = jq_Reference.getTypeOf(obj);
        if (type.isClassType()) {
            int[] referenceOffsets = ((jq_Class)type).getReferenceOffsets();
            for (int i = 0, n = referenceOffsets.length; i < n; i++) {
                HeapAddress objRef = HeapAddress.addressOf(obj);
                DefaultHeapAllocator.processPtrField(objRef.offset(referenceOffsets[i]), b);
            }
        } else {
            jq_Type elementType = ((jq_Array)type).getElementType();
            if (elementType.isReferenceType()) {
                int num_elements = Array.getLength(obj);
                int numBytes = num_elements * HeapAddress.size();
                HeapAddress objRef = HeapAddress.addressOf(obj);
                HeapAddress location = (HeapAddress) objRef.offset(ObjectLayout.ARRAY_ELEMENT_OFFSET);
                HeapAddress end = (HeapAddress) location.offset(numBytes);
                while (location.difference(end) < 0) {
                    DefaultHeapAllocator.processPtrField(location, b);
                    location =
                        (HeapAddress) location.offset(HeapAddress.size());
                }
            }
        }
    }
    
    public void processPtrField(Address a, boolean b) {
        if (!isValidObjectRef(a)) return;
        gcWorkQueue.addToQueue(a, b);
    }
    
    public boolean isValidObjectRef(Address a) {
        if (!isValidAddress(a)) return false;
        Address vt = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        return isValidVTable(vt);
    }
    
    public boolean isValidArrayRef(Address a) {
        if (!isValidAddress(a)) return false;
        Address vt = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        return isValidArrayVTable(vt);
    }
    
    public boolean isValidVTable(Address a) {
        if (!isValidAddress(a)) return false;
        Address vtableTypeAddr = a.offset(ObjectLayout.VTABLE_OFFSET);
        jq_Reference r = PrimordialClassLoader.getAddressArray();
        if (!isType(vtableTypeAddr, r)) return false;
        return isValidType((HeapAddress) a.peek());
    }
    
    public boolean isValidArrayVTable(Address a) {
        if (!isValidAddress(a)) return false;
        Address vtableTypeAddr = a.offset(ObjectLayout.VTABLE_OFFSET);
        jq_Reference r = PrimordialClassLoader.getAddressArray();
        if (!isType(vtableTypeAddr, r)) return false;
        return isValidArray((HeapAddress) a.peek());
    }
    
    public boolean isType(Address a, jq_Reference t) {
        if (!isValidAddress(a)) return false;

        Address vtable = a.offset(ObjectLayout.VTABLE_OFFSET).peek();
        if (!isValidAddress(vtable)) return false;
        Address type = vtable.peek();
        Address expected = HeapAddress.addressOf(t);
        return expected.difference(type) == 0;
    }
    
    public boolean isValidType(Address typeAddress) {
        if (!isValidAddress(typeAddress)) return false;

        // check if vtable is one of three possible values
        Object vtable = ObjectLayoutMethods.getVTable(((HeapAddress) typeAddress).asObject());
        boolean valid = vtable == jq_Class._class.getVTable() ||
                        vtable == jq_Array._class.getVTable() ||
                        vtable == jq_Primitive._class.getVTable();
        return valid;
    }
    
    public boolean isValidArray(Address typeAddress) {
        if (!isValidAddress(typeAddress)) return false;

        Object vtable = ObjectLayoutMethods.getVTable(((HeapAddress) typeAddress).asObject());
        boolean valid = vtable == jq_Array._class.getVTable();
        return valid;
    }
    
    public boolean isValidAddress(Address a) {
        HeapAddress p = heapFirst;
        while (!p.isNull()) {
            int diff = a.difference(p); 
            if (diff >= 0 && diff < BLOCK_SIZE - 2 * HeapAddress.size())
                return true;
            p = (HeapAddress) p.offset(BLOCK_SIZE - HeapAddress.size()).peek();
        }
        return false;
    }
    
    public static final jq_Class _class;
    public static final jq_InstanceMethod _allocateObject;
    public static final jq_InstanceMethod _allocateObjectAlign8;
    public static final jq_InstanceMethod _allocateArray;
    public static final jq_InstanceMethod _allocateArrayAlign8;

    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Allocator/SimpleAllocator;");
        _allocateObject = _class.getOrCreateInstanceMethod("allocateObject", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateObjectAlign8 = _class.getOrCreateInstanceMethod("allocateObjectAlign8", "(ILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArray = _class.getOrCreateInstanceMethod("allocateArray", "(IILjava/lang/Object;)Ljava/lang/Object;");
        _allocateArrayAlign8 = _class.getOrCreateInstanceMethod("allocateArrayAlign8", "(IILjava/lang/Object;)Ljava/lang/Object;");
    }
}

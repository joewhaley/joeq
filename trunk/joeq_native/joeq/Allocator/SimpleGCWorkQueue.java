// SimpleGCWorkQueue.java, created Aug 3, 2004 3:29:21 AM by joewhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import joeq.Memory.HeapAddress;
import joeq.Runtime.Debug;
import joeq.Runtime.SystemInterface;
import joeq.Util.Assert;

/**
 * SimpleGCWorkQueue
 * 
 * @author John Whaley
 * @version $Id$
 */
public class SimpleGCWorkQueue {
    
    public static int WORKQUEUE_SIZE = 65536; // should be word-multiple
    
    HeapAddress queueStart, queueEnd;
    HeapAddress blockStart, blockEnd;
    
    /**
     * 
     */
    public SimpleGCWorkQueue() {
        super();
    }
    
    public void free() {
        if (SimpleAllocator.TRACE) Debug.writeln("Freeing work queue.");
        if (!blockStart.isNull()) {
            SystemInterface.sysfree(blockStart);
            blockStart = blockEnd = queueStart = queueEnd = HeapAddress.getNull();
        }
    }
    
    public void growQueue(int newSize) {
        // todo: use realloc here.
        if (SimpleAllocator.TRACE) Debug.writeln("Growing work queue to size ", newSize);
        HeapAddress new_queue = (HeapAddress) SystemInterface.syscalloc(newSize);
        if (SimpleAllocator.TRACE) Debug.writeln("New Queue start: ", new_queue);
        if (new_queue.isNull())
            HeapAllocator.outOfMemory();
        if (SimpleAllocator.TRACE) Debug.writeln("Queue start: ", queueStart);
        if (SimpleAllocator.TRACE) Debug.writeln("Queue end: ", queueEnd);
        if (SimpleAllocator.TRACE) Debug.writeln("Block start: ", blockStart);
        if (SimpleAllocator.TRACE) Debug.writeln("Block end: ", blockEnd);
        int size = queueEnd.difference(queueStart);
        if (size > 0) {
            if (SimpleAllocator.TRACE) Debug.writeln("Current size ", size);
            Assert._assert(newSize > size);
            SystemInterface.mem_cpy(new_queue, queueStart, size);
        } else {
            if (SimpleAllocator.TRACE) Debug.writeln("Pointers are flipped");
            int size2 = blockEnd.difference(queueStart);
            if (SimpleAllocator.TRACE) Debug.writeln("Size of first part: ", size2);
            SystemInterface.mem_cpy(new_queue, queueStart, size2);
            int size3 = queueEnd.difference(blockStart);
            if (SimpleAllocator.TRACE) Debug.writeln("Size of second part: ", size3);
            SystemInterface.mem_cpy(new_queue.offset(size2), blockStart, size3);
            size = size2 + size3;
        }
        if (SimpleAllocator.TRACE) Debug.writeln("Freeing old queue");
        SystemInterface.sysfree(blockStart);
        queueStart = blockStart = new_queue;
        blockEnd = (HeapAddress) blockStart.offset(newSize);
        queueEnd = (HeapAddress) queueStart.offset(size);
        if (SimpleAllocator.TRACE) Debug.writeln("New Block end:", blockEnd);
        if (SimpleAllocator.TRACE) Debug.writeln("New Queue end:", queueEnd);
    }
    
    public int size() {
        int size = queueEnd.difference(queueStart);
        if (size < 0) {
            size = blockEnd.difference(queueStart) +
                   queueEnd.difference(blockStart);
        }
        return size;
    }
    
    public int space() {
        int size = queueEnd.difference(queueStart);
        if (size < 0) {
            return -size;
        } else {
            return blockEnd.difference(queueEnd) +
                   queueStart.difference(blockStart);
        }
    }
    
    public boolean addToQueue(Object o, boolean b) {
        HeapAddress a = HeapAddress.addressOf(o);
        return addToQueue(a, b);
    }
    public boolean addToQueue(HeapAddress a, boolean b) {
        int statusWord = a.offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
        if (b) {
            if ((statusWord & ObjectLayout.GC_BIT) != 0) {
                if (SimpleAllocator.TRACE) Debug.writeln("Already visited, skipping ", a);
                return false;
            }
            a.offset(ObjectLayout.STATUS_WORD_OFFSET).poke4(statusWord | ObjectLayout.GC_BIT);
        } else {
            if ((statusWord & ObjectLayout.GC_BIT) == 0) {
                if (SimpleAllocator.TRACE) Debug.writeln("Already visited, skipping ", a);
                return false;
            }
            a.offset(ObjectLayout.STATUS_WORD_OFFSET).poke4(statusWord & ~ObjectLayout.GC_BIT);
        }
        if (SimpleAllocator.TRACE) Debug.writeln("Adding object to queue: ", a);
        if (space() <= HeapAddress.size()) {
            // need a bigger work queue!
            int size = blockEnd.difference(blockStart);
            if (size == 0) size = WORKQUEUE_SIZE;
            else size = (WORKQUEUE_SIZE *= 2);
            growQueue(size);
        }
        if (SimpleAllocator.TRACE) Debug.writeln("Adding at: ", queueEnd);
        queueEnd.poke(a);
        queueEnd = (HeapAddress) queueEnd.offset(HeapAddress.size());
        if (queueEnd.difference(blockEnd) == 0) {
            queueEnd = blockStart;
            if (SimpleAllocator.TRACE) Debug.writeln("Queue end pointer wrapped around to: ", queueEnd);
        }
        Assert._assert(queueEnd.difference(queueStart) != 0);
        return true;
    }
    
    public HeapAddress pull() {
        if (queueEnd.difference(queueStart) == 0) {
            return HeapAddress.getNull();
        }
        if (SimpleAllocator.TRACE) Debug.writeln("Pulling from: ", queueStart);
        HeapAddress a = (HeapAddress) queueStart.peek();
        queueStart = (HeapAddress) queueStart.offset(HeapAddress.size());
        if (queueStart.difference(blockEnd) == 0) {
            queueStart = blockStart;
            if (SimpleAllocator.TRACE) Debug.writeln("Queue start pointer wrapped around to: ", queueStart);
        }
        return a;
    }
}

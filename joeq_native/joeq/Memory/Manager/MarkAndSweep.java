// MarkAndSweep.java, created Tue Dec 10 14:02:32 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Manager;

import java.lang.reflect.Array;

import Allocator.ObjectLayoutMethods;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Memory.Address;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Memory.Heap.BootHeap;
import Memory.Heap.Heap;
import Memory.Heap.ImmortalHeap;
import Memory.Heap.LargeHeap;
import Memory.Heap.MallocHeap;
import Memory.Heap.SegregatedListHeap;
import Run_Time.Debug;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;
import Scheduler.jq_RegisterState;
import Util.Assert;

/**
 * MarkAndSweep collector, adapted from Jikes RVM version
 * written by Dick Attanasio.
 * 
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class MarkAndSweep implements GCConstants {

    /**
     * Control chattering during progress of GC
     */
    static int verbose = 0;

    // following are referenced from elsewhere in VM
    public static final boolean movesObjects = false;
    public static final boolean writeBarrier = false;
    static final int SMALL_SPACE_MAX = 2048; // largest object in small heap

    static final int GC_RETRY_COUNT = 3;
    // number of times to GC before giving up

    // statistics reporting
    //
    static boolean flag2nd = false;

    static boolean gc_collect_now = false;
    // flag to do a collection (new logic)

    private static BootHeap bootHeap = BootHeap.INSTANCE;
    private static MallocHeap mallocHeap = new MallocHeap();
    private static SegregatedListHeap smallHeap =
        new SegregatedListHeap("Small Object Heap", mallocHeap);
    public static ImmortalHeap immortalHeap = new ImmortalHeap();
    private static LargeHeap largeHeap = new LargeHeap(immortalHeap);

    static boolean gcInProgress = false;

    /** "getter" function for gcInProgress
     */
    public static boolean gcInProgress() {
        return gcInProgress;
    }

    /**
     * Setup done during bootimage building
     */
    public static void init() {
        smallHeap.init(jq_NativeThread.initial_native_thread);
        CollectorThread.init();
        // to alloc its rendezvous arrays, if necessary
    }

    public static void boot() {
        /*
        verbose = VM_Interface.verbose();
        int smallHeapSize = VM_Interface.smallHeapSize();
        smallHeapSize = (smallHeapSize / GC_BLOCKALIGNMENT) * GC_BLOCKALIGNMENT;
        smallHeapSize = VM_Memory.roundUpPage(smallHeapSize);
        int largeSize = VM_Interface.largeHeapSize();
        int immortalSize =
            VM_Memory.roundUpPage(
                4 * (largeSize / VM_Memory.getPagesize())
                    + ((int) (0.05 * smallHeapSize))
                    + 4 * VM_Memory.getPagesize());
        */

        Heap.boot();
        /*
        immortalHeap.attach(immortalSize);
        largeHeap.attach(largeSize);
        smallHeap.attach(smallHeapSize);
         */
        jq_NativeThread st = jq_NativeThread.initial_native_thread;
        smallHeap.boot(st, immortalHeap);

        if (verbose >= 1)
            showParameter();
    }

    static void showParameter() {
        Debug.writeln("\nMark-Sweep Collector (verbose = ", verbose, ")");
        bootHeap.show();
        immortalHeap.show();
        smallHeap.show();
        largeHeap.show();
        Debug.writeln(
            "  Work queue buffer size = ",
            GCWorkQueue.WORK_BUFFER_SIZE);
    }

    /**
     * Allocate a "scalar" (non-array) Java object.
     * 
     *   @param size Size of the object in bytes, including the header
     *   @param tib Pointer to the Type Information Block for the object type
     *   @return Initialized object reference
     */
    public static Object allocateObject(int size, Object[] tib)
        throws OutOfMemoryError {
        if (size > SMALL_SPACE_MAX) {
            return largeHeap.allocateObject(size, tib);
        } else {
            HeapAddress objaddr = SegregatedListHeap.allocateFastPath(size);
            return ObjectLayoutMethods.initializeObject(objaddr, tib, size);
        }
    }

    /**
     * Allocate an array object.
     * 
     *   @param numElements Number of elements in the array
     *   @param size Size of the object in bytes, including the header
     *   @param tib Pointer to the Type Information Block for the object type
     *   @return Initialized array reference
     */
    public static Object allocateArray(int numElements, int size, Object[] tib)
        throws OutOfMemoryError {
        if (size > SMALL_SPACE_MAX) {
            return largeHeap.allocateArray(numElements, size, tib);
        } else {
            HeapAddress objaddr = SegregatedListHeap.allocateFastPath(size);
            return ObjectLayoutMethods.initializeArray(
                objaddr,
                tib,
                numElements,
                size);
        }
    }

    /**
     * Handle heap exhaustion.
     * 
     * @param heap the exhausted heap
     * @param size number of bytes requested in the failing allocation
     * @param count the retry count for the failing allocation.
     */
    public static void heapExhausted(Heap heap, int size, int count) {
        flag2nd = count > 0;
        if (heap == smallHeap) {
            if (count > GC_RETRY_COUNT)
                GCUtil.outOfMemory(
                    "small object space",
                    heap.getSize(),
                    "-X:h=nnn");
            gc1("GC triggered by object request of ", size);
        } else if (heap == largeHeap) {
            if (count > GC_RETRY_COUNT)
                GCUtil.outOfMemory(
                    "large object space",
                    heap.getSize(),
                    "-X:lh=nnn");
            gc1("GC triggered by large object request of ", size);
        } else {
            Assert.UNREACHABLE("unexpected heap");
        }
    }

    // following is called from CollectorThread.boot() - to set the number
    // of system threads into the synchronization object; this number
    // is not yet available at Allocator.boot() time
    public static void gcSetup(int numSysThreads) {
        GCWorkQueue.workQueue.initialSetup(numSysThreads);
    }

    private static void prepareNonParticipatingVPsForGC() {
    }

    private static void prepareNonParticipatingVPsForAllocation() {
    }

    public static void collect() {
        if (!gc_collect_now) {
            return; // to avoid cascading gc
        }

        // set running threads context regs so that a scan of its stack
        // will start at the caller of collect (ie. CollectorThread.run)
        //
        StackAddress fp = StackAddress.getBasePointer();
        CodeAddress caller_ip = (CodeAddress) fp.offset(4).peek();
        StackAddress caller_fp = (StackAddress) fp.peek(); 
        jq_RegisterState rs = Unsafe.getThreadBlock().getRegisterState();
        rs.setEip(caller_ip);
        rs.setEbp(caller_fp);

        CollectorThread mylocal = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();

        //   SYNCHRONIZATION CODE for parallel gc
        if (true) {

            if (gcInProgress) {
                Debug.write(
                    "VM_Allocator: Garbage Collection entered recursively \n");
                SystemInterface.die(1000);
            } else {
                gcInProgress = true;
            }

            // setup common workqueue for num VPs participating
            GCWorkQueue.workQueue.initialSetup(
                CollectorThread.numCollectors());

            bootHeap.startCollect();
            immortalHeap.startCollect();

            // Now initialize the large object space mark array
            largeHeap.startCollect();

            // Initialize collection in the small heap
            smallHeap.startCollect();

            // perform per vp processing for non-participating vps
            prepareNonParticipatingVPsForGC();

        }
        //   END OF SYNCHRONIZED INITIALIZATION BLOCK

        mylocal.rendezvous();

        // ALL GC THREADS IN PARALLEL

        // reset collector thread local work queue buffers
        GCWorkQueue.resetWorkQBuffers();

        // Each participating processor clears the mark array for the blocks it owns
        smallHeap.zeromarks(Unsafe.getThreadBlock().getNativeThread());

        mylocal.rendezvous();

        ScanStatics.scanStatics(); // all threads scan JTOC in parallel

        ScanThreads.scanThreads(null);
        // all GC threads process thread objects & scan their stacks

        gc_emptyWorkQueue();

        // If counting or timing in GCWorkQueue, save current counter values
        //
        if (GCWorkQueue.WORKQUEUE_COUNTS)
            GCWorkQueue.saveCounters(mylocal);
        if (GCWorkQueue.MEASURE_WAIT_TIMES
            || CollectorThread.MEASURE_WAIT_TIMES)
            GCWorkQueue.saveWaitTimes(mylocal);

        // Do finalization now.
        if (mylocal.getGCOrdinal() == 1) {
            GCWorkQueue.workQueue.reset();
            // reset work queue shared control variables
            //VM_Finalizer.moveToFinalizable();
        }
        mylocal.rendezvous();
        // the prevents a speedy thread from thinking there is no work
        gc_emptyWorkQueue();
        mylocal.rendezvous();

        // done
        //if (VM.ParanoidGCCheck)
            smallHeap.clobberfree();

        // Sweep large space
        if (mylocal.gcOrdinal == 1) {
            if (verbose >= 1)
                Debug.write("Sweeping large space");
            largeHeap.endCollect();
        }

        // Sweep small heap
        smallHeap.sweep(mylocal);

        mylocal.rendezvous();

        // Each GC thread increments adds its wait times for this collection
        // into its total wait time - for printSummaryStatistics output
        //
        if (CollectorThread.MEASURE_WAIT_TIMES)
            mylocal.incrementWaitTimeTotals();

        if (mylocal.gcOrdinal == 1) {
            prepareNonParticipatingVPsForAllocation();

            gc_collect_now = false; // reset flag
            gcInProgress = false;
            //VM_GCLocks.reset();

            // done with collection...except for measurement counters, diagnostic output etc

            CollectorThread.printRendezvousTime();

            smallHeap.postCollectionReport();
        }
    } // end of collect

    static HeapAddress gc_getMatureSpace(int size) {
        Assert.UNREACHABLE();
        return null;
    }

    static void gc_scanObjectOrArray(HeapAddress objRef) {
        // NOTE: no need to process TIB; 
        //       all TIBS found through JTOC and are never moved.

        jq_Reference type = jq_Reference.getTypeOf(objRef.asObject());
        if (type.isClassType()) {
            int[] referenceOffsets = ((jq_Class) type).getReferenceOffsets();
            for (int i = 0, n = referenceOffsets.length; i < n; ++i) {
                processPtrValue((HeapAddress) objRef.offset(referenceOffsets[i]).peek());
            }
        } else if (type.isArrayType()) {
            if (((jq_Array) type).getElementType().isReferenceType()) {
                int num_elements = Array.getLength(objRef.asObject());
                HeapAddress location = objRef;
                // for arrays = address of [0] entry
                HeapAddress end = (HeapAddress) objRef.offset(num_elements * 4);
                while (location.difference(end) < 0) {
                    processPtrValue((HeapAddress) location.peek());
                    //  USING  "4" where should be using "size_of_pointer" (for 64-bits)
                    location =
                        (HeapAddress) location.offset(HeapAddress.size());
                }
            }
        } else {
            Debug.write(
                "VM_Allocator.gc_scanObjectOrArray: type not Array or Class");
            SystemInterface.die(1000);
        }
    } //  gc_scanObjectOrArray

    //  process objects in the work queue buffers until no more buffers to process
    //
    static void gc_emptyWorkQueue() {
        HeapAddress ref = GCWorkQueue.getFromWorkBuffer();

        if (GCWorkQueue.WORKQUEUE_COUNTS) {
            CollectorThread myThread = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();
            myThread.rootWorkCount = myThread.putWorkCount;
        }

        while (!ref.isNull()) {
            gc_scanObjectOrArray(ref);
            ref = GCWorkQueue.getFromWorkBuffer();
        }
    } // gc_emptyWorkQueue

    //  To be able to be called from java/lang/runtime, or internally
    //
    public static void gc() {
        gc1("GC triggered by external call to gc() ", 0);
    }

    public static void gc1(String why, int size) {
        gc_collect_now = true;

        if (verbose >= 1)
            Debug.writeln(why, size);

        //  Tell gc thread to reclaim space, then wait for it to complete its work.
        //  The gc thread will do its work by calling collect(), below.
        //
        CollectorThread.collect(CollectorThread.collect);
    }

    public static long totalMemory() {
        return smallHeap.getSize() + largeHeap.getSize();
    }

    public static long freeMemory() {
        return smallHeap.freeBlocks() * GC_BLOCKSIZE;
    }

    /*
     * Includes freeMemory and per-processor local storage
     * and partial blocks in small heap.
     */
    public static long allSmallFreeMemory() {
        return freeMemory() + smallHeap.partialBlockFreeMemory();
    }

    public static long allSmallUsableMemory() {
        return smallHeap.getSize();
    }

    static boolean gc_isLive(HeapAddress ref) {
        if (bootHeap.refInHeap(ref)) {
            return bootHeap.isLive(ref);
        } else if (immortalHeap.refInHeap(ref)) {
            return immortalHeap.isLive(ref);
        } else if (smallHeap.refInHeap(ref)) {
            return smallHeap.isLive(ref);
        } else if (largeHeap.refInHeap(ref)) {
            return largeHeap.isLive(ref);
        } else {
            Debug.write("gc_isLive: ref not in any known heap: ", ref);
            Assert.UNREACHABLE();
            return false;
        }
    }

    // Normally called from constructor of VM_Processor
    // Also called a second time for the PRIMORDIAL processor during VM.boot
    //
    public static void setupProcessor(jq_NativeThread st) {
        // for the PRIMORDIAL processor allocation of sizes, etc occurs
        // during init(), nothing more needs to be done
        //
        if (st != jq_NativeThread.initial_native_thread)
            smallHeap.setupProcessor(st);
    }

    public static void printclass(HeapAddress ref) {
        jq_Reference type = jq_Reference.getTypeOf(ref.asObject());
        type.getDesc().debugWrite();
    }

    // Called from WriteBuffer code for generational collectors.
    // Argument is a modified old object which needs to be scanned
    //
    static void processWriteBufferEntry(HeapAddress ref) {
        ScanObject.scanObjectOrArray(ref);
    }

    public static Object getLiveObject(Object obj) {
        return obj; // this collector does not copy objects
    }

    public static boolean processFinalizerCandidate(HeapAddress ref) {
        boolean is_live = gc_isLive(ref);
        if (!is_live) {
            // process the ref, to mark it live & enqueue for scanning
            processPtrValue(ref);
        }
        return is_live;
    }

    /**
     * Process an object reference field during collection.
     *
     * @param location  address of a reference field
     */
    public static void processPtrField(Address location) {
        location.poke(processPtrValue((HeapAddress) location.peek()));
    }

    /**
     * Process an object reference (value) during collection.
     *
     * @param ref  object reference (value)
     */
    public static HeapAddress processPtrValue(HeapAddress ref) {
        if (ref.isNull())
            return ref; // TEST FOR NULL POINTER

        if (smallHeap.refInHeap(ref)) {
            if (smallHeap.mark(ref))
                GCWorkQueue.putToWorkBuffer(ref);
            return ref;
        }

        if (largeHeap.refInHeap(ref)) {
            if (largeHeap.mark(ref))
                GCWorkQueue.putToWorkBuffer(ref);
            return ref;
        }

        if (bootHeap.refInHeap(ref)) {
            if (bootHeap.mark(ref))
                GCWorkQueue.putToWorkBuffer(ref);
            return ref;
        }

        if (immortalHeap.refInHeap(ref)) {
            if (immortalHeap.mark(ref))
                GCWorkQueue.putToWorkBuffer(ref);
            return ref;
        }

        if (mallocHeap.refInHeap(ref))
            return ref;

        Debug.write("processPtrValue: ref not in any known heap: ", ref);
        Assert.UNREACHABLE();
        return null;
    } // processPtrValue

}

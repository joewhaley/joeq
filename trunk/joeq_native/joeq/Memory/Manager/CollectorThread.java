package Memory.Manager;

import Allocator.DefaultHeapAllocator;
import Memory.Debug;
import Memory.HeapAddress;
import Run_Time.HighResolutionTimer;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;
import Util.AtomicCounter;

/**
 * @author John Whaley
 */
public class CollectorThread extends Thread implements GCConstants {
    
    private final static boolean debug_native = false;

    private final static boolean trace = false; // emit trace messages?

    /** When true, causes RVM collectors to display heap configuration at startup */
    static final boolean DISPLAY_OPTIONS_AT_BOOT = false;

    /**
     * When true, causes RVM collectors to measure time spent in each phase of
     * collection. Will also force summary statistics to be generated.
     */
    static final boolean TIME_GC_PHASES = false;

    /**
     * When true, collector threads measure time spent waiting for buffers while
     * processing the Work Queue, and time spent waiting in Rendezvous during
     * the collection process. Will also force summary statistics to be generated.
     */
    final static boolean MEASURE_WAIT_TIMES = false;

    /** Measure & print entry & exit times for rendezvous */
    final static boolean MEASURE_RENDEZVOUS_TIMES = false;
    final static boolean SHOW_RENDEZVOUS_TIMES = false;

    public static AtomicCounter participantCount;
    // to count arriving collector threads

    public static CollectorThread[] collectorThreads;
    // maps processor id to assoicated collector thread

    public static int collectionCount; // number of collections

    /**
     * The Handshake object that contains the state of the next or current
     * (in progress) collection.  Read by mutators when detecting a need for
     * a collection, and passed to the collect method when requesting a
     * collection.
     */
    static Handshake collect;

    /** Use by collector threads to rendezvous during collection */
    static SynchronizationBarrier gcBarrier;

    static {
        collect = new Handshake();
        participantCount = new AtomicCounter(0);
        // counter for threads starting a collection
    }

    /**
     * Initialize for boot image. Should be called from VM_Allocator.init() 
     */
    static void init() {
        gcBarrier = new SynchronizationBarrier();

        collectorThreads =
            new CollectorThread[1 + jq_NativeThread.MAX_NATIVE_THREADS];

    }

    /**
     * Initiate a garbage collection. 
     * Called by a mutator thread when its allocator runs out of space.
     * The caller should pass the Handshake that was referenced by the
     * static variable "collect" at the time space was unavailable.
     *
     * @param handshake Handshake for the requested collection
     */
    static void collect(Handshake handshake) {
        if (trace) {
            double start = HighResolutionTimer.now();
            handshake.requestAndAwaitCompletion();
            double stop = HighResolutionTimer.now();
            Debug.write(
                "VM_CollectorThread.collect: " + (stop - start) + " seconds\n");
        } else {
            handshake.requestAndAwaitCompletion();
        }
    }

    // overrides VM_Thread.toString
    public String toString() {
        return "VM_CollectorThread";
    }

    // returns number of collector threads participating in a collection
    //
    public static int numCollectors() {
        return participantCount.value();
    }

    /**
     * Return the GC ordinal for this collector thread. A integer, 1,2,...
     * assigned to each collector thread participating in the current
     * collection.  Only valid while GC is "InProgress".
     *
     * @return The GC ordinal
     */
    public final int getGCOrdinal() {
        return gcOrdinal;
    }

    /**
     * Run method for collector thread (one per VM_Processor).
     * Enters an infinite loop, waiting for collections to be requested,
     * performing those collections, and then waiting again.
     * Calls VM_Allocator.collect to perform the collection, which
     * will be different for the different allocators/collectors
     * that the RVM can be configured to use.
     */
    public void run() {
        int mypid;
        // id of processor thread is running on - constant for the duration
        // of each collection - actually should always be id of associated processor

        //  make sure Opt compiler does not compile this method
        //  references stored in registers by the opt compiler will not be relocated by GC
        //VM_Magic.pragmaNoOptCompile();

        while (true) {

            // suspend this thread: it will resume when scheduled by Handshake 
            // initiateCollection().  while suspended, collector threads reside on
            // the schedulers collectorQueue
            //
            /*
            VM_Scheduler.collectorMutex.lock();
            VM_Thread.getCurrentThread().yield(
                VM_Scheduler.collectorQueue,
                VM_Scheduler.collectorMutex);
            */

            // block mutators from running on the current processor
            Unsafe.getThreadBlock().disableThreadSwitch();

            // record time it took to stop mutators on this processor and get this
            // collector thread dispatched
            //
            if (MEASURE_WAIT_TIMES)
                stoppingTime = HighResolutionTimer.now() - SynchronizationBarrier.rendezvousStartTime;

            gcOrdinal = participantCount.increment();

            // the first RVM VP will wait for all the native VPs not blocked in native
            // to reach a SIGWAIT state
            //
            if (gcOrdinal == 1) {

                // TODO

            } // gcOrdinal==1

            // wait for other collector threads to arrive or be made non-participants
            gcBarrier.startupRendezvous();

            // record time it took for running collector thread to start GC
            //
            if (MEASURE_WAIT_TIMES)
                startingTime = HighResolutionTimer.now() - SynchronizationBarrier.rendezvousStartTime;

            // MOVE THIS INTO collect
            //
            // setup common workqueue for num VPs participating, used to be called once.
            // now count varies for each GC, so call for each GC
            if (gcOrdinal == 1)
                GCWorkQueue.workQueue.initialSetup(participantCount.value());

            if (this.isActive) {
                DefaultHeapAllocator.collect(); // gc
            }

            // wait for other collector threads to arrive here
            rendezvousWaitTime
                += gcBarrier.rendezvous(MEASURE_RENDEZVOUS_TIMES);

            // Wake up mutators waiting for this gc cycle and create new collection
            // handshake object to be used for next gc cycle.
            // Note that mutators will not run until after thread switching is enabled,
            // so no mutators can possibly arrive at old handshake object: it's safe to
            // replace it with a new one.
            //
            if (gcOrdinal == 1) {
                // Snip reference to any methods that are still marked obsolete after
                // we've done stack scans. This allows reclaiming them on next GC.
                //VM_CompiledMethods.snipObsoleteCompiledMethods();

                collectionCount += 1;

                // notify mutators waiting on previous handshake object - actually we
                // don't notify anymore, mutators are simply in processor ready queues
                // waiting to be dispatched.
                collect.notifyCompletion();
                collect = new Handshake();
                // replace handshake with new one for next collection

                // schedule the FinalizerThread, if there is work to do & it is idle
                // THIS NOW HAPPENS DURING GC - SWITCH TO DOING IT HERE 
                // VM_Finalizer.schedule();
            }

            // wait for other collector threads to arrive here
            rendezvousWaitTime
                += gcBarrier.rendezvous(MEASURE_RENDEZVOUS_TIMES);
            if (MEASURE_RENDEZVOUS_TIMES) {
                gcBarrier.rendezvous(false);
                // need extra barrier call to let all processors set prev rendezvouz time
                jq_NativeThread nt = Unsafe.getThreadBlock().getNativeThread();
                if (nt.getIndex() == 1)
                    SynchronizationBarrier.printRendezvousTimes();
            }

            // final cleanup for initial collector thread
            if (gcOrdinal == 1) {
                // unblock any native processors executing in native that were blocked
                // in native at the start of GC
                //
                
                // It is VERY unlikely, but possible that some RVM processors were
                // found in C, and were BLOCKED_IN_NATIVE, during the collection, and now
                // need to be unblocked.
                //

                // if NativeDaemonProcessor was BLOCKED_IN_SIGWAIT, unblock it
                //

                // resume any attached Processors blocked prior to Collection
                //

                // clear the GC flag
            }

            if (MEASURE_WAIT_TIMES)
                resetWaitTimers(); // reset for next GC

            // all collector threads enable thread switching on their processors
            // allowing waiting mutators to be scheduled and run.  The collector
            // threads go back to the top of the run loop, to place themselves
            // back on the collectorQueue, to wait for the next collection.
            //
            Unsafe.getThreadBlock().enableThreadSwitch();
            // resume normal scheduling

        } // end of while(true) loop

    } // run

    static void quiesceAttachedProcessors() {

        // if there are any attached processors (for user pthreads that entered the VM
        // via an AttachJVM) we may be briefly IN_JAVA during transitions to a RVM
        // processor. Ususally they are either IN_NATIVE (having returned to the users 
        // native C code - or invoked a native method) or IN_SIGWAIT (the native code
        // invoked a JNI Function, migrated to a RVM processor, leaving the attached
        // processor running its IdleThread, which enters IN_SIGWAIT and wait for a signal.
        //
        // We wait for the processor to be in either IN_NATIVE or IN_SIGWAIT and then 
        // block it there for the duration of the Collection

    } // quiesceAttachedProcessors

    static void resumeAttachedProcessors() {

        // any attached processors were quiesced in either BLOCKED_IN_SIGWAIT
        // or BLOCKED_IN_NATIVE.  Unblock them.

    } // resumeAttachedProcessors

    // Make a collector thread that will participate in gc.
    // Taken:    stack to run on
    //           processor to run on
    // Returned: collector
    // Note: "stack" must be in pinned memory: currently done by allocating it in the boot image.
    //
    public static CollectorThread createActiveCollectorThread(jq_NativeThread processorAffinity) {
        return new CollectorThread(true, processorAffinity);
    }

    // Make a collector thread that will not participate in gc.
    // It will serve only to lock out mutators from the current processor.
    // Taken:    stack to run on
    //           processor to run on
    // Returned: collector
    // Note: "stack" must be in pinned memory: currently done by allocating it in the boot image.
    //
    static CollectorThread createPassiveCollectorThread(
        jq_NativeThread processorAffinity) {
        return new CollectorThread(false, processorAffinity);
    }

    void rendezvous() {
        rendezvousWaitTime += gcBarrier.rendezvous(MEASURE_WAIT_TIMES);
    }

    void rendezvousRecord(double start, double end) {
        if (MEASURE_RENDEZVOUS_TIMES)
            rendezvousWaitTime += gcBarrier.rendezvousRecord(start, end);
    }

    static void printRendezvousTime() {
        if (SHOW_RENDEZVOUS_TIMES)
            SynchronizationBarrier.printRendezvousTimes();
    }

    //-----------------//
    // Instance fields //
    //-----------------//

    boolean isActive; // are we an "active participant" in gc?

    /** arrival order of collectorThreads participating in a collection */
    public int gcOrdinal;

    /** used by each CollectorThread when scanning thread stacks for references */
    GCMapIteratorGroup iteratorGroup;

    // pointers into work queue put and get buffers, used with loadbalanced 
    // workqueues where per thread buffers when filled are given to a common shared
    // queue of buffers of work to be done, and buffers for scanning are obtained
    // from that common queue (see GCWorkQueue)
    //
    // Put Buffer is filled from right (end) to left (start)
    //    | next ptr | ......      <--- | 00000 | entry | entry | entry |
    //      |                             |   
    //    putStart                      putTop
    //
    // Get Buffer is emptied from left (start) to right (end)
    //    | next ptr | xxxxx | xxxxx | entry | entry | entry | entry |
    //      |                   |                                     |   
    //    getStart             getTop ---->                         getEnd
    //

    /** start of current work queue put buffer */
    HeapAddress putBufferStart;
    /** current position in current work queue put buffer */
    HeapAddress putBufferTop;
    /** start of current work queue get buffer */
    HeapAddress getBufferStart;
    /** current position in current work queue get buffer */
    HeapAddress getBufferTop;
    /** end of current work queue get buffer */
    HeapAddress getBufferEnd;
    /** extra Work Queue Buffer */
    HeapAddress extraBuffer;
    /** second extra Work Queue Buffer */
    HeapAddress extraBuffer2;

    int timeInRendezvous; // time waiting in rendezvous (milliseconds)

    double stoppingTime; // mutator stopping time - until enter Rendezvous 1
    double startingTime; // time leaving Rendezvous 1
    double rendezvousWaitTime; // accumulated wait time in GC Rendezvous's

    // for measuring load balancing of work queues
    //
    int copyCount; // for saving count of objects copied
    int rootWorkCount; // initial count of entries == number of roots
    int putWorkCount; // workqueue entries found and put into buffers
    int getWorkCount; // workqueue entries got from buffers & scanned
    int swapBufferCount; // times get & put buffers swapped
    int putBufferCount; // number of buffers put to common workqueue
    int getBufferCount; // number of buffers got from common workqueue
    int bufferWaitCount; // number of times had to wait for a get buffer
    double bufferWaitTime; // accumulated time waiting for get buffers
    double finishWaitTime; // time waiting for all buffers to be processed;

    int copyCount1; // for saving count of objects copied
    int rootWorkCount1; // initial count of entries == number of roots
    int putWorkCount1; // workqueue entries found and put into buffers
    int getWorkCount1; // workqueue entries got from buffers & scanned
    int swapBufferCount1; // times get & put buffers swapped
    int putBufferCount1; // number of buffers put to common workqueue
    int getBufferCount1; // number of buffers got from common workqueue
    int bufferWaitCount1; // number of times had to wait for a get buffer
    double bufferWaitTime1; // accumulated time waiting for get buffers
    double finishWaitTime1; // time waiting for all buffers to be processed;

    double totalBufferWait; // total time waiting for get buffers
    double totalFinishWait; // total time waiting for no more buffers
    double totalRendezvousWait; // total time waiting for no more buffers

    // constructor
    //
    CollectorThread(boolean isActive, jq_NativeThread processorAffinity) {
        
        setDaemon(true); // this is redundant, but harmless
        this.isActive = isActive;
        //this.isGCThread = true;
        //this.processorAffinity = processorAffinity;
        //this.iteratorGroup = new GCMapIteratorGroup();

        // associate this collector thread with its affinity processor
        collectorThreads[processorAffinity.getIndex()] = this;
    }

    // Record number of processors that will be participating in gc synchronization.
    //
    public static void boot(int numProcessors) {
        //VM_Allocator.gcSetup(numProcessors);
    }

    void incrementWaitTimeTotals() {
        totalBufferWait += bufferWaitTime + bufferWaitTime1;
        totalFinishWait += finishWaitTime + finishWaitTime1;
        totalRendezvousWait += rendezvousWaitTime;
    }

    void resetWaitTimers() {
        bufferWaitTime = 0.0;
        bufferWaitTime1 = 0.0;
        finishWaitTime = 0.0;
        finishWaitTime1 = 0.0;
        rendezvousWaitTime = 0.0;
    }

    static void printThreadWaitTimes() {
        CollectorThread ct;

        Debug.write("*** Collector Thread Wait Times (in micro-secs)\n");
        for (int i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            Debug.write(i);
            Debug.write(" stop ");
            Debug.write((int) ((ct.stoppingTime) * 1000000.0));
            Debug.write(" start ");
            Debug.write((int) ((ct.startingTime) * 1000000.0));
            Debug.write(" SBW ");
            if (ct.bufferWaitCount1 > 0)
                Debug.write(ct.bufferWaitCount1 - 1);
            // subtract finish wait
            else
                Debug.write(0);
            Debug.write(" SBWT ");
            Debug.write((int) ((ct.bufferWaitTime1) * 1000000.0));
            Debug.write(" SFWT ");
            Debug.write((int) ((ct.finishWaitTime1) * 1000000.0));
            Debug.write(" FBW ");
            if (ct.bufferWaitCount > 0)
                Debug.write(ct.bufferWaitCount - 1);
            // subtract finish wait
            else
                Debug.write(0);
            Debug.write(" FBWT ");
            Debug.write((int) ((ct.bufferWaitTime) * 1000000.0));
            Debug.write(" FFWT ");
            Debug.write((int) ((ct.finishWaitTime) * 1000000.0));
            Debug.write(" RWT ");
            Debug.write((int) ((ct.rendezvousWaitTime) * 1000000.0));
            Debug.write("\n");

            ct.stoppingTime = 0.0;
            ct.startingTime = 0.0;
            ct.rendezvousWaitTime = 0.0;
            GCWorkQueue.resetWaitTimes(ct);
        }
    }

}

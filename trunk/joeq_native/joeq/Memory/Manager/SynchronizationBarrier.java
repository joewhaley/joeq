package Memory.Manager;

import Main.jq;
import Memory.Debug;
import Run_Time.HighResolutionTimer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;

/**
 * @author John Whaley
 */
public class SynchronizationBarrier {

    /** maximum processor id for rendezvous, sometimes includes the native daemon processor */
    private int maxProcessorId;

    /** number of physical processors on running computer */
    private int numRealProcessors;

    /** number of times i-th processor has entered barrier */
    private int[] entryCounts;

    /** measure rendezvous times - outer index is processor id - inner index is which rendezvous point */
    static double rendezvousStartTime;
    static int rendezvousIn[][] = null;
    static int rendezvousOut[][] = null;
    static int rendezvousCount[] = null;
    // indicates which rendezvous a processor is at

    /**
     * Constructor
     */
    SynchronizationBarrier() {
        // initialize numRealProcessors to 1. Will be set to actual value later.
        // Using without resetting will cause waitABit() to yield instead of spinning
        numRealProcessors = 1;
        entryCounts = new int[jq_NativeThread.MAX_NATIVE_THREADS];
        // No one will probably do more than 15 rendezvous without a reset
        rendezvousIn = new int[jq_NativeThread.MAX_NATIVE_THREADS][15];
        rendezvousOut = new int[jq_NativeThread.MAX_NATIVE_THREADS][15];
        rendezvousCount = new int[jq_NativeThread.MAX_NATIVE_THREADS];
    }

    /**
     * Wait for all other collectorThreads/processors to arrive at this barrier.
     */
    double rendezvous(boolean time) {

        double start = time ? HighResolutionTimer.now() : 0.0;
        int myProcessorId = Unsafe.getThreadBlock().getNativeThread().getIndex();
        int myCount = entryCounts[myProcessorId] + 1;

        // enter barrier
        //

        entryCounts[myProcessorId] = myCount;
        //VM_Magic.sync();
        // update main memory so other processors will see it in "while" loop, below

        // wait for other processors to catch up.
        //
        for (int i = 0; i < maxProcessorId; ++i) {
            if (entryCounts[i] < 0)
                continue; // skip non participating VP
            if (i == myProcessorId)
                continue;
            while (entryCounts[i] < myCount) {
                // yield virtual processor's time slice (more polite to o/s than spinning)
                //
                // VM_BootRecord bootRecord = VM_BootRecord.the_boot_record;
                // VM.sysCall0(bootRecord.sysVirtualProcessorYieldIP);
                //
                // ...put original spinning code back in, in place of above, since this
                // is only being used by parallel GC threads, running on fewer that all
                // available processors.
                //
                waitABit(1);
            }
        }

        //VM_Magic.isync(); // so subsequent instructions won't see stale values

        // leave barrier
        //
        if (time)
            return rendezvousRecord(start, HighResolutionTimer.now());
        return 0.0;
    }

    double rendezvousRecord(double start, double end) {
        int myProcessorId = Unsafe.getThreadBlock().getNativeThread().getIndex();
        int which = rendezvousCount[myProcessorId]++;
        jq.Assert(which < rendezvousIn[0].length);
        rendezvousIn[myProcessorId][which] =
            (int) ((start - rendezvousStartTime) * 1000000);
        rendezvousOut[myProcessorId][which] =
            (int) ((end - rendezvousStartTime) * 1000000);
        return end - start;
    }

    static void printRendezvousTimes() {

        Debug.writeln(
            "**** Rendezvous entrance & exit times (microsecs) **** ");
        for (int i = 0; i < jq_NativeThread.native_threads.length; i++) {
            Debug.write("  Thread ", i, ": ");
            for (int j = 0; j < rendezvousCount[i]; j++) {
                Debug.write("   R", j);
                Debug.write(" in ", rendezvousIn[i][j]);
                Debug.write(" out ", rendezvousOut[i][j]);
            }
            Debug.writeln();
        }
        Debug.writeln();
    }

    /**
     * Coments are for default implementation of jni (not the alternative implemenation)
     * <p>
     * First rendezvous for a collection, called by all CollectorThreads that arrive
     * to participate in a collection.  Thread with gcOrdinal==1 is responsible for
     * detecting RVM processors stuck in Native C, blocking them in Native, and making
     * them non-participants in the collection (by setting their counters to -1).
     * Other arriving collector threads just wait until all have either arrived or
     * been declared non-participating.
     */
    void startupRendezvous() {
        
    } // startupRendezvous

    /**
     * reset the rendezvous counters for all VPs to 0.
     * Also sets numRealProcessors to number of real CPUs.
     */
    void resetRendezvous() {

    }

    /**
     * method to give a waiting thread/processor something do without interferring
     * with other waiting threads or those trying to enter the rendezvous.
     * Spins if running with fewer RVM "processors" than physical processors.
     * Yields (to Operating System) if running with more "processors" than
     * real processors.
     *
     * @param x amount to spin in some unknown units
     */
    private int waitABit(int x) {
        // yield executing operating system thread back to the operating system
        SystemInterface.yield();
        return 0;
    }

    /**
     * remove a processor from the rendezvous for the current collection.
     * The removed processor in considered a "non-participant" for the collection.
     *
     * @param id  processor id of processor to be removed.
     */
    private void removeProcessor(int id) {

    } // removeProcessor

}

// GCWorkQueue.java, created Tue Dec 10 14:02:30 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Manager;

import Memory.HeapAddress;
import Run_Time.Debug;
import Run_Time.HighResolutionTimer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;
import Util.Assert;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class GCWorkQueue {

    //-----------------------
    //static variables

    // validate refs when put into buffers - to catch bads refs early
    private static final boolean VALIDATE_BUFFER_PUTS = false;

    private static final boolean trace = false;
    private static final boolean debug = false;

    /**
     * Flag to cause per thread counts of WorkQueue buffer activity
     * to be recorded and reported when -verbose:gc is specified.
     */
    static final boolean WORKQUEUE_COUNTS = false;

    /**
     * Flag to include counts of Gets and Puts counts in the counts
     * measured and reported when WORKQUEUE_COUNTS = true.
     * To a very close approximation, the Put count is the number of
     * objects marked (and thus scanned). The sum of the Gets & the
     * sum of the Puts should be approx. equal. (some system objects
     * do not get put into the workqueue buffers)
     */
    static final boolean COUNT_GETS_AND_PUTS = WORKQUEUE_COUNTS && false;

    /**
     * Flag to cause measurement of time waiting for get buffers,
     * and time waiting for all threads to finish processing their
     * buffers, measured on a per thread basis.
     * See CollectorThread.MEASURE_WAIT_TIMES.
     */
    static final boolean MEASURE_WAIT_TIMES =
        CollectorThread.MEASURE_WAIT_TIMES;

    /**
     * Default size of each work queue buffer. It can be overridden by the
     * command line argument -X:wbsize=nnn where nnn is in entries (words).
     * Changing the buffer size can significantly affect the performance
     * of the load-balancing Work Queue.
     */
    public static int WORK_BUFFER_SIZE = 4 * 1024;

    /** single instance of GCWorkQueue, allocated in the bootImage */
    public static GCWorkQueue workQueue = new GCWorkQueue();

    //-----------------------
    //instance variables

    private int numRealProcessors;
    private int numThreads;
    private int numThreadsWaiting;
    private HeapAddress bufferHead;
    private boolean completionFlag;

    //-----------------------

    /** constructor */
    public GCWorkQueue() {
        numRealProcessors = 1; // default to 1 real physical processor
    }

    /**
     * Reset the shared work queue, setting the number of
     * participating gc threads.
     */
    public synchronized void initialSetup(int n) {

        if (trace)
            Debug.write(" GCWorkQueue.initialSetup entered\n");

        numThreads = n;
        numThreadsWaiting = 0;
        completionFlag = false;
        bufferHead = HeapAddress.getNull();

    }

    /**
     * Reset the shared work queue. There should be nothing to do,
     * since all flags & counts should have been reset to their
     * proper initial value by the last thread leaving the previous
     * use of the Work Queue.  Reset will wait until all threads
     * have left a previous use of the queue. It should always
     * be called (by 1 thread) before any reuse of the Work Queue.
     */
    void reset() {

        int debug_counter = 0;
        int debug_counter_counter = 0;

        // Last thread to leave a previous use of the Work Queue will reset
        // the completionFlag to false, wait here for that to happen.
        //
        while (completionFlag == true) {

            // spin here a while waiting for others to leave
            int x = spinABit(5);

        } // end of while loop

    } // reset

    /**
     * Reset the thread local work queue buffers for the calling
     * GC thread (a CollectorThread).
     */
    static void resetWorkQBuffers() {

        CollectorThread myThread = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();

        if (trace)
            Debug.write(" GCWorkQueue.resetWorkQBuffers entered\n");

        if (myThread.putBufferStart.isNull())
            GCWorkQueue.allocatePutBuffer(myThread);
        else {
            // reset TOP pointer for existing buffer
            myThread.putBufferTop = (HeapAddress) 
                myThread.putBufferStart.offset(WORK_BUFFER_SIZE - 4);
        }

        //check get buffer 
        if (!myThread.getBufferStart.isNull()) {
            GCWorkQueue.freeGetBuffer(myThread);
            // release old work q buffer
            myThread.getBufferStart = HeapAddress.getNull();
        }
        //clear remaining pointers
        myThread.getBufferTop = HeapAddress.getNull();
        myThread.getBufferEnd = HeapAddress.getNull();
    }

    /**
     * Add a full thread local "put" buffer to the shared work queue
     *
     * @param bufferAddress address of buffer to add to shared queue
     */
    void addBuffer(HeapAddress bufferAddress) {

        synchronized (this) {

            if (trace)
                Debug.write(" GCWorkQueue.addBuffer entered\n");

            // add to buffer list
            HeapAddress temp = bufferHead;
            bufferHead = bufferAddress;

            // set forward ptr in first word of buffer
            bufferAddress.poke(temp);

            // wake up any waiting threads (if any)
            if (numThreadsWaiting == 0)
                return;
            else
                this.notify(); // wake up 1 thread thats waiting
        }
    }

    /**
     * Get a buffer from the shared work queue. Returns zero if no buffers
     * are currently available.
     *
     * @return address of buffer or 0 if none available
     */
    synchronized HeapAddress getBuffer() {

        if (trace)
            Debug.write(" GCWorkQueue.getBuffer entered\n");

        if (bufferHead.isNull())
            return bufferHead;
        HeapAddress temp = bufferHead;
        bufferHead = (HeapAddress) temp.peek();
        return temp;
    }

    /**
     * Get a buffer from the shared work queue.  Waits for a buffer
     * if none are currently available. Returns the address of a
     * buffer if one becomes available. Returns zero if it reaches
     * a state where all participating threads are waiting for a buffer.
     * This is used during collection to indicate the end of a scanning
     * phase.
     *
     * CURRENTLY WAIT BY SPINNING - SOMEDAY DO WAIT NOTIFY
     *
     * @return address of a buffer
     *         zero if no buffers available & all participants waiting
     */
    HeapAddress getBufferAndWait()
         {
        HeapAddress temp;
        int debug_counter = 0;
        int debug_counter_counter = 0;

        if (trace)
            Debug.write(" GCWorkQueue.getBufferAndWait entered\n");

        synchronized (this) {

            // see if any work to do, if so, return next work buffer
            if (!bufferHead.isNull()) {
                temp = bufferHead;
                bufferHead = (HeapAddress) temp.peek();
                return temp;
            }

            if (numThreads == 1)
                return HeapAddress.getNull();
            // only 1 thread, no work, just return

            numThreadsWaiting++; // add self to number of gc threads waiting

            // if this is last gc thread, ie all threads waiting, then we are done
            //
            if (numThreadsWaiting == numThreads) {
                numThreadsWaiting--; // take ourself out of count
                completionFlag = true; // to lets waiting threads return
                return HeapAddress.getNull();
            }
        } // end of synchronized block

        // wait for work to arrive (or for end) currently spin a while & then 
        // recheck the work queue (lacking a "system" level wait-notify)
        //
        while (true) {

            // if we had a system wait-notify...we could do following...
            // try { wait(); } 
            // catch (InterruptedException e) {
            //        Debug.write("Interrupted Exception in getBufferAndWait")
            //        }

            //spin here a while
            int x = spinABit(5);

            synchronized (this) {
                // see if any work to do
                if (!bufferHead.isNull()) {
                    temp = bufferHead;
                    bufferHead = (HeapAddress) temp.peek();
                    numThreadsWaiting--;
                    return temp;
                }
                //currently no work - are we finished
                if (completionFlag == true) {
                    numThreadsWaiting--; // take ourself out of count
                    if (numThreadsWaiting == 0)
                        completionFlag = false;
                    // last thread out resets completion flag
                    return HeapAddress.getNull(); // are we complete
                }
            } // end of synchronized block

        } // end of while loop

    } // end of method

    //----------------------------------------------------
    // methods used by gc threads to put entries into and get entries out of
    // thread local work buffers they are processing
    //----------------------------------------------------

    /**
     * Add a reference to the thread local put buffer.  If it is full 
     * add it to the shared queue of buffers and acquire a new empty buffer.
     * Put buffers are filled from end to start, with "top" pointing to an
     * empty slot to be filled.
     *
     * @param ref  object reference to add to the put buffer
     */
    static void putToWorkBuffer(HeapAddress ref) {

        if (VALIDATE_BUFFER_PUTS) {
            if (!GCUtil.validRef(ref)) {
                GCUtil.dumpRef(ref);
                //VM_Memory.dumpMemory(ref, 64, 64);
                // dump 16 words on either side of bad ref
                Assert.UNREACHABLE();
            }
        }

        CollectorThread myThread = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();
        
        if (COUNT_GETS_AND_PUTS)
            myThread.putWorkCount++;

        myThread.putBufferTop.poke(ref);
        myThread.putBufferTop = (HeapAddress) myThread.putBufferTop.offset(-4);
        if (myThread.putBufferTop.difference(myThread.putBufferStart) == 0) {

            // current buffer is full, give to common pool, get new buffer        
            if (WORKQUEUE_COUNTS)
                myThread.putBufferCount++;
            GCWorkQueue.workQueue.addBuffer(myThread.putBufferStart);

            GCWorkQueue.allocatePutBuffer(myThread);
            // allocate new Put Buffer
        }
    }

    /**
     * Get a reference from the thread local get buffer.  If the get buffer
     * is empty, and the put buffer has entries, then swap the get and put
     * buffers.  If the put buffer is also empty, get a buffer to process
     * from the shared queue of buffers.
     *
     * Get buffer entries are extracted from start to end with "top" pointing
     * to the last entry returned (and now empty).
     *
     * Returns zero when there are no buffers in the shared queue and
     * all participating GC threads are waiting.
     *
     * @return object reference from the get buffer
     *         zero when there are no more references to process
     */
    public static HeapAddress getFromWorkBuffer() {

        HeapAddress newbufaddress;
        HeapAddress temp;
        double temptime;

        CollectorThread myThread = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();

        if (COUNT_GETS_AND_PUTS)
            myThread.getWorkCount++;

        myThread.getBufferTop = (HeapAddress) myThread.getBufferTop.offset(4);

        if (myThread.getBufferTop.difference(myThread.getBufferEnd) < 0)
            //easy case- return next work q item
            return (HeapAddress) myThread.getBufferTop.peek();

        // get buffer is empty, getBufferTop == getBufferEnd

        // get buffer from shared queue of work buffers
        newbufaddress = workQueue.getBuffer();

        if (!newbufaddress.isNull()) {
            if (WORKQUEUE_COUNTS)
                myThread.getBufferCount++;
            if (!myThread.getBufferStart.isNull())
                GCWorkQueue.freeGetBuffer(myThread);
            // release old work q buffer
            myThread.getBufferStart = newbufaddress;
            //set up pointers for new get buffer
            myThread.getBufferTop = (HeapAddress) myThread.getBufferStart.offset(4);
            myThread.getBufferEnd = (HeapAddress) 
                myThread.getBufferStart.offset(WORK_BUFFER_SIZE);
            return (HeapAddress) myThread.getBufferTop.peek();
        }

        // no buffers in work queue at this time.  if our putBuffer is not empty, swap
        // it with our empty get buffer, and start processing the items in it
        if (myThread
            .putBufferTop
            .difference(myThread.putBufferStart.offset(WORK_BUFFER_SIZE - 4)) < 0) {

            if (WORKQUEUE_COUNTS)
                myThread.swapBufferCount++;

            if (!myThread.getBufferStart.isNull()) {
                // have get buffer, swap of get buffer and put buffer
                if (trace)
                    Debug.write(" GCWorkQueue.getFromWorkBuffer swapping\n");
                // swap start addresses
                temp = myThread.putBufferStart;
                myThread.putBufferStart = myThread.getBufferStart;
                myThread.getBufferStart = temp;
                //set end pointer of get buffer
                myThread.getBufferEnd = (HeapAddress) 
                    myThread.getBufferStart.offset(WORK_BUFFER_SIZE);
                // swap current top pointer
                temp = myThread.putBufferTop;
                myThread.putBufferTop = (HeapAddress) myThread.getBufferTop.offset(-4);
                // -4 to compensate for +4 above
                myThread.getBufferTop = temp;
                // points to empty slot preceding first occupied slot
            } else {
                // no get buffer, take put buffer and allocate new put buffer
                if (trace)
                    Debug.write(
                        " GCWorkQueue.getFromWorkBuffer swapping-no get buffer\n");
                myThread.getBufferStart = myThread.putBufferStart;
                myThread.getBufferTop = myThread.putBufferTop;
                myThread.getBufferEnd = (HeapAddress) 
                    myThread.getBufferStart.offset(WORK_BUFFER_SIZE);

                GCWorkQueue.allocatePutBuffer(myThread);
                //get a new Put Buffer
            }
            //return first entry in new get buffer
            myThread.getBufferTop = (HeapAddress) myThread.getBufferTop.offset(4);
            return (HeapAddress) myThread.getBufferTop.peek();
        }
        // put buffer and get buffer are both empty
        // go wait for work or notification that gc is finished

        // get buffer from queue or wait for more work
        if (MEASURE_WAIT_TIMES || CollectorThread.MEASURE_WAIT_TIMES) {
            temptime = HighResolutionTimer.now();
            myThread.bufferWaitCount++;
        }
        newbufaddress = workQueue.getBufferAndWait();

        if (trace)
            Debug.write(
                " GCWorkQueue.getFromWorkBuffer return from getBuffernAndWait\n");

        // get a new buffer of work
        if (!newbufaddress.isNull()) {
            if (MEASURE_WAIT_TIMES || CollectorThread.MEASURE_WAIT_TIMES)
                myThread.bufferWaitTime += (HighResolutionTimer.now() - temptime);

            if (WORKQUEUE_COUNTS)
                myThread.getBufferCount++;

            GCWorkQueue.freeGetBuffer(myThread);
            // release the old work q buffer
            myThread.getBufferStart = newbufaddress;

            //set up pointers for new get buffer
            myThread.getBufferTop = (HeapAddress) myThread.getBufferStart.offset(4);
            myThread.getBufferEnd = (HeapAddress) 
                myThread.getBufferStart.offset(WORK_BUFFER_SIZE);
            return (HeapAddress) myThread.getBufferTop.peek();
        }

        // no more work and no more buffers ie end of work queue phase of gc

        if (MEASURE_WAIT_TIMES || CollectorThread.MEASURE_WAIT_TIMES)
            myThread.finishWaitTime = (HighResolutionTimer.now() - temptime);

        // reset top ptr for get buffer to its proper "empty" state
        myThread.getBufferTop = (HeapAddress) 
            myThread.getBufferStart.offset(WORK_BUFFER_SIZE - 4);

        if (trace)
            Debug.write(" GCWorkQueue.getFromWorkBuffer no more work\n");
        return HeapAddress.getNull();

    } // getFromWorkBuffer

    /**
     * allocate a work queue "put" buffer for a CollectorThread,
     * first look to see if the thread has a saved "extra buffer",
     * if none available, then allocate from the system heap.
     *
     * @param   CollectorThread needing a new put buffer
     */
    private static void allocatePutBuffer(CollectorThread myThread) {
        HeapAddress bufferAddress;

        if (!myThread.extraBuffer.isNull()) {
            bufferAddress = myThread.extraBuffer;
            myThread.extraBuffer = HeapAddress.getNull();
        } else if (!myThread.extraBuffer2.isNull()) {
            bufferAddress = myThread.extraBuffer2;
            myThread.extraBuffer2 = HeapAddress.getNull();
        } else {
            bufferAddress = (HeapAddress) SystemInterface.syscalloc(WORK_BUFFER_SIZE);
            if (bufferAddress.isNull()) {
                Debug.write(
                    " In GCWorkQueue: call to sysMalloc for work buffer returned 0\n");
                SystemInterface.die(1901);
            }
        }
        myThread.putBufferStart = bufferAddress;
        myThread.putBufferTop = (HeapAddress) bufferAddress.offset(WORK_BUFFER_SIZE - 4);
    } // allocatePutBuffer

    /**
     * free current "get" buffer for a CollectorThread, first try
     * to save as one of the threads two "extra" buffers, then return
     * buffer to the system heap.
     *
     * @param   CollectorThread with a get buffer to free
     */
    private static void freeGetBuffer(CollectorThread myThread) {

        if (myThread.extraBuffer.isNull())
            myThread.extraBuffer = myThread.getBufferStart;
        else if (myThread.extraBuffer2.isNull())
            myThread.extraBuffer2 = myThread.getBufferStart;
        else
            SystemInterface.sysfree(myThread.getBufferStart);

    } // freeGetBuffer

    /**
     * method to give a waiting thread/processor something do for a while
     * without interferring with others trying to access the synchronized block.
     * Spins if running with fewer RVM "processors" than physical processors.
     * Yields (to Operating System) if running with more "processors" than
     * real processors.
     *
     * @param x amount to spin in some unknown units
     */
    private int spinABit(int x) {
        int sum = 0;

        if (numThreads < numRealProcessors) {
            // spin for a while, keeping the operating system thread
            for (int i = 0; i < (x * 100); i++) {
                sum = sum + i;
            }
            return sum;
        } else {
            // yield executing operating system thread back to the operating system
            SystemInterface.yield();
            return 0;
        }
    }

    /**
     * Process references in work queue buffers until empty.
     */
    static void emptyWorkQueue() {

        HeapAddress ref = GCWorkQueue.getFromWorkBuffer();

        if (WORKQUEUE_COUNTS) {
            CollectorThread myThread = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();
            myThread.rootWorkCount = myThread.putWorkCount;
        }

        while (!ref.isNull()) {
            ScanObject.scanObjectOrArray(ref);
            ref = GCWorkQueue.getFromWorkBuffer();
        }
    } // emptyWorkQueue

    // methods for measurement statistics

    static void resetCounters(CollectorThread ct) {
        ct.copyCount = 0;
        ct.rootWorkCount = 0;
        ct.putWorkCount = 0;
        ct.getWorkCount = 0;
        ct.swapBufferCount = 0;
        ct.putBufferCount = 0;
        ct.getBufferCount = 0;
    }

    public static void resetWaitTimes(CollectorThread ct) {
        ct.bufferWaitCount = 0;
        ct.bufferWaitTime = 0.0;
        ct.finishWaitTime = 0.0;
    }

    static void saveAllCounters() {
        int i;
        CollectorThread ct;
        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            saveCounters(ct);
            resetCounters(ct);
        }
    }

    static void resetAllCounters() {
        int i;
        CollectorThread ct;
        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            resetCounters(ct);
        }
    }

    static void saveCounters(CollectorThread ct) {
        ct.copyCount1 = ct.copyCount;
        ct.rootWorkCount1 = ct.rootWorkCount;
        ct.putWorkCount1 = ct.putWorkCount;
        ct.getWorkCount1 = ct.getWorkCount;
        ct.swapBufferCount1 = ct.swapBufferCount;
        ct.putBufferCount1 = ct.putBufferCount;
        ct.getBufferCount1 = ct.getBufferCount;
        resetCounters(ct);
    }

    static void saveAllWaitTimes() {
        int i;
        CollectorThread ct;
        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            saveWaitTimes(ct);
            resetWaitTimes(ct);
        }
    }

    static void resetAllWaitTimes() {
        int i;
        CollectorThread ct;
        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            resetWaitTimes(ct);
        }
    }

    static void saveWaitTimes(CollectorThread ct) {
        ct.bufferWaitCount1 = ct.bufferWaitCount;
        ct.bufferWaitTime1 = ct.bufferWaitTime;
        ct.finishWaitTime1 = ct.finishWaitTime;
        resetWaitTimes(ct);
    }

    static void printAllWaitTimes() {
        int i;
        CollectorThread ct;
        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            Debug.write(i);
            Debug.write(" number of waits ");
            Debug.write(ct.bufferWaitCount1);
            Debug.write("  buffer wait time ");
            Debug.write((int) ((ct.bufferWaitTime1) * 1000000.0));
            Debug.write("(us)  finish wait time ");
            Debug.write((int) ((ct.finishWaitTime1) * 1000000.0));
            Debug.write("(us)\n");
        }
    } // printAllWaitTimes

    static void printAllCounters() {
        // print load balancing work queue counts
        int i;
        CollectorThread ct;

        for (i = 0; i < jq_NativeThread.native_threads.length; i++) {
            ct = (CollectorThread) jq_NativeThread.native_threads[i].getCurrentThread().getJavaLangThreadObject();
            Debug.write(i);
            Debug.write(" copied ");
            Debug.write(ct.copyCount1);
            Debug.write(" roots ");
            Debug.write(ct.rootWorkCount1);
            Debug.write(" puts ");
            Debug.write(ct.putWorkCount1);
            Debug.write(" gets ");
            Debug.write(ct.getWorkCount1);
            Debug.write(" put bufs ");
            Debug.write(ct.putBufferCount1);
            Debug.write(" get bufs ");
            Debug.write(ct.getBufferCount1);
            Debug.write(" swaps ");
            Debug.write(ct.swapBufferCount1);
            Debug.write("\n");
        }
    }

}

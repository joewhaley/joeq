// ScanThreads.java, created Tue Dec 10 14:02:35 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Manager;

import java.util.Iterator;

import Memory.HeapAddress;
import Memory.StackAddress;
import Memory.Heap.Heap;
import Run_Time.Unsafe;
import Scheduler.jq_NativeThread;
import Scheduler.jq_Thread;
import Util.Collections.AppendIterator;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ScanThreads {
    
    static void scanThreads(Heap fromHeap) {
        // get ID of running GC thread
        jq_NativeThread nt = Unsafe.getThreadBlock().getNativeThread();
        int myThreadId = Unsafe.getThreadBlock().getThreadId();
        StackAddress oldstack;

        Iterator i = nt.getReadyQueue(0).threads();
        i = new AppendIterator(i, nt.getIdleQueue().threads());
        i = new AppendIterator(i, nt.getTransferQueue().threads());
        while (i.hasNext()) {
            jq_Thread t = (jq_Thread) i.next();
            HeapAddress ta = HeapAddress.addressOf(t);

            if (t == null) {
                // Nothing to do (no thread object...)
            } else if (t.getThreadId() == myThreadId) {
                // let each GC thread scan its own thread object

                // GC threads are assumed not to have native processors.  if this proves
                // false, then we will have to deal with its write buffers

                // all threads should have been copied out of fromspace earlier

                ScanObject.scanObjectOrArray(t);

                ScanObject.scanObjectOrArray(t.getRegisterState());
                //ScanObject.scanObjectOrArray(t.hardwareExceptionRegisters);

                ScanStack.scanThreadStack(t, HeapAddress.getNull(), true);
                ScanStack.processRoots();

            } else if (false) {
                // skip other collector threads participating (have ordinal number) in this GC
            } else if (true) {
                // have thread to be processed, compete for it with other GC threads

                // all threads should have been copied out of fromspace earlier

                // scan thread object to force "interior" objects to be copied, marked, and
                // queued for later scanning.
                ScanObject.scanObjectOrArray(t);

                // if stack moved, adjust interior stack pointers

                // the above scanThread(t) will have marked and copied the threads JNIEnvironment object,
                // but not have scanned it (likely queued for later scanning).  We force a scan of it now,
                // to force copying of the JNI Refs array, which the following scanStack call will update,
                // and we want to ensure that the updates go into the "new" copy of the array.
                //

                // Likewise we force scanning of the threads contextRegisters, to copy 
                // contextRegisters.gprs where the threads registers were saved when it yielded.
                // Any saved object references in the gprs will be updated during the scan
                // of its stack.
                //
                ScanObject.scanObjectOrArray(t.getRegisterState());
                //ScanObject.scanObjectOrArray(t.hardwareExceptionRegisters);

                // all threads in "unusual" states, such as running threads in
                // SIGWAIT (nativeIdleThreads, nativeDaemonThreads, passiveCollectorThreads),
                // set their ContextRegisters before calling SIGWAIT so that scans of
                // their stacks will start at the caller of SIGWAIT
                //
                // fp = -1 case, which we need to add support for again
                // this is for "attached" threads that have returned to C, but
                // have been given references which now reside in the JNIEnv sidestack

                ScanStack.scanThreadStack(t, HeapAddress.getNull(), true);
                ScanStack.processRoots();
            }
        }
    }

}

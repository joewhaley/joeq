// SemiConservative.java, created Aug 3, 2004 4:18:21 AM by joewhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_StaticField;
import joeq.Class.jq_Type;
import joeq.Memory.Address;
import joeq.Memory.CodeAddress;
import joeq.Memory.HeapAddress;
import joeq.Memory.StackAddress;
import joeq.Runtime.Debug;
import joeq.Runtime.Unsafe;
import joeq.Scheduler.jq_NativeThread;
import joeq.Scheduler.jq_RegisterState;
import joeq.Scheduler.jq_Thread;
import joeq.Scheduler.jq_ThreadQueue;

/**
 * SemiConservative
 * 
 * @author John Whaley
 * @version $Id$
 */
public abstract class SemiConservative {
    
    public static void collect() {
        if (SimpleAllocator.TRACE) Debug.writeln("Starting collection.");
        
        jq_Thread t = Unsafe.getThreadBlock();
        t.disableThreadSwitch();
        
        jq_NativeThread.suspendAllThreads();
        
        if (SimpleAllocator.TRACE) Debug.writeln("Threads suspended.");
        SimpleAllocator s = (SimpleAllocator) DefaultHeapAllocator.def();
        if (SimpleAllocator.TRACE) Debug.writeln("--> Marking roots.");
        scanRoots();
        if (SimpleAllocator.TRACE) Debug.writeln("--> Marking queue.");
        s.scanGCQueue();
        if (SimpleAllocator.TRACE) Debug.writeln("--> Sweeping.");
        s.sweep();
        
        if (SimpleAllocator.TRACE) Debug.writeln("Resuming threads.");
        jq_NativeThread.resumeAllThreads();
        
        t.enableThreadSwitch();
    }
    
    public static void scanRoots() {
        scanStatics();
        scanAllThreads();
    }
    
    /**
     * Scan static variables for object references.
     */
    public static void scanStatics() {
        // todo: other classloaders?
        jq_Type[] types = PrimordialClassLoader.loader.getAllTypes();
        int num = PrimordialClassLoader.loader.getNumTypes();
        for (int i = 0; i < num; ++i) {
            Object o = types[i];
            if (o instanceof jq_Class) {
                jq_Class c = (jq_Class) o;
                if (c.isSFInitialized()) {
                    jq_StaticField[] sfs = c.getDeclaredStaticFields();
                    for (int j=0; j<sfs.length; ++j) {
                        jq_StaticField sf = sfs[j];
                        if (sf.getType().isReferenceType()) {
                            HeapAddress a = sf.getAddress();
                            DefaultHeapAllocator.processObjectReference(a);
                        }
                    }
                }
            }
        }
    }
    
    public static void scanAllThreads() {
        if (jq_NativeThread.allNativeThreadsInitialized()) {
            for (int i = 0; i < jq_NativeThread.native_threads.length; ++i) {
                jq_NativeThread nt = jq_NativeThread.native_threads[i];
                scanQueuedThreads(nt);
                //addObject(nt, b);
            }
        } else {
            jq_NativeThread nt = Unsafe.getThreadBlock().getNativeThread();
            scanQueuedThreads(nt);
        }
        scanCurrentThreadStack(3);
    }
    
    public static void scanQueuedThreads(jq_NativeThread nt) {
        for (int i = 0; i < jq_NativeThread.NUM_OF_QUEUES; ++i) {
            scanThreadQueue(nt.getReadyQueue(i));
        }
        scanThreadQueue(nt.getIdleQueue());
        scanThreadQueue(nt.getTransferQueue());
    }
    
    public static void scanThreadQueue(jq_ThreadQueue q) {
        jq_Thread t = q.peek();
        while (t != null) {
            scanThreadStack(t);
            //addObject(t);
            t = t.getNext();
        }
    }
    
    public static void scanCurrentThreadStack(int skip) {
        StackAddress fp = StackAddress.getBasePointer();
        StackAddress sp = StackAddress.getStackPointer();
        CodeAddress ip = (CodeAddress) fp.offset(HeapAddress.size()).peek();
        while (!fp.isNull()) {
            if (--skip < 0) {
                while (fp.difference(sp) > 0) {
                    addConservativeAddress(sp.peek());
                    sp = (StackAddress) sp.offset(HeapAddress.size());
                }
            }
            ip = (CodeAddress) fp.offset(HeapAddress.size()).peek();
            sp = fp;
            fp = (StackAddress) fp.peek();
        }
    }
    
    public static void scanThreadStack(jq_Thread t) {
        jq_RegisterState s = t.getRegisterState();
        StackAddress fp = s.getEbp();
        CodeAddress ip = s.getEip();
        StackAddress sp = s.getEsp();
        while (!fp.isNull()) {
            while (fp.difference(sp) > 0) {
                addConservativeAddress(sp.peek());
                sp = (StackAddress) sp.offset(HeapAddress.size());
            }
            ip = (CodeAddress) fp.offset(HeapAddress.size()).peek();
            sp = fp;
            fp = (StackAddress) fp.peek();
        }
    }
    
    public static void addConservativeAddress(Address a) {
        // todo: check if address falls in heap
        DefaultHeapAllocator.processObjectReference((HeapAddress) a);
    }
    
    public static boolean isMarked(Object o) {
        int status = HeapAddress.addressOf(o).offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
        return (status & ObjectLayout.GC_BIT) != 0;
    }

}

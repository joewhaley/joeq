// SemiConservative.java, created Aug 3, 2004 4:18:21 AM by joewhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Allocator;

import java.util.Collection;
import java.util.Iterator;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_StaticField;
import joeq.Memory.Address;
import joeq.Memory.CodeAddress;
import joeq.Memory.HeapAddress;
import joeq.Memory.StackAddress;
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
    
    public static void scanRoots(boolean b) {
        scanStatics(b);
        scanAllThreads(b);
    }
    
    /**
     * Scan static variables for object references.
     */
    public static void scanStatics(boolean b) {
        // todo: other classloaders?
        Collection/*<jq_Type>*/ types = PrimordialClassLoader.loader.getAllTypes();
        for (Iterator i = types.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof jq_Class) {
                jq_Class c = (jq_Class) o;
                jq_StaticField[] sfs = c.getDeclaredStaticFields();
                for (int j=0; j<sfs.length; ++j) {
                    jq_StaticField sf = sfs[j];
                    if (sf.getType().isReferenceType()) {
                        HeapAddress a = sf.getAddress();
                        DefaultHeapAllocator.processPtrField(a, b);
                    }
                }
            }
        }
    }
    
    public static void scanAllThreads(boolean b) {
        for (int i = 0; i < jq_NativeThread.native_threads.length; ++i) {
            jq_NativeThread nt = jq_NativeThread.native_threads[i];
            scanQueuedThreads(nt, b);
            //addObject(nt, b);
        }
    }
    
    public static void scanQueuedThreads(jq_NativeThread nt, boolean b) {
        for (int i = 0; i < jq_NativeThread.NUM_OF_QUEUES; ++i) {
            scanThreadQueue(nt.getReadyQueue(i), b);
        }
        scanThreadQueue(nt.getIdleQueue(), b);
        scanThreadQueue(nt.getTransferQueue(), b);
    }
    
    public static void scanThreadQueue(jq_ThreadQueue q, boolean b) {
        jq_Thread t = q.peek();
        while (t != null) {
            scanThreadStack(t, b);
            //addObject(t);
            t = t.getNext();
        }
    }
    
    public static void scanThreadStack(jq_Thread t, boolean b) {
        jq_RegisterState s = t.getRegisterState();
        StackAddress fp = s.getEbp();
        CodeAddress ip = s.getEip();
        StackAddress sp = s.getEsp();
        while (!fp.isNull()) {
            while (fp.difference(sp) > 0) {
                addConservativeAddress(sp.peek(), b);
                sp = (StackAddress) sp.offset(HeapAddress.size());
            }
            ip = (CodeAddress) fp.offset(HeapAddress.size()).peek();
            sp = fp;
            fp = (StackAddress) fp.peek();
        }
    }
    
    public static void addObject(Object o, boolean b) {
        DefaultHeapAllocator.processPtrField(HeapAddress.addressOf(o), b);
    }
    
    public static void addConservativeAddress(Address a, boolean b) {
        DefaultHeapAllocator.processPtrField(a, b);
    }
    
    public static boolean isMarked(Object o) {
        int status = HeapAddress.addressOf(o).offset(ObjectLayout.STATUS_WORD_OFFSET).peek4();
        return (status & ObjectLayout.GC_BIT) != 0;
    }

}

// GCUtil.java, created Tue Dec 10 14:02:22 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Manager;

import joeq.Allocator.ObjectLayoutMethods;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_Primitive;
import joeq.Class.jq_Reference;
import joeq.Memory.HeapAddress;
import joeq.Memory.Heap.BootHeap;
import joeq.Memory.Heap.Heap;
import joeq.Runtime.Debug;
import joeq.Runtime.SystemInterface;
import joeq.Runtime.Unsafe;
import joeq.Scheduler.jq_NativeThread;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class GCUtil {
    
    private final static boolean TRACE = false;

    static Object vtableForArrayType = jq_Array._class.getVTable();
    static Object vtableForClassType = jq_Class._class.getVTable();
    static Object vtableForPrimitiveType = jq_Primitive._class.getVTable();

    public static void boot() {
    }

    public static boolean refInVM(HeapAddress ref) {
        return Heap.refInAnyHeap(ref);
    }

    public static boolean addrInVM(HeapAddress address) {
        return Heap.addrInAnyHeap(address);
    }

    public static boolean refInBootImage(HeapAddress ref) {
        return BootHeap.INSTANCE.refInHeap(ref);
    }

    public static boolean refInHeap(HeapAddress ref) {
        return (refInVM(ref) && (!refInBootImage(ref)));
    }

    public static boolean addrInBootImage(HeapAddress address) {
        return BootHeap.INSTANCE.addrInHeap(address);
    }

    // check if an address appears to point to an instance of VM_Type
    public static boolean validType(HeapAddress typeAddress) {
        if (!refInVM(typeAddress)) {
            return false; // type address is outside of heap
        }

        // check if vtable is one of three possible values
        Object vtable = ObjectLayoutMethods.getVTable(typeAddress.asObject());
        boolean valid = vtable == vtableForClassType ||
                        vtable == vtableForArrayType ||
                        vtable == vtableForPrimitiveType;
        if (!valid)
            Debug.writeln(
                "vtable is invalid: ",
                HeapAddress.addressOf(vtable));
        return valid;
    }

    /**
     * check if a ref, its tib pointer & type pointer are all in the heap
     */
    public static boolean validObject(Object ref) {
        return validRef(HeapAddress.addressOf(ref));
    }

    public static boolean validRef(HeapAddress ref) {

        if (ref.isNull())
            return true;
        if (!refInVM(ref)) {
            Debug.write("validRef: REF outside heap, ref = ");
            Debug.write(ref);
            Debug.write("\n");
            Heap.showAllHeaps();
            return false;
        }

        Object tib = ObjectLayoutMethods.getVTable(ref);
        HeapAddress tibAddr = HeapAddress.addressOf(tib);
        if (!refInVM(tibAddr)) {
            Debug.write("validRef: vtable outside heap, ref = ");
            Debug.write(ref);
            Debug.write(" tib = ");
            Debug.write(tibAddr);
            Debug.write("\n");
            return false;
        }
        if (tibAddr.isNull()) {
            Debug.write("validRef: vtable is Zero! ");
            Debug.write(ref);
            Debug.write("\n");
            return false;
        }

        HeapAddress type = (HeapAddress) tibAddr.peek();
        if (!validType(type)) {
            Debug.write("validRef: invalid TYPE, ref = ");
            Debug.write(ref);
            Debug.write(" tib = ");
            Debug.write(tibAddr);
            Debug.write(" type = ");
            Debug.write(type);
            Debug.write("\n");
            return false;
        }
        return true;
    } // validRef

    public static void dumpRef(HeapAddress ref) {
        Debug.write("REF=");
        if (ref.isNull()) {
            Debug.write("NULL\n");
            return;
        }
        Debug.write(ref);
        if (!refInVM(ref)) {
            Debug.write(" (REF OUTSIDE OF HEAP)\n");
            return;
        }
        //ObjectLayout.dumpHeader(ref);
        HeapAddress tib = HeapAddress.addressOf(ObjectLayoutMethods.getVTable(ref.asObject()));
        if (!refInVM(tib)) {
            Debug.write(" (INVALID TIB: CLASS NOT ACCESSIBLE)\n");
            return;
        }
        jq_Reference type = jq_Reference.getTypeOf(ref.asObject());
        HeapAddress itype = HeapAddress.addressOf(type);
        Debug.write(" TYPE=");
        Debug.write(itype);
        if (!validType(itype)) {
            Debug.write(" (INVALID TYPE: CLASS NOT ACCESSIBLE)\n");
            return;
        }
        Debug.write(" CLASS=");
        type.getDesc().debugWrite();
        Debug.write("\n");
    }

    public static void printclass(HeapAddress ref) {
        if (validRef(ref)) {
            jq_Reference type =
                jq_Reference.getTypeOf(ref.asObject());
            if (validRef(HeapAddress.addressOf(type)))
                type.getDesc().debugWrite();
        }
    }

    static void dumpProcessorsArray() {
        jq_NativeThread st;
        Debug.write("jq_NativeThread.native_threads[]:\n");
        for (int i = 0; i < jq_NativeThread.native_threads.length; ++i) {
            st = jq_NativeThread.native_threads[i];
            Debug.write(" i = ");
            Debug.write(i);
            if (st == null)
                Debug.write(" st is NULL");
            else {
                Debug.write(", id = ");
                Debug.write(st.getIndex());
                Debug.write(", address = ");
                Debug.write(HeapAddress.addressOf(st));
                /*
                Debug.write(", buffer = ");
                Debug.write(HeapAddress.addressOf(st.modifiedOldObjects));
                Debug.write(", top = ");
                Debug.write(st.modifiedOldObjectsTop);
                */
            }
            Debug.write("\n");
        }
    }

    private static final Object outOfMemoryLock = new Object();
    private static boolean outOfMemoryReported = false;

    /**
     * Print OutOfMemoryError message and exit.
     * TODO: make it possible to throw an exception, but this will have
     * to be done without doing further allocations (or by using temp space)
     */
    public static void outOfMemory(String heapName, int heapSize, String commandLine) {
        synchronized (outOfMemoryLock) {
            if (!outOfMemoryReported) {
                outOfMemoryReported = true;
                Unsafe.getThreadBlock().disableThreadSwitch();
                Debug.writeln("\nOutOfMemoryError");
                Debug.write("Failing heap was ");
                Debug.writeln(heapName);
                Debug.writeln("Current heap size = ", heapSize / 1024, " Kb");
                Debug.write("Specify a larger heap using ");
                Debug.writeln(commandLine);
                // call shutdown while holding the processor lock
                SystemInterface.die(-5);
            }
        }
        while (true); // spin until VM shuts down
    }
}

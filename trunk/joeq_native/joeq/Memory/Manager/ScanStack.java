// ScanStack.java, created Tue Dec 10 14:02:32 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Manager;

import java.util.Iterator;

import joeq.Allocator.CodeAllocator;
import joeq.Allocator.DefaultHeapAllocator;
import joeq.Bootstrap.PrimordialClassLoader;
import joeq.Clazz.jq_Class;
import joeq.Clazz.jq_CompiledCode;
import joeq.Clazz.jq_Method;
import joeq.Clazz.jq_StaticField;
import joeq.Memory.Address;
import joeq.Memory.CodeAddress;
import joeq.Memory.HeapAddress;
import joeq.Memory.StackAddress;
import joeq.Run_Time.Unsafe;
import joeq.Scheduler.jq_Thread;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ScanStack {

    // quietly validates each ref reported by map iterators
    static final boolean VALIDATE_STACK_REFS = true;

    // debugging options to produce printout during scanStack
    // MULTIPLE GC THREADS WILL PRODUCE SCRAMBLED OUTPUT so only
    // use these when running with PROCESSORS=1

    // includes in output a dump of the contents of each frame
    // forces DUMP_STACK_REFS & TRACE_STACKS on (ie. everything!!)
    static final boolean DUMP_STACK_FRAMES = false;

    // includes in output the refs reported by map iterators
    // forces TRACE_STACKS on 
    static final boolean DUMP_STACK_REFS = DUMP_STACK_FRAMES || false;

    // outputs names of methods as their frames are scanned
    static final boolean TRACE_STACKS = DUMP_STACK_REFS || false;

    static int stackDumpCount = 0;
    /**
     * Scans a threads stack during collection to find object references.
     * Locates and updates references in stack frames using stack maps,
     * and references associated with JNI native frames.  Located references
     * are processed by calling VM_Allocator.processPtrField.
     * <p>
     * If relocate_code is true, moves code objects, and updates saved
     * link registers in the stack frames.
     *
     * @param t              VM_Thread for the thread whose stack is being scanned
     * @param top_frame      address of stack frame at which to begin the scan
     * @param relocate_code  if true, relocate code & update return addresses
     */
    public static void scanThreadStack(jq_Thread t, Address top_frame, boolean relocate_code) {
        
        CodeAddress ip, newip, code, newcode;
        StackAddress fp;
        Address prevFp;
        Address refaddr;
        int delta;
        jq_Method method;
        jq_CompiledCode compiledMethod;
        GCMapIterator iterator;
        GCMapIteratorGroup iteratorGroup;

        // Before scanning the thread's stack, copy forward any machine code that is
        // referenced by the thread's hardwareExceptionRegisters, if they are in use.
        //
        
        // get gc thread local iterator group from our VM_CollectorThread object
        CollectorThread collector = (CollectorThread) Unsafe.getThreadBlock().getJavaLangThreadObject();
        iteratorGroup = collector.iteratorGroup;
        iteratorGroup.newStackWalk(t);

        if (!top_frame.isNull()) {
            prevFp = (StackAddress)top_frame;
            // start scan at caller of passed in fp
            ip = (CodeAddress) top_frame.offset(4).peek();
            fp = (StackAddress) top_frame.peek();
        } else {
            prevFp = HeapAddress.getNull();
            // start scan using fp & ip in threads saved context registers
            ip = t.getRegisterState().getEip();
            fp = t.getRegisterState().getEbp();
        }

        if (!fp.isNull()) {

            // At start of loop:
            //   fp -> frame for method invocation being processed
            //   ip -> instruction pointer in the method (normally a call site)

            while (!fp.peek().isNull()) {
                
                // following is for normal Java (and JNI Java to C transition) frames

                compiledMethod = CodeAllocator.getCodeContaining(ip);
                method = compiledMethod.getMethod();

                // initialize MapIterator for this frame
                int offset = ip.difference(compiledMethod.getEntrypoint());
                iterator = iteratorGroup.selectIterator(compiledMethod);
                iterator.setupIterator(compiledMethod, offset, fp);

                // scan the map for this frame and process each reference
                //
                for (refaddr = iterator.getNextReferenceAddress();
                    !refaddr.isNull();
                    refaddr = iterator.getNextReferenceAddress()) {

                    DefaultHeapAllocator.processPtrField(refaddr);
                }

                iterator.cleanupPointers();

                // if at a JNIFunction method, it is preceeded by native frames that must be skipped
                //
                
                // set fp & ip for next frame
                prevFp = fp;
                ip = (CodeAddress) fp.offset(4).peek();
                fp = (StackAddress) fp.peek();

            } // end of while != sentinel

        } // end of if (fp != STACKFRAME_SENTINAL_FP)

    } //gc_scanStack

    public static void processRoots() {
        for (Iterator i=PrimordialClassLoader.loader.getAllTypes().iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (o instanceof jq_Class) {
                jq_Class c = (jq_Class) o;
                jq_StaticField[] sfs = c.getDeclaredStaticFields();
                for (int j=0; j<sfs.length; ++j) {
                    jq_StaticField sf = sfs[j];
                    if (sf.getType().isReferenceType()) {
                        HeapAddress addr = sf.getAddress();
                        DefaultHeapAllocator.processPtrField(addr);
                    }
                }
            }
        }
    }
}

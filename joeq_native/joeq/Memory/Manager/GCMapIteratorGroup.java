// GCMapIteratorGroup.java, created Tue Dec 10 14:02:30 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Manager;

import Clazz.jq_CompiledCode;
import Memory.Address;
import Scheduler.jq_RegisterState;
import Scheduler.jq_Thread;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class GCMapIteratorGroup {

    /** current location (memory address) of each gpr register */
    private Address[] registerLocations;

    /** iterator for VM_BootImageCompiler stackframes */
    private GCMapIterator bootImageCompilerIterator;

    /** iterator for VM_RuntimeCompiler stackframes */
    private GCMapIterator runtimeCompilerIterator;

    /** iterator for VM_HardwareTrap stackframes */
    private GCMapIterator hardwareTrapIterator;

    /** iterator for fallback compiler (baseline) stackframes */
    private GCMapIterator fallbackCompilerIterator;

    /** iterator for test compiler (opt) stackframes */
    private GCMapIterator testOptCompilerIterator;

    /** iterator for JNI Java -> C  stackframes */
    private GCMapIterator jniIterator;

    GCMapIteratorGroup() {
        /*
        registerLocations = new int[VM_Constants.NUM_GPRS];
        bootImageCompilerIterator =
            VM_BootImageCompiler.createGCMapIterator(registerLocations);
        runtimeCompilerIterator =
            VM_RuntimeCompiler.createGCMapIterator(registerLocations);
        hardwareTrapIterator =
            new VM_HardwareTrapGCMapIterator(registerLocations);
        fallbackCompilerIterator =
            new VM_BaselineGCMapIterator(registerLocations);
        jniIterator = new VM_JNIGCMapIterator(registerLocations);
        */
    }

    /**
     * Prepare to scan a thread's stack for object references.
     * Called by collector threads when beginning to scan a threads stack.
     * Calls newStackWalk for each of the contained GCMapIterators.
     * <p>
     * Assumption:  the thread is currently suspended, ie. its saved gprs[]
     * contain the thread's full register state.
     * <p>
     * Side effect: registerLocations[] initialized with pointers to the
     * thread's saved gprs[] (in thread.contextRegisters.gprs)
     * <p>
     * @param thread  VM_Thread whose registers and stack are to be scanned
     */
    void newStackWalk(jq_Thread thread) {
        jq_RegisterState rs = thread.getRegisterState();
        /*
        Address registerLocation = 
            VM_Magic.objectAsAddress(thread.contextRegisters.gprs);
        for (int i = 0; i < VM_Constants.NUM_GPRS; ++i) {
            registerLocations[i] = registerLocation.toInt();
            registerLocation = registerLocation.offset(4);
        }
        bootImageCompilerIterator.newStackWalk(thread);
        runtimeCompilerIterator.newStackWalk(thread);
        hardwareTrapIterator.newStackWalk(thread);
        fallbackCompilerIterator.newStackWalk(thread);
        if (testOptCompilerIterator != null)
            testOptCompilerIterator.newStackWalk(thread);
        if (jniIterator != null)
            jniIterator.newStackWalk(thread);
            */
    }

    /**
     * Select iterator for scanning for object references in a stackframe.
     * Called by collector threads while scanning a threads stack.
     *
     * @param compiledMethod  VM_CompiledMethod for the method executing
     *                        in the stack frame
     *
     * @return GCMapIterator to use
     */
    GCMapIterator selectIterator(jq_CompiledCode compiledMethod) {
        
        /*
        int type = compiledMethod.getCompilerType();

        if (type == bootImageCompilerIterator.getType())
            return bootImageCompilerIterator;

        if (type == runtimeCompilerIterator.getType())
            return runtimeCompilerIterator;

        if (type == hardwareTrapIterator.getType())
            return hardwareTrapIterator;

        if (type == fallbackCompilerIterator.getType())
            return fallbackCompilerIterator;

        if (jniIterator != null && type == jniIterator.getType())
            return jniIterator;

        if (testOptCompilerIterator != null
            && type == testOptCompilerIterator.getType())
            return testOptCompilerIterator;

        if (VM.VerifyAssertions)
            VM._assert(VM.NOT_REACHED);
            */
        return null;
        
    }

    /**
     * get the GCMapIterator used for scanning JNI native stack frames.
     *
     * @return jniIterator
     */
    GCMapIterator getJniIterator() {
        return jniIterator;
    }

}

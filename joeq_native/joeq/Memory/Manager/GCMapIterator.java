// GCMapIterator.java, created Tue Dec 10 14:02:28 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory.Manager;

import joeq.Class.jq_CompiledCode;
import joeq.Memory.StackAddress;
import joeq.Scheduler.jq_Thread;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class GCMapIterator {
    
    /** thread whose stack is currently being scanned */
    public jq_Thread thread;

    /** address of stackframe currently being scanned */
    public StackAddress framePtr;

    /** address where each gpr register was saved by previously scanned stackframe(s) */
    public StackAddress[] registerLocations;

    /**
     * Prepare to scan a thread's stack and saved registers for object references.
     *
     * @param thread jq_Thread whose stack is being scanned
     */
    public void newStackWalk(jq_Thread thread) {
        this.thread = thread;
    }

    /**
     * Prepare to iterate over object references and JSR return addresses held by a stackframe.
     * 
     * @param compiledMethod     method running in the stackframe
     * @param instructionOffset  offset of current instruction within that method's code
     * @param framePtr           address of stackframe to be visited
     */
    public abstract void setupIterator(
        jq_CompiledCode compiledMethod,
        int instructionOffset,
        StackAddress framePtr);

    /**
     * Get address of next object reference held by current stackframe.
     * Returns zero when there are no more references to report.
     * <p>
     * Side effect: registerLocations[] updated at end of iteration.
     * TODO: registerLocations[] update should be done via separately called
     * method instead of as side effect.
     * <p>
     *
     * @return address of word containing an object reference
     *         zero if no more references to report
     */
    public abstract StackAddress getNextReferenceAddress();

    /**
     * Get address of next JSR return address held by current stackframe.
     *
     * @return address of word containing a JSR return address
     *         zero if no more return addresses to report
     */
    public abstract StackAddress getNextReturnAddressAddress();

    /**
     * Prepare to re-iterate on same stackframe, and to switch between
     * "reference" iteration and "JSR return address" iteration.
     */
    public abstract void reset();

    /**
     * Iteration is complete, release any internal data structures including 
     * locks acquired during setupIterator for jsr maps.
     * 
     */
    public abstract void cleanupPointers();

    /**
     * Get the type of this iterator (BASELINE, OPT, etc.).
     * Called from GCMapIteratorGroup to select which iterator
     * to use for a stackframe.  The possible types are specified 
     * in jq_CompiledCode.
     *
     * @return type code for this iterator
     */
    public abstract int getType();

}

// TraceRootSet.java, created Mon Sep 23  8:11:56 2002 by laudney
// Copyright (C) 2001-3 laudney <laudney@acm.org>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package GC;

import java.util.ArrayList;

import Scheduler.jq_RegisterState;

/**
 * TraceRootSet
 *
 * starting points for tracing algorithm used by Mark-and-Sweep garbage collectors,
 * which include thread stacks, statics, interned strings and JNI references.
 *
 * @author laudney <laudney@acm.org>
 * @version $Id$
 */
class TraceRootSet {

    static /*final*/ boolean TRACE = false;

    private ArrayList suspendedThreadStates = new ArrayList();

    boolean add(Object o) {
        if (!(o instanceof jq_RegisterState)) {
            throw new IllegalArgumentException();
        }
        return suspendedThreadStates.add(o);
    }
}

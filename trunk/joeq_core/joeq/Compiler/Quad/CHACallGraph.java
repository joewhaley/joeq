// CHACallGraph.java, created Mon Mar  3 18:01:33 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Quad;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Set;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import joeq.Compiler.Analysis.IPA.*;

/**
 * A simple call graph implementation based on class-hierarchy analysis with
 * optional rapid type analysis.
 * 
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class CHACallGraph extends CallGraph {

    public static final CHACallGraph INSTANCE = new CHACallGraph();

    protected final Set classes;

    /**
     * Construct a call graph assuming only the given types are
     * used by the program, i.e. rapid type analysis.
     * 
     * @param classes set of types from which to build the call graph
     */
    public CHACallGraph(Set/*jq_Type*/ classes) { this.classes = classes; }
    protected CHACallGraph() { this.classes = null; }

    /**
     * @see joeq.Compiler.Quad.CallGraph#getTargetMethods(java.lang.Object, joeq.Compiler.Quad.ProgramLocation)
     */
    public Collection getTargetMethods(Object context, ProgramLocation callSite) {
        jq_Method method = (jq_Method) callSite.getTargetMethod();
        if (callSite.isSingleTarget())
            return Collections.singleton(method);
        
        Collection result;
        if (callSite.isInterfaceCall()) {
            result = new LinkedHashSet();
            Collection s = classes!=null?classes:PrimordialClassLoader.loader.getAllTypes();
            for (Iterator i=new ArrayList(s).iterator(); i.hasNext(); ) {
                jq_Type t = (jq_Type) i.next();
                if (t instanceof jq_Class) {
                    jq_Class c = (jq_Class) t;
                    c.prepare();
                    if (c.implementsInterface(method.getDeclaringClass())) {
                        jq_Method m2 = c.getVirtualMethod(method.getNameAndDesc());
                        if (m2 != null && !m2.isAbstract()) result.add(m2);
                    }
                }
            }
        } else {
            result = new LinkedList();
            LinkedList worklist = new LinkedList();
            worklist.add(method.getDeclaringClass());
            while (!worklist.isEmpty()) {
                jq_Class c = (jq_Class) worklist.removeFirst();
                c.load();
                jq_Method m2 = (jq_Method) c.getDeclaredMember(method.getNameAndDesc());
                if (m2 != null) {
                    if (!m2.isAbstract()) {
                        result.add(m2);
                    }
                    if (m2.isFinal() || m2.isPrivate()) {
                        continue;
                    }
                }
                for (Iterator i=Arrays.asList(c.getSubClasses()).iterator(); i.hasNext(); ) {
                    jq_Class c2 = (jq_Class) i.next();
                    if (classes == null || classes.contains(c2)) worklist.add(c2);
                }
            }
        }
        return result;
    }

    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.CallGraph#getRoots()
     */
    public Collection getRoots() {
        throw new UnsupportedOperationException();
    }

    /* (non-Javadoc)
     * @see joeq.Compiler.Quad.CallGraph#setRoots(java.util.Collection)
     */
    public void setRoots(Collection roots) {
        throw new UnsupportedOperationException();
    }

}

/*
 * CHACallGraph.java
 * 
 * Created on Mar 3, 2003
 */
package Compil3r.Quad;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Set;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;

/**
 * A simple call graph implementation based on class-hierarchy analysis with
 * optional rapid type analysis.
 * 
 * @author John Whaley
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
     * @see Compil3r.Quad.CallGraph#getTargetMethods(java.lang.Object, Compil3r.Quad.ProgramLocation)
     */
    public Collection getTargetMethods(Object context, ProgramLocation callSite) {
        jq_Method method = (jq_Method) callSite.getTargetMethod();
        if (callSite.isSingleTarget())
            return Collections.singleton(method);
        
        Collection result;
        if (callSite.isInterfaceCall()) {
            result = new LinkedHashSet();
            Set s = classes!=null?classes:PrimordialClassLoader.loader.getAllTypes();
            for (Iterator i=s.iterator(); i.hasNext(); ) {
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
     * @see Compil3r.Quad.CallGraph#getRoots()
     */
    protected Collection getRoots() {
        throw new UnsupportedOperationException();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#setRoots(java.util.Collection)
     */
    public void setRoots(Collection roots) {
        throw new UnsupportedOperationException();
    }

}

// LoadedCallGraph.java, created Jun 27, 2003 12:46:40 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import Util.Assert;
import Util.Collections.GenericInvertibleMultiMap;
import Util.Collections.GenericMultiMap;
import Util.Collections.InvertibleMultiMap;
import Util.Collections.MultiMap;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Compil3r.Analysis.IPA.*;
import Compil3r.Analysis.IPA.ProgramLocation.BCProgramLocation;

/**
 * A call graph that is loaded from a file.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class LoadedCallGraph extends CallGraph {

    public static void write(CallGraph cg, DataOutput out) throws IOException {
        Assert._assert(cg.getAllMethods().containsAll(cg.getRoots()));
        MultiMap classesToMethods = new GenericMultiMap();
        for (Iterator i = cg.getAllMethods().iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            classesToMethods.add(m.getDeclaringClass(), m);
        }
        for (Iterator i = classesToMethods.keySet().iterator(); i.hasNext(); ) {
            jq_Class klass = (jq_Class) i.next();
            out.writeBytes("CLASS "+klass.getJDKName().replace('.', '/')+"\n");
            for (Iterator j = classesToMethods.getValues(klass).iterator(); j.hasNext(); ) {
                jq_Method m = (jq_Method) j.next();
                out.writeBytes(" METHOD "+m.getName()+" "+m.getDesc());
                if (cg.getRoots().contains(m)) out.writeBytes(" ROOT\n");
                else out.writeByte('\n');
                for (Iterator k = cg.getCallSites(m).iterator(); k.hasNext(); ) {
                    ProgramLocation pl = (ProgramLocation) k.next();
                    out.writeBytes("  CALLSITE "+pl.getBytecodeIndex()+"\n");
                    for (Iterator l = cg.getTargetMethods(pl).iterator(); l.hasNext(); ) {
                        jq_Method target = (jq_Method) l.next();
                        out.writeBytes("   TARGET "+target.getDeclaringClass().getJDKName().replace('.', '/')+"."+target.getName()+" "+target.getDesc()+"\n");
                    }
                }
            }
        }
    }

    protected Set/*<jq_Method>*/ methods;
    protected Set/*<jq_Method>*/ roots;
    protected MultiMap/*<jq_Method,Integer>*/ callSites;
    protected InvertibleMultiMap/*<ProgramLocation,jq_Method>*/ edges;

    public LoadedCallGraph(String filename) throws IOException {
        this.methods = new LinkedHashSet();
        this.roots = new LinkedHashSet();
        this.callSites = new GenericMultiMap();
        this.edges = new GenericInvertibleMultiMap();
        DataInput in = new DataInputStream(new FileInputStream(filename));
        read(in);
    }
    
    protected void read(DataInput in) throws IOException {
        jq_Class k = null;
        jq_Method m = null;
        int bcIndex = -1;
        for (;;) {
            String s = in.readLine();
            if (s == null)
                break;
            s.trim();
            StringTokenizer st = new StringTokenizer(s, ". ");
            if (!st.hasMoreTokens())
                break;
            String id = st.nextToken();
            if (id.equals("CLASS")) {
                if (!st.hasMoreTokens())
                    throw new IOException();
                String className = st.nextToken();
                k = (jq_Class) jq_Type.parseType(className);
                k.load();
                continue;
            }
            if (id.equals("METHOD")) {
                if (!st.hasMoreTokens())
                    throw new IOException();
                String methodName = st.nextToken();
                if (!st.hasMoreTokens())
                    throw new IOException();
                String methodDesc = st.nextToken();
                m = (jq_Method) k.getDeclaredMember(methodName, methodDesc);
                if (m == null) {
                    System.err.println("Cannot find \""+methodName+"\" \""+methodDesc+"\" in "+k);
                    continue;
                }
                methods.add(m);
                if (st.hasMoreTokens()) {
                    String arg = st.nextToken();
                    if (arg.equals("ROOT")) roots.add(m);
                }
                continue;
            }
            if (id.equals("CALLSITE")) {
                if (!st.hasMoreTokens())
                    throw new IOException();
                String num = st.nextToken();
                bcIndex = Integer.parseInt(num);
                continue;
            }
            if (id.equals("TARGET")) {
                if (!st.hasMoreTokens())
                    throw new IOException();
                String className = st.nextToken();
                if (!st.hasMoreTokens())
                    throw new IOException();
                String methodName = st.nextToken();
                if (!st.hasMoreTokens())
                    throw new IOException();
                String methodDesc = st.nextToken();
                jq_Class targetClass = (jq_Class) jq_Type.parseType(className);
                targetClass.load();
                jq_Method targetMethod = (jq_Method) targetClass.getDeclaredMember(methodName, methodDesc);
                if (m == null) {
                    // reported above.
                    continue;
                }
                if (targetMethod == null) {
                    System.err.println("Cannot find \""+methodName+"\" \""+methodDesc+"\" in "+targetClass);
                    continue;
                }
                Assert._assert(targetMethod != null);
                add(m, bcIndex, targetMethod);
                continue;
            }
        }
    }

    public void add(jq_Method caller, int bcIndex, jq_Method callee) {
        ProgramLocation pl = new BCProgramLocation(caller, bcIndex);
        callSites.add(caller, pl);
        edges.add(pl, callee);
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#setRoots(java.util.Collection)
     */
    public void setRoots(Collection roots) {
        // Root set should be the same!
        Assert._assert(this.roots.equals(roots));
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getRoots()
     */
    public Collection getRoots() {
        return roots;
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getTargetMethods(java.lang.Object, Compil3r.Quad.ProgramLocation)
     */
    public Collection getTargetMethods(Object context, ProgramLocation callSite) {
        if (callSite instanceof ProgramLocation.QuadProgramLocation) {
            jq_Method m = (jq_Method) callSite.getMethod();
            Map map = CodeCache.getBCMap(m);
            int bcIndex = ((Integer) map.get(((ProgramLocation.QuadProgramLocation) callSite).getQuad())).intValue();
            callSite = new BCProgramLocation(m, bcIndex);
        }
        return edges.getValues(callSite);
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#entrySet()
     */
    public Set entrySet() {
        return edges.entrySet();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getAllCallSites()
     */
    public Collection getAllCallSites() {
        return edges.keySet();
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getAllMethods()
     */
    public Collection getAllMethods() {
        return methods;
    }

    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallees(Clazz.jq_Method)
     */
    public Collection getCallees(jq_Method caller) {
        Collection c = CachedCallGraph.getFromMultiMap(callSites, edges, caller);
        return c;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallers(Clazz.jq_Method)
     */
    public Collection getCallers(jq_Method callee) {
        MultiMap m1 = edges.invert();
        Collection c1 = m1.getValues(callee);
        return c1;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallerMethods(Clazz.jq_Method)
     */
    public Collection getCallerMethods(jq_Method callee) {
        MultiMap m1 = edges.invert();
        Collection c1 = m1.getValues(callee);
        Iterator i = c1.iterator();
        if (!i.hasNext()) {
            return Collections.EMPTY_SET;
        }
        ProgramLocation o = (ProgramLocation) i.next();
        if (!i.hasNext()) {
            return Collections.singleton(o.getMethod());
        }
        Set result = new LinkedHashSet();
        for (;;) {
            result.add(o.getMethod());
            if (!i.hasNext()) break;
            o = (ProgramLocation) i.next();
        }
        return result;
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Quad.CallGraph#getCallSites(Clazz.jq_Method)
     */
    public Collection getCallSites(jq_Method caller) {
        Collection c = callSites.getValues(caller);
        return c;
    }

    /* (non-Javadoc)
     * @see java.util.AbstractMap#keySet()
     */
    public Set keySet() {
        return edges.keySet();
    }

}

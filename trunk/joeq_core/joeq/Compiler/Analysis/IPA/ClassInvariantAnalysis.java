// ClassInvariantAnalysis.java, created Jun 20, 2003 9:22:16 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Initializer;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Primitive;
import Clazz.jq_Type;
import Clazz.jq_TypeVisitor;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.FieldNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.NodeSet;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Util.Assert;
import Util.Graphs.Navigator;

/**
 * ClassInvariantAnalysis
 * 
 * @author John Whaley
 * @version $Id$
 */
public class ClassInvariantAnalysis
    implements jq_TypeVisitor {

    public static final boolean TRACE = true;
    public static final boolean TRACE_INTRA = true;
    public static final PrintStream out = System.out;

    MethodSummary summary;
    ParamNode dis;
    Map returned;
    Map thrown;

    public void initialize(jq_Class k) {
        jq_InstanceMethod[] ms = k.getDeclaredInstanceMethods();
        jq_Initializer init = null;
        for (int i=0; i<ms.length; ++i) {
            if (ms[i] instanceof jq_Initializer) {
                init = (jq_Initializer) ms[i];
                break;
            }
        }
        dis = ParamNode.get(init, 0, k);
        summary = new MethodSummary(new ParamNode[] { dis });
        returned = new HashMap();
        thrown = new HashMap();
    }

    static MethodSummary getSummary(jq_Method m) {
        if (m.getBytecode() == null) {
            return null;
        }
        ControlFlowGraph cfg = CodeCache.getCode(m);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        return ms;
    }

    static class LocalCallGraphNavigator implements Navigator {

        jq_Class klass;
        CallGraph cg;

        /**
         * @see Util.Graphs.Navigator#next(java.lang.Object)
         */
        public Collection next(Object node) {
            jq_Method caller = (jq_Method) node;
            Collection callees = cg.getCallees(caller);
            LinkedList ll = new LinkedList();
            for (Iterator i = callees.iterator(); i.hasNext(); ) {
                jq_Method callee = (jq_Method) i.next();
                if (klass.isSubtypeOf(callee.getDeclaringClass())) {
                    ll.add(callee);
                }
            }
            return ll;
        }

        /**
         * @see Util.Graphs.Navigator#prev(java.lang.Object)
         */
        public Collection prev(Object node) {
            jq_Method callee = (jq_Method) node;
            Collection s = cg.getCallerMethods(callee);
            LinkedList ll = new LinkedList();
            for (Iterator i = s.iterator(); i.hasNext(); ) {
                jq_Method caller = (jq_Method) i.next();
                if (caller.getDeclaringClass().isSubtypeOf(klass)) {
                    ll.add(caller);
                }
            }
            return s;
        }
        
    }

    public void instantiateLocalCalls(jq_Method m) {
    }

    public void visitMethod(jq_Method m) {
        if (m.getBytecode() == null) {
            if (TRACE) out.println("NOTE: "+m.getName()+"() is a native method, we don't know what goes on in there.");
            return;
        }
        if (m.isPrivate()) {
            //if (TRACE) out.println(m.getName()+"() is private, skipping.");
            //return;
        }
        ControlFlowGraph cfg = CodeCache.getCode(m);
        MethodSummary ms = MethodSummary.getSummary(cfg);
        if (m.isStatic()) {
            if (TRACE) out.println(m.getName()+"() is static.");
            // TODO.
        } else {
            ParamNode n = ms.getParamNode(0);
            n.replaceBy(Collections.singleton(dis), true);
            
            NodeSet s1 = new NodeSet(ms.getReturned());
            if (s1.remove(n)) s1.add(dis);
            returned.put(m, s1);
            
            NodeSet s2 = new NodeSet(ms.getThrown());
            if (s2.remove(n)) s2.add(dis);
            thrown.put(m, s2);
        }
    }

    public void unifyAccessPathEdges(Node n) {
        if (n instanceof UnknownTypeNode) return;
        if (TRACE_INTRA) out.println("Unifying access path edges from: "+n);
        if (n.hasAccessPathEdges()) {
            for (Iterator i=n.getAccessPathEdges().iterator(); i.hasNext(); ) {
                java.util.Map.Entry e = (java.util.Map.Entry)i.next();
                jq_Field f = (jq_Field)e.getKey();
                Object o = e.getValue();
                Assert._assert(o != null);
                FieldNode n2;
                if (o instanceof FieldNode) {
                    n2 = (FieldNode)o;
                } else {
                    Set s = (Set)NodeSet.FACTORY.makeSet((Set)o);
                    if (s.size() == 0) {
                        i.remove();
                        continue;
                    }
                    if (s.size() == 1) {
                        n2 = (FieldNode)s.iterator().next();
                        e.setValue(n2);
                        continue;
                    }
                    if (TRACE_INTRA) out.println("Node "+n+" has duplicate access path edges on field "+f+": "+s);
                    n2 = FieldNode.unify(f, s);
                    for (Iterator j=s.iterator(); j.hasNext(); ) {
                        FieldNode n3 = (FieldNode)j.next();
                        for (Iterator k=returned.values().iterator(); k.hasNext(); ) {
                            Set s2 = (Set) k.next();
                            if (s2.remove(n3)) {
                                s2.add(n2);
                            }
                        }
                        for (Iterator k=thrown.values().iterator(); k.hasNext(); ) {
                            Set s2 = (Set) k.next();
                            if (s2.remove(n3)) {
                                s2.add(n2);
                            }
                        }
                        //nodes.remove(n3);
                    }
                    //nodes.put(n2, n2);
                    e.setValue(n2);
                }
            }
        }
    }
    
    public void finish() {
        unifyAccessPathEdges(dis);
        for (Iterator i=dis.getAccessPathEdges().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            jq_Field f = (jq_Field) e.getKey();
            System.out.println("Field "+f.getName()+" = "+dis.getAllEdges(f));
        }
        for (Iterator i=returned.entrySet().iterator(); i.hasNext(); ) {
            Map.Entry e = (Map.Entry) i.next();
            jq_Method f = (jq_Method) e.getKey();
            System.out.println("Method "+f.getName()+" returns "+e.getValue());
        }
    }

    /* (non-Javadoc)
     * @see Clazz.jq_TypeVisitor#visitClass(Clazz.jq_Class)
     */
    public void visitClass(jq_Class c) {
        this.initialize(c);
        for (Iterator i=Arrays.asList(c.getDeclaredInstanceMethods()).iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            visitMethod(m);
        }
        this.finish();
    }

    /* (non-Javadoc)
     * @see Clazz.jq_TypeVisitor#visitArray(Clazz.jq_Array)
     */
    public void visitArray(jq_Array m) {}

    /* (non-Javadoc)
     * @see Clazz.jq_TypeVisitor#visitPrimitive(Clazz.jq_Primitive)
     */
    public void visitPrimitive(jq_Primitive m) {}

    /* (non-Javadoc)
     * @see Clazz.jq_TypeVisitor#visitType(Clazz.jq_Type)
     */
    public void visitType(jq_Type m) {}

}

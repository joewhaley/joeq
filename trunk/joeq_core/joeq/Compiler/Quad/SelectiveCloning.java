// SelectiveCloning.java, created Sun Nov 10 10:35:17 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.io.PrintStream;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Clazz.jq_Method;
import Compil3r.Quad.AndersenInterface.AndersenField;
import Compil3r.Quad.AndersenInterface.AndersenMethod;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.FieldNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.OutsideNode;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.PointerExplorer.InlineSet;
import Util.Assert;
import Util.Collections.LinearSet;
import Util.Collections.Pair;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class SelectiveCloning {

    public static AndersenPointerAnalysis pa;
    
    public static boolean TRACE = false;
    public static boolean PRINT_INLINE = true;
    public static PrintStream out = System.out;
    
    public static boolean FIND_ALL = true;
    
    public static void searchForCloningOpportunities(Map/*<CallSite,Set<jq_Method>>*/ toInline,
                                                     Set/*<CallSite>*/ selectedCallSites) {
        out.println("Searching for cloning opportunities for "+selectedCallSites.size()+" call sites");
        LinkedList/*<Node>*/ node_worklist = new LinkedList();
        HashMap/*<Node,Collection<ProgramLocation>>*/ visited = new HashMap();
        for (Iterator i=selectedCallSites.iterator(); i.hasNext(); ) {
            CallSite cs = (CallSite)i.next();
            ProgramLocation mc = cs.m;
            MethodSummary ms = cs.caller;
            PassedParameter pp = new PassedParameter(mc, 0);
            LinkedHashSet set = new LinkedHashSet();
            ms.getNodesThatCall(pp, set);
            if (TRACE) System.out.println("Call site: "+cs);
            for (Iterator it2 = set.iterator(); it2.hasNext(); ) {
                Node node = (Node) it2.next();
                if (node instanceof OutsideNode) {
                    if (TRACE) System.out.println("Outside node: "+node);
                    Collection/*<ProgramLocation>*/ s = (Collection) visited.get(node);
                    if (s == null) {
                        visited.put(node, s = new LinkedList());
                        node_worklist.add(node);
                    }
                    s.add(mc);
                }
            }
        }
        System.out.println("Root set of nodes to check: "+node_worklist.size());
        
outer:
        while (!node_worklist.isEmpty()) {
            Node n = (Node) node_worklist.removeFirst();
            Collection/*<ProgramLocation>*/ mc = (Collection) visited.get(n);
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            
            Set concreteNodes = null;
            LinkedHashSet exact_types = null;
            int num_exact_types = -1;
            boolean has_nonexact_types = false;
            Map/*<ProgramLocation,Set<jq_Method>>*/ exact_targets_map = null;
            if (outEdges.size() > 1) {
                if (n instanceof ParamNode) {
                    if (TRACE) out.println("found parameter node "+n+" with outdegree "+outEdges.size());
                    concreteNodes = pa.getConcreteNodes(n);
                    exact_types = new LinkedHashSet();
                    has_nonexact_types = false;
                    for (Iterator i=concreteNodes.iterator(); i.hasNext(); ) {
                        Node n3 = (Node)i.next();
                        if (n3 instanceof ConcreteTypeNode) {
                            if (n3.getDeclaredType() != null)
                                exact_types.add(n3.getDeclaredType());
                        } else {
                            has_nonexact_types = true;
                        }
                    }
                    if (TRACE) out.println("exact types of concrete nodes: "+exact_types);
                    if (TRACE) out.println("has nonexact types="+has_nonexact_types);
                    num_exact_types = exact_types.size();
                    if (!has_nonexact_types && num_exact_types <= 1) {
                        if (TRACE) out.println("only one type, skipping");
                        continue;
                    } else if (has_nonexact_types && num_exact_types == 0) {
                        if (TRACE) out.println("only nonexact types, skipping");
                        continue;
                    }
                } else if (n instanceof FieldNode) {
                    boolean b = TRACE;
                    TRACE = true;
                    FieldNode n3 = (FieldNode) n;
                    LinkedList ap = new LinkedList();
                    for (;;) {
                        ap.addFirst(n3);
                        Set outEdges3 = (Set) pa.nodeToInclusionEdges.get(n3);
                        if (outEdges3 == null || outEdges3.size() <= 1) break;
                        Node n4 = (Node) outEdges3.iterator().next();
                        if (n4 instanceof OutsideNode)
                            while (((OutsideNode)n4).skip != null)
                                n4 = ((OutsideNode)n4).skip;
                        Object reason = pa.edgesToReasons.get(new Pair(n3, n4));
                        if (reason instanceof FieldNode) {
                            if (TRACE) out.println("found field node "+n3+" with outdegree "+outEdges3.size());
                            if (ap.contains(reason)) {
                                if (TRACE) out.println("cyclic access path "+reason+", skipping.");
                                break;
                            }
                            n3 = (FieldNode) reason;
                            continue;
                        } else if (reason instanceof ParamNode) {
                            if (TRACE) out.println("found field node "+n3+" with outdegree "+outEdges3.size());
                            n4 = (ParamNode) reason;
                            outEdges3 = (Set) pa.nodeToInclusionEdges.get(n4);
                            if (outEdges3 == null || outEdges3.size() <= 1) break;
                            if (TRACE) out.println("found param node "+n4+" with outdegree "+outEdges3.size()+" access path "+ap);
                            for (Iterator it2 = outEdges3.iterator(); it2.hasNext(); ) {
                                Node n5 = (Node) it2.next();
                                if (n5 instanceof OutsideNode)
                                    while (((OutsideNode)n5).skip != null)
                                        n5 = ((OutsideNode)n5).skip;
                                Object reason2 = pa.edgesToReasons.get(new Pair(n4, n5));
                                if (reason2 instanceof ProgramLocation) {
                                    ProgramLocation mc2 = (ProgramLocation) reason2;
                                    AndersenMethod targetMethod = ((ParamNode) n4).m;
                                    if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                        ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                                        MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                        CallSite cs = new CallSite(ms, mc2);
                                        if (TRACE || PRINT_INLINE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                        InlineSet targetMethods = (InlineSet) toInline.get(cs);
                                        if (targetMethods == null) {
                                            targetMethods = new InlineSet(new LinkedHashSet(), false);
                                            toInline.put(cs, targetMethods);
                                        }
                                        targetMethods.backing_set.add(targetMethod);
                                    } else {
                                        System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                                    }
                                } else {
                                    if (TRACE) out.println("unknown reason: "+reason);
                                }
                            }
                            break;
                        } else {
                            //if (TRACE) out.println("unknown reason: "+reason);
                            break;
                        }
                    }
                    TRACE = b;
                    Assert._assert(exact_types == null);
                }
            }
            for (Iterator it=outEdges.iterator(); it.hasNext(); ) {
                Node n2 = (Node) it.next();
                if (n2 instanceof OutsideNode)
                    while (((OutsideNode)n2).skip != null)
                        n2 = ((OutsideNode)n2).skip;
                if (exact_types == null) {
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Object reason = pa.edgesToReasons.get(new Pair(n, n2));
                if (!(reason instanceof ProgramLocation)) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : unknown reason "+reason);
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (TRACE) out.println("Edge "+n+"=>"+n2+" : from "+reason);
                ProgramLocation mc2 = (ProgramLocation) reason;
                Set concreteNodes2 = pa.getConcreteNodes(n2);
                if (concreteNodes.size() == concreteNodes2.size()) {
                    if (TRACE) out.println("Same number of concrete nodes, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                LinkedHashSet exact_types2 = new LinkedHashSet();
                boolean has_nonexact_types2 = false;
                for (Iterator i2=concreteNodes2.iterator(); i2.hasNext(); ) {
                    Node n3 = (Node)i2.next();
                    if (n3 instanceof ConcreteTypeNode) {
                        if (n3.getDeclaredType() != null)
                            exact_types2.add(n3.getDeclaredType());
                    } else {
                        has_nonexact_types2 = true;
                    }
                }
                if (TRACE) out.println("exact types of successor concrete nodes: "+exact_types2);
                if (TRACE) out.println("successor has nonexact types="+has_nonexact_types2);
                int num_exact_types2 = exact_types2.size();
                if (has_nonexact_types == has_nonexact_types2 &&
                    num_exact_types == num_exact_types2) {
                    if (TRACE) out.println("Same number of types, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (!has_nonexact_types && num_exact_types2 == 0) {
                    if (TRACE) out.println("Not enough types, skipping this edge.");
                    continue;
                }
                if (has_nonexact_types && !has_nonexact_types2) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : loss of nonexact targets");
                    AndersenMethod targetMethod = ((ParamNode) n).m;
                    if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                        ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                        MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                        CallSite cs = new CallSite(ms, mc2);
                        if (TRACE || PRINT_INLINE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                        InlineSet targetMethods = (InlineSet) toInline.get(cs);
                        if (targetMethods == null) {
                            targetMethods = new InlineSet(new LinkedHashSet(), false);
                            toInline.put(cs, targetMethods);
                        }
                        targetMethods.backing_set.add(targetMethod);
                    } else {
                        if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                    }
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Collection/*<ProgramLocation>*/ new_mc = mc;
                if (exact_targets_map == null) {
                    new_mc = new LinkedList();
                    exact_targets_map = new HashMap();
                    for (Iterator ii=mc.iterator(); ii.hasNext(); ) {
                        ProgramLocation methodcall = (ProgramLocation) ii.next();
                        Set exact_target_set = methodcall.getCallTargets(exact_types, true);
                        if (TRACE) out.println(n+" method call: "+methodcall+" exact call targets: "+exact_target_set.size());
                        if (!has_nonexact_types && exact_target_set.size() <= 1) {
                            if (TRACE) out.println("call target set too small, removing "+methodcall);
                        } else {
                            new_mc.add(methodcall);
                            exact_targets_map.put(methodcall, exact_target_set);
                        }
                    }
                    mc = new_mc;
                }
                
                Collection/*<ProgramLocation>*/ new_mc2 = new LinkedList();
                for (Iterator ii=new_mc.iterator(); ii.hasNext(); ) {
                    ProgramLocation methodcall = (ProgramLocation) ii.next();
                    Set exact_target_set2 = methodcall.getCallTargets(exact_types2, true);
                    if (TRACE) out.println(n2+" method call: "+methodcall+" exact call targets: "+exact_target_set2.size());
                    if (!has_nonexact_types && exact_target_set2.size() < 1) {
                        if (TRACE) out.println("call target set too small, removing "+methodcall);
                    } else {
                        new_mc2.add(methodcall);
                        Set exact_target_set = (Set) exact_targets_map.get(methodcall);
                        Assert._assert(exact_target_set != null);
                        if (exact_target_set.size() > exact_target_set2.size()) {
                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_target_set2.size()+" targets (smaller than "+exact_target_set.size()+")");
                            AndersenMethod targetMethod = ((ParamNode) n).m;
                            if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                                MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                CallSite cs = new CallSite(ms, mc2);
                                if (TRACE || PRINT_INLINE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                InlineSet targetMethods = (InlineSet) toInline.get(cs);
                                if (targetMethods == null) {
                                    targetMethods = new InlineSet(new LinkedHashSet(), false);
                                    toInline.put(cs, targetMethods);
                                }
                                targetMethods.backing_set.add(targetMethod);
                            } else {
                                if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                            }
                            addNextToWorklist(n2, new_mc2, visited, node_worklist);
                        }
                    }
                }
            }
        }
    }
    
    public static void searchForCloningOpportunities2(Map/*<CallSite,Set<jq_Method>>*/ toInline,
                                                     Set/*<CallSite>*/ selectedCallSites) {
        out.println("Searching for cloning opportunities for "+selectedCallSites.size()+" call sites");
        LinkedList/*<Node>*/ node_worklist = new LinkedList();
        HashMap/*<Node,Collection<ProgramLocation>>*/ visited = new HashMap();
        for (Iterator i=selectedCallSites.iterator(); i.hasNext(); ) {
            CallSite cs = (CallSite)i.next();
            ProgramLocation mc = cs.m;
            MethodSummary ms = cs.caller;
            PassedParameter pp = new PassedParameter(mc, 0);
            LinkedHashSet set = new LinkedHashSet();
            ms.getNodesThatCall(pp, set);
            if (TRACE) System.out.println("Call site: "+cs);
            for (Iterator it2 = set.iterator(); it2.hasNext(); ) {
                Node node = (Node) it2.next();
                if (node instanceof OutsideNode) {
                    if (TRACE) System.out.println("Outside node: "+node);
                    Collection/*<ProgramLocation>*/ s = (Collection) visited.get(node);
                    if (s == null) {
                        visited.put(node, s = new LinkedList());
                        node_worklist.add(node);
                    }
                    s.add(mc);
                }
            }
        }
        System.out.println("Root set of nodes to check: "+node_worklist.size());
        
outer:
        while (!node_worklist.isEmpty()) {
            Node n = (Node) node_worklist.removeFirst();
            Collection/*<ProgramLocation>*/ mc = (Collection) visited.get(n);
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            
            Set concreteNodes = null;
            LinkedHashSet exact_types = null;
            int num_exact_types = -1;
            boolean has_nonexact_types = false;
            Map/*<ProgramLocation,Set<jq_Method>>*/ exact_targets_map = null;
            if (n instanceof ParamNode) {
                if (outEdges.size() > 1) {
                    if (TRACE) out.println("found parameter node "+n+" with outdegree "+outEdges.size());
                    concreteNodes = pa.getConcreteNodes(n);
                    exact_types = new LinkedHashSet();
                    has_nonexact_types = false;
                    for (Iterator i=concreteNodes.iterator(); i.hasNext(); ) {
                        Node n3 = (Node)i.next();
                        if (n3 instanceof ConcreteTypeNode) {
                            if (n3.getDeclaredType() != null)
                                exact_types.add(n3.getDeclaredType());
                        } else {
                            has_nonexact_types = true;
                        }
                    }
                    if (TRACE) out.println("exact types of concrete nodes: "+exact_types);
                    if (TRACE) out.println("has nonexact types="+has_nonexact_types);
                    num_exact_types = exact_types.size();
                    if (!has_nonexact_types && num_exact_types <= 1) {
                        if (TRACE) out.println("only one type, skipping");
                        continue;
                    } else if (has_nonexact_types && num_exact_types == 0) {
                        if (TRACE) out.println("only nonexact types, skipping");
                        continue;
                    }
                }
            }
            for (Iterator it=outEdges.iterator(); it.hasNext(); ) {
                Node n2 = (Node) it.next();
                if (n2 instanceof OutsideNode)
                    while (((OutsideNode)n2).skip != null)
                        n2 = ((OutsideNode)n2).skip;
                if (exact_types == null) {
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Object reason = pa.edgesToReasons.get(new Pair(n, n2));
                if (!(reason instanceof ProgramLocation)) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : unknown reason "+reason);
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (TRACE) out.println("Edge "+n+"=>"+n2+" : from "+reason);
                ProgramLocation mc2 = (ProgramLocation) reason;
                Set concreteNodes2 = pa.getConcreteNodes(n2);
                if (concreteNodes.size() == concreteNodes2.size()) {
                    if (TRACE) out.println("Same number of concrete nodes, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                LinkedHashSet exact_types2 = new LinkedHashSet();
                boolean has_nonexact_types2 = false;
                for (Iterator i2=concreteNodes2.iterator(); i2.hasNext(); ) {
                    Node n3 = (Node)i2.next();
                    if (n3 instanceof ConcreteTypeNode) {
                        if (n3.getDeclaredType() != null)
                            exact_types2.add(n3.getDeclaredType());
                    } else {
                        has_nonexact_types2 = true;
                    }
                }
                if (TRACE) out.println("exact types of successor concrete nodes: "+exact_types2);
                if (TRACE) out.println("successor has nonexact types="+has_nonexact_types2);
                int num_exact_types2 = exact_types2.size();
                if (has_nonexact_types == has_nonexact_types2 &&
                    num_exact_types == num_exact_types2) {
                    if (TRACE) out.println("Same number of types, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (!has_nonexact_types && num_exact_types2 == 0) {
                    if (TRACE) out.println("Not enough types, skipping this edge.");
                    continue;
                }
                if (has_nonexact_types && !has_nonexact_types2) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : loss of nonexact targets");
                    AndersenMethod targetMethod = ((ParamNode) n).m;
                    if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                        ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                        MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                        CallSite cs = new CallSite(ms, mc2);
                        if (TRACE || PRINT_INLINE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                        InlineSet targetMethods = (InlineSet) toInline.get(cs);
                        if (targetMethods == null) {
                            targetMethods = new InlineSet(new LinkedHashSet(), false);
                            toInline.put(cs, targetMethods);
                        }
                        targetMethods.backing_set.add(targetMethod);
                    } else {
                        if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                    }
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Collection/*<ProgramLocation>*/ new_mc = mc;
                if (exact_targets_map == null) {
                    new_mc = new LinkedList();
                    exact_targets_map = new HashMap();
                    for (Iterator ii=mc.iterator(); ii.hasNext(); ) {
                        ProgramLocation methodcall = (ProgramLocation) ii.next();
                        Set exact_target_set = methodcall.getCallTargets(exact_types, true);
                        if (TRACE) out.println(n+" method call: "+methodcall+" exact call targets: "+exact_target_set.size());
                        if (!has_nonexact_types && exact_target_set.size() <= 1) {
                            if (TRACE) out.println("call target set too small, removing "+methodcall);
                        } else {
                            new_mc.add(methodcall);
                            exact_targets_map.put(methodcall, exact_target_set);
                        }
                    }
                    mc = new_mc;
                }
                
                Collection/*<ProgramLocation>*/ new_mc2 = new LinkedList();
                for (Iterator ii=new_mc.iterator(); ii.hasNext(); ) {
                    ProgramLocation methodcall = (ProgramLocation) ii.next();
                    Set exact_target_set2 = methodcall.getCallTargets(exact_types2, true);
                    if (TRACE) out.println(n2+" method call: "+methodcall+" exact call targets: "+exact_target_set2.size());
                    if (!has_nonexact_types && exact_target_set2.size() < 1) {
                        if (TRACE) out.println("call target set too small, removing "+methodcall);
                    } else {
                        new_mc2.add(methodcall);
                        Set exact_target_set = (Set) exact_targets_map.get(methodcall);
                        Assert._assert(exact_target_set != null);
                        if (exact_target_set.size() > exact_target_set2.size()) {
                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_target_set2.size()+" targets (smaller than "+exact_target_set.size()+")");
                            AndersenMethod targetMethod = ((ParamNode) n).m;
                            if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                                MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                CallSite cs = new CallSite(ms, mc2);
                                if (TRACE || PRINT_INLINE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                InlineSet targetMethods = (InlineSet) toInline.get(cs);
                                if (targetMethods == null) {
                                    targetMethods = new InlineSet(new LinkedHashSet(), false);
                                    toInline.put(cs, targetMethods);
                                }
                                targetMethods.backing_set.add(targetMethod);
                            }
                            addNextToWorklist(n2, new_mc2, visited, node_worklist);
                        }
                    }
                }
            }
        }
    }

    public static class Specialization {
        ControlFlowGraph target;
        Set/*<SpecializationParameter>*/ set;
        Specialization(ControlFlowGraph t, SpecializationParameter s) {
            this.target = t; this.set = new LinearSet(); this.set.add(s);
        }
        Specialization(ControlFlowGraph t, Set s) {
            this.target = t; this.set = s;
        }
        public boolean equals(Object o) {
            return equals((Specialization) o);
        }
        public boolean equals(Specialization that) {
            if (this.target != that.target) {
                return false;
            }
            if (!this.set.equals(that.set)) {
                return false;
            }
            return true;
        }
        public int hashCode() { return target.hashCode() ^ set.hashCode(); }
        
        public String toString() {
            return "Specialization of "+target.getMethod()+" on "+set;
        }
    }
    
    public static class AccessPath {
        AndersenField f;
        Node node;
        AccessPath n;
        
        public int length() {
            if (n == null) return 1;
            return 1+n.length();
        }

        public AndersenField first() { return f; }

        public AccessPath next() { return n; }
        
        public String toString() {
            String s;
            if (f == null) s = "[]";
            else s = "."+f.getName().toString();
            if (n == null) return s;
            return s+n.toString();
        }
        
        public boolean equals(Object o) {
            return equals((AccessPath)o);
        }

        public boolean equals(AccessPath that) {
            if (that == null) return false;
            if (this.f != that.f) return false;
            if (this.n == that.n) return true;
            if (this.n == null || that.n == null) return false;
            return this.n.equals(that.n);
        }

        public int hashCode() {
            int hashcode = f==null?0x1337:f.hashCode();
            if (n != null) hashcode ^= (n.hashCode() << 1);
            return hashcode;
        }

        public AccessPath findNode(Node node) {
            if (this.node == node) return this;
            else if (this.n == null) return null;
            else return this.n.findNode(node);
        }
        public static AccessPath create(AndersenField f, Node node, AccessPath n) {
            AccessPath ap;
            if (n != null) {
                ap = n.findNode(node);
                if (ap != null) return null;
                if (n.length() >= 3) return null;
            }
            ap = new AccessPath();
            ap.f = f; ap.node = node; ap.n = n;
            return ap;
        }
    }
    
    public static class SpecializationParameter {
        int paramNum;
        AccessPath ap;
        Set types;
        SpecializationParameter(int paramNum, AccessPath ap, Set types) {
            this.paramNum = paramNum; this.ap = ap; this.types = types;
        }
        public boolean equals(Object o) {
            return equals((SpecializationParameter) o);
        }
        public boolean equals(SpecializationParameter that) {
            if (this.paramNum != that.paramNum || !this.types.equals(that.types)) return false;
            if (this.ap == that.ap) return true;
            if (this.ap == null || that.ap == null) return false;
            return this.ap.equals(that.ap);
        }
        public int hashCode() {
            int aphash = ap==null?0:ap.hashCode();
            return paramNum ^ types.hashCode() ^ aphash;
        }
        public String toString() {
            if (ap == null)
                return "Param#"+paramNum+" types: "+types;
            return "Param#"+paramNum+ap.toString()+" types: "+types;
        }
    }

    public static HashMap/*<Specialization,Set<ProgramLocation>>*/ to_clone = new HashMap();
    
    public static HashMap/*<Pair<ProgramLocation,ControlFlowGraph>,Specialization>*/ callSitesToClones = new HashMap();

    public static void searchForCloningOpportunities4(Set/*<CallSite>*/ selectedCallSites) {
        out.println("Searching for cloning opportunities for "+selectedCallSites.size()+" call sites");
        LinkedList/*<Pair<Node,AccessPath>>*/ worklist = new LinkedList();
        HashMap/*<Pair<Node,AccessPath>,Collection<ProgramLocation>>*/ multimap = new HashMap();
        
        //TRACE =true;

        for (Iterator i=selectedCallSites.iterator(); i.hasNext(); ) {
            CallSite cs = (CallSite)i.next();
            ProgramLocation call_site = cs.m;
            MethodSummary caller_summary = cs.caller;
            PassedParameter param0 = new PassedParameter(call_site, 0);
            LinkedHashSet nodes = new LinkedHashSet();
            caller_summary.getNodesThatCall(param0, nodes);
            for (Iterator j=nodes.iterator(); j.hasNext(); ) {
                Node node = (Node) j.next();
                if (node instanceof OutsideNode) {
                    while (((OutsideNode)node).skip != null)
                        node = ((OutsideNode)node).skip;
                    List pair = new Pair(node, null);
                    if (!multimap.containsKey(pair)) {
                        worklist.add(pair);
                    }
                    MethodSummary.addToMultiMap(multimap, pair, call_site);
                }
            }
        }
        
        HashMap/*<Pair<Pair<Node,AccessPath>,ProgramLocation>,Set<jq_Method>>*/ chaCache = new HashMap();
        
outerloop:
        while (!worklist.isEmpty()) {
            List pair = (List) worklist.removeFirst();
            Set s = (Set) multimap.get(pair);
            if (s == null || s.isEmpty())
                continue;
            
            if (TRACE) out.println("Worklist ("+worklist.size()+") :"+pair);
            
            Node n = (Node) pair.get(0);
            AccessPath ap = (AccessPath) pair.get(1);
            
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            
            Set n_concretenodes = null;
            TypeSet n_typeset = null;
            for (Iterator i=outEdges.iterator(); i.hasNext(); ) {
                Node n2 = (Node) i.next();
                if (n2 instanceof OutsideNode)
                    while (((OutsideNode)n2).skip != null)
                        n2 = ((OutsideNode)n2).skip;
                Set s2 = s;
                Object reason = pa.edgesToReasons.get(new Pair(n, n2));
                if (TRACE) out.println("Edge: "+n2+" Reason: "+reason);
                if (false) {
                    if (outEdges.size() >= 2 &&
                        n instanceof FieldNode &&
                        reason instanceof Node) {
                        AndersenField f = ((FieldNode)n).f;
                        AccessPath ap2 = AccessPath.create(f, n, ap);
                        if (ap2 != null) {
                            Object key = new Pair(reason, ap2);
                            boolean change = multimap.containsKey(key);
                            MethodSummary.addToMultiMap(multimap, key, s);
                            if (change) {
                                if (TRACE) out.println("Adding to Worklist :"+key+","+s);
                                worklist.add(key);
                            }
                        }
                    }
                }
                if (outEdges.size() >= 2 && 
                    n instanceof ParamNode &&
                    reason instanceof ProgramLocation &&
                    ((ParamNode)n).m.getNameAndDesc().equals(((ProgramLocation)reason).getTargetMethod().getNameAndDesc())) {
                    if (n_concretenodes == null) {
                        n_concretenodes = pa.getConcreteNodes(n, ap);
                        if (n_concretenodes.isEmpty())
                            continue outerloop;
                        n_typeset = buildTypeSet(n_concretenodes);
                    }
                    Set n2_concretenodes = pa.getConcreteNodes(n2, ap);
                    if (n2_concretenodes.isEmpty())
                        continue;
                    if (n2_concretenodes.size() < n_concretenodes.size()) {
                        TypeSet n2_typeset = buildTypeSet(n2_concretenodes);
                        /*
                        if (n_typeset.isImprecise() && n2_typeset.isPrecise()) {
                            markForCloning((ParamNode)n, (ProgramLocation)reason);
                        } else {
                        */
                        s2 = new LinkedHashSet();
                        for (Iterator j=s.iterator(); j.hasNext(); ) {
                            ProgramLocation mc = (ProgramLocation) j.next();
                            Object key = new Pair(pair, mc);
                            Set n_targets_mc = (Set) chaCache.get(key);
                            if (n_targets_mc == null)
                                chaCache.put(key, n_targets_mc = mc.getCallTargets(n_typeset, true));
                            if (n_targets_mc.isEmpty()) {
                                j.remove();
                                continue;
                            }
                            Object key2 = new Pair(new Pair(n2, ap), mc);
                            Set n2_targets_mc = (Set) chaCache.get(key2);
                            if (n2_targets_mc == null)
                                chaCache.put(key, n2_targets_mc = mc.getCallTargets(n2_typeset, true));
                            if (n2_targets_mc.isEmpty()) {
                                continue;
                            }
                            s2.add(mc);
                            if (n2_targets_mc.size() < n_targets_mc.size()) {
                                markForCloning((ParamNode)n, (ProgramLocation)reason, n2_typeset, ap);
                            }
                        }
                    }
                }
                Object key = new Pair(n2, ap);
                boolean change = multimap.containsKey(key);
                MethodSummary.addToMultiMap(multimap, key, s2);
                if (change) {
                    if (TRACE) out.println("Adding to Worklist :"+key+","+s2);
                    worklist.add(key);
                }
            }
        }
    }

    public static void markForCloning(ParamNode n, ProgramLocation mc2, Set exact_types2, AccessPath ap) {
        AndersenMethod targetMethod = ((ParamNode) n).m;
        int paramNum = ((ParamNode) n).n;
        if (TRACE) out.println("Cloning call graph edge "+mc2+" to "+targetMethod);
        SpecializationParameter specialp = new SpecializationParameter(paramNum, ap, exact_types2);
        ControlFlowGraph target_cfg = CodeCache.getCode((jq_Method)targetMethod);
        Object pair = new Pair(mc2, target_cfg);
        Specialization special = (Specialization) callSitesToClones.get(pair);
        if (special != null) {
            TRACE = true;
            if (TRACE) System.out.println("Specialization already exists for call site! "+special);
            Assert._assert(special.target == target_cfg);
            Specialization special2 = new Specialization(target_cfg, specialp);
            boolean change = special2.set.addAll(special.set);
            if (change) {
                if (TRACE) System.out.println("Made new specialization: "+special2);
                Set set = (Set)to_clone.get(special);
                Assert._assert(set != null);
                if (TRACE) System.out.println("Removed call site: "+mc2);
                set.remove(mc2);
                if (set.isEmpty()) {
                    if (TRACE) System.out.println("Removed old unused specialization: "+special);
                    to_clone.remove(special);
                }
                special = special2;
            }
            TRACE = false;
        } else {
            special = new Specialization(target_cfg, specialp);
        }
        Set set = (Set)to_clone.get(special);
        if (set == null) {
            to_clone.put(special, set = new LinearSet());
            out.println("First clone for: "+special);
        }
        boolean change = set.add(mc2);
        if (change && PRINT_INLINE)
            out.println("Cloning call graph edge "+mc2+" to "+special);
        
    }

    public static TypeSet buildTypeSet(Set/*<ConcreteNode>*/ concreteNodes) {
        LinkedHashSet exact_types = new LinkedHashSet();
        boolean has_nonexact_types = false;
        for (Iterator i=concreteNodes.iterator(); i.hasNext(); ) {
            Node n3 = (Node)i.next();
            if (n3 instanceof ConcreteTypeNode) {
                if (n3.getDeclaredType() != null)
                    exact_types.add(n3.getDeclaredType());
            } else {
                has_nonexact_types = true;
            }
        }
        return new TypeSet(exact_types, has_nonexact_types);
    }

    public static class TypeSet extends AbstractSet {
        Set/*jq_Type*/ types;
        boolean is_complete;
        TypeSet(Set t, boolean b) {
            this.types = t; this.is_complete = b;
        }
        public Iterator iterator() { return types.iterator(); }
        public int size() { return types.size(); }
    }

    public static void searchForCloningOpportunities3(Set/*<CallSite>*/ selectedCallSites) {
        out.println("Searching for cloning opportunities for "+selectedCallSites.size()+" call sites");
        //to_clone.clear();
        //callSitesToClones.clear();
        LinkedList/*<Node>*/ node_worklist = new LinkedList();
        HashMap/*<Node,Collection<ProgramLocation>>*/ visited = new HashMap();
        for (Iterator i=selectedCallSites.iterator(); i.hasNext(); ) {
            CallSite cs = (CallSite)i.next();
            ProgramLocation mc = cs.m;
            MethodSummary ms = cs.caller;
            PassedParameter pp = new PassedParameter(mc, 0);
            LinkedHashSet set = new LinkedHashSet();
            ms.getNodesThatCall(pp, set);
            if (TRACE) System.out.println("Call site: "+cs);
            for (Iterator it2 = set.iterator(); it2.hasNext(); ) {
                Node node = (Node) it2.next();
                if (node instanceof OutsideNode) {
                    if (TRACE) System.out.println("Outside node: "+node);
                    Collection/*<ProgramLocation>*/ s = (Collection) visited.get(node);
                    if (s == null) {
                        visited.put(node, s = new LinkedList());
                        node_worklist.add(node);
                    }
                    s.add(mc);
                }
            }
        }
        System.out.println("Root set of nodes to check: "+node_worklist.size());
        
outer:
        while (!node_worklist.isEmpty()) {
            Node n = (Node) node_worklist.removeFirst();
            Collection/*<ProgramLocation>*/ mc = (Collection) visited.get(n);
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            
            Set concreteNodes = null;
            LinkedHashSet exact_types = null;
            int num_exact_types = -1;
            boolean has_nonexact_types = false;
            Map/*<ProgramLocation,Set<jq_Method>>*/ exact_targets_map = null;
            if (n instanceof ParamNode) {
                if (outEdges.size() > 1) {
                    if (TRACE) out.println("found parameter node "+n+" with outdegree "+outEdges.size());
                    concreteNodes = pa.getConcreteNodes(n);
                    exact_types = new LinkedHashSet();
                    has_nonexact_types = false;
                    for (Iterator i=concreteNodes.iterator(); i.hasNext(); ) {
                        Node n3 = (Node)i.next();
                        if (n3 instanceof ConcreteTypeNode) {
                            if (n3.getDeclaredType() != null)
                                exact_types.add(n3.getDeclaredType());
                        } else {
                            has_nonexact_types = true;
                        }
                    }
                    if (TRACE) out.println("exact types of concrete nodes: "+exact_types);
                    if (TRACE) out.println("has nonexact types="+has_nonexact_types);
                    num_exact_types = exact_types.size();
                    if (!has_nonexact_types && num_exact_types <= 1) {
                        if (TRACE) out.println("only one type, skipping");
                        continue;
                    } else if (has_nonexact_types && num_exact_types == 0) {
                        if (TRACE) out.println("only nonexact types, skipping");
                        continue;
                    }
                }
            }
            for (Iterator it=outEdges.iterator(); it.hasNext(); ) {
                Node n2 = (Node) it.next();
                if (n2 instanceof OutsideNode)
                    while (((OutsideNode)n2).skip != null)
                        n2 = ((OutsideNode)n2).skip;
                if (exact_types == null) {
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Object reason = pa.edgesToReasons.get(new Pair(n, n2));
                if (!(reason instanceof ProgramLocation)) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : unknown reason "+reason);
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (TRACE) out.println("Edge "+n+"=>"+n2+" : from "+reason);
                ProgramLocation mc2 = (ProgramLocation) reason;
                Set concreteNodes2 = pa.getConcreteNodes(n2);
                if (concreteNodes.size() == concreteNodes2.size()) {
                    if (TRACE) out.println("Same number of concrete nodes, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                LinkedHashSet exact_types2 = new LinkedHashSet();
                boolean has_nonexact_types2 = false;
                for (Iterator i2=concreteNodes2.iterator(); i2.hasNext(); ) {
                    Node n3 = (Node)i2.next();
                    if (n3 instanceof ConcreteTypeNode) {
                        if (n3.getDeclaredType() != null)
                            exact_types2.add(n3.getDeclaredType());
                    } else {
                        has_nonexact_types2 = true;
                    }
                }
                if (TRACE) out.println("exact types of successor concrete nodes: "+exact_types2);
                if (TRACE) out.println("successor has nonexact types="+has_nonexact_types2);
                int num_exact_types2 = exact_types2.size();
                if (has_nonexact_types == has_nonexact_types2 &&
                    num_exact_types == num_exact_types2) {
                    if (TRACE) out.println("Same number of types, skipping this edge.");
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                if (!has_nonexact_types && num_exact_types2 == 0) {
                    if (TRACE) out.println("Not enough types, skipping this edge.");
                    continue;
                }
                if (has_nonexact_types && !has_nonexact_types2) {
                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : loss of nonexact targets");
                    AndersenMethod targetMethod = ((ParamNode) n).m;
                    if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                        int paramNum = ((ParamNode) n).n;
                        if (TRACE) out.println("Cloning call graph edge "+mc2+" to "+targetMethod);
                        SpecializationParameter specialp = new SpecializationParameter(paramNum, null, exact_types2);
                        ControlFlowGraph target_cfg = CodeCache.getCode((jq_Method)targetMethod);
                        Object pair = new Pair(mc2, target_cfg);
                        Specialization special = (Specialization) callSitesToClones.get(pair);
                        if (special != null) {
                            TRACE = true;
                            if (TRACE) System.out.println("Specialization already exists for call site! "+special);
                            Assert._assert(special.target == target_cfg);
                            Specialization special2 = new Specialization(target_cfg, specialp);
                            boolean change = special2.set.addAll(special.set);
                            if (change) {
                                if (TRACE) System.out.println("Made new specialization: "+special2);
                                Set set = (Set)to_clone.get(special);
                                Assert._assert(set != null);
                                if (TRACE) System.out.println("Removed call site: "+mc2);
                                set.remove(mc2);
                                if (set.isEmpty()) {
                                    if (TRACE) System.out.println("Removed old unused specialization: "+special);
                                    to_clone.remove(special);
                                }
                                special = special2;
                            }
                            TRACE = false;
                        } else {
                            special = new Specialization(target_cfg, specialp);
                        }
                        Set set = (Set)to_clone.get(special);
                        if (set == null) to_clone.put(special, set = new LinearSet());
                        boolean change = set.add(mc2);
                        if (change && PRINT_INLINE)
                            out.println("Cloning call graph edge "+mc2+" to "+special);
                    } else {
                        if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                    }
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Collection/*<ProgramLocation>*/ new_mc = mc;
                if (exact_targets_map == null) {
                    new_mc = new LinkedList();
                    exact_targets_map = new HashMap();
                    for (Iterator ii=mc.iterator(); ii.hasNext(); ) {
                        ProgramLocation methodcall = (ProgramLocation) ii.next();
                        Set exact_target_set = methodcall.getCallTargets(exact_types, true);
                        if (TRACE) out.println(n+" method call: "+methodcall+" exact call targets: "+exact_target_set.size());
                        if (!has_nonexact_types && exact_target_set.size() <= 1) {
                            if (TRACE) out.println("call target set too small, removing "+methodcall);
                        } else {
                            new_mc.add(methodcall);
                            exact_targets_map.put(methodcall, exact_target_set);
                        }
                    }
                    mc = new_mc;
                }
                
                Collection/*<ProgramLocation>*/ new_mc2 = new LinkedList();
                for (Iterator ii=new_mc.iterator(); ii.hasNext(); ) {
                    ProgramLocation methodcall = (ProgramLocation) ii.next();
                    Set exact_target_set2 = methodcall.getCallTargets(exact_types2, true);
                    if (TRACE) out.println(n2+" method call: "+methodcall+" exact call targets: "+exact_target_set2.size());
                    if (!has_nonexact_types && exact_target_set2.size() < 1) {
                        if (TRACE) out.println("call target set too small, removing "+methodcall);
                    } else {
                        new_mc2.add(methodcall);
                        Set exact_target_set = (Set) exact_targets_map.get(methodcall);
                        Assert._assert(exact_target_set != null);
                        if (exact_target_set.size() > exact_target_set2.size()) {
                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_target_set2.size()+" targets (smaller than "+exact_target_set.size()+")");
                            AndersenMethod targetMethod = ((ParamNode) n).m;
                            if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                int paramNum = ((ParamNode) n).n;
                                if (TRACE) out.println("Cloning call graph edge "+mc2+" to "+targetMethod);
                                SpecializationParameter specialp = new SpecializationParameter(paramNum, null, exact_types2);
                                ControlFlowGraph target_cfg = CodeCache.getCode((jq_Method)targetMethod);
                                Object pair = new Pair(mc2, target_cfg);
                                Specialization special = (Specialization) callSitesToClones.get(pair);
                                if (special != null) {
                                    TRACE = true;
                                    if (TRACE) System.out.println("Specialization already exists! "+special);
                                    Assert._assert(special.target == target_cfg);
                                    Specialization special2 = new Specialization(target_cfg, specialp);
                                    boolean change = special2.set.addAll(special.set);
                                    if (change) {
                                        if (TRACE) System.out.println("Made new specialization: "+special2);
                                        Set set = (Set)to_clone.get(special);
                                        Assert._assert(set != null);
                                        if (TRACE) System.out.println("Removed call site: "+mc2);
                                        set.remove(mc2);
                                        if (set.isEmpty()) {
                                            if (TRACE) System.out.println("Removed old unused specialization: "+special);
                                            to_clone.remove(special);
                                        }
                                        special = special2;
                                    }
                                    TRACE = false;
                                } else {
                                    special = new Specialization(target_cfg, specialp);
                                }
                                Set set = (Set)to_clone.get(special);
                                if (set == null) to_clone.put(special, set = new LinearSet());
                                boolean change = set.add(mc2);
                                if (change && PRINT_INLINE)
                                    out.println("Cloning call graph edge "+mc2+" to "+special);
                            } else {
                                if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                            }
                            addNextToWorklist(n2, mc, visited, node_worklist);
                        }
                    }
                }
            }
        }
    }
    
    private static void addNextToWorklist(Node n2, Collection mc, HashMap visited, LinkedList node_worklist) {
        if (n2 instanceof OutsideNode)
            while (((OutsideNode) n2).skip != null)
                n2 = ((OutsideNode) n2).skip;
        else if (n2 instanceof ConcreteTypeNode)
            return;
        if (TRACE) out.println("Adding "+n2+" to worklist, mc set size="+mc.size());
        Collection c = (Collection) visited.get(n2);
        if (c != null) {
            if (TRACE) out.println("Already in worklist, existing mc set size="+c.size());
            LinkedHashSet lhs = new LinkedHashSet(c);
            lhs.addAll(mc);
            if (TRACE) out.println("New mc set size="+lhs.size());
            if (lhs.size() == c.size()) return;
            node_worklist.remove(n2);
            mc = lhs;
        }
        visited.put(n2, mc); node_worklist.add(n2);
    }
    
    // Follow the inclusion edges from a node, looking for a ParamNode with multiple
    // outgoing inclusion edges where some of the inclusion edges have fewer call
    // targets than the total.
    public static void searchForCloningOpportunities(Map toInline, OutsideNode on, ProgramLocation mc) {
        while (on.skip != null) on = on.skip;
        if (TRACE) out.println("searching for cloning opportunities for "+mc+", starting at "+on);
        LinkedList worklist = new LinkedList();
        HashSet visited = new HashSet();
        worklist.add(on); visited.add(on);
        while (!worklist.isEmpty()) {
            Node n = (Node) worklist.removeFirst();
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            boolean found = false, skip = false;
            if (n instanceof ParamNode) {
                if (outEdges.size() > 1) {
                    if (TRACE) out.println("found parameter node "+n+" with outdegree "+outEdges.size());
                    Set concreteNodes = pa.getConcreteNodes(n);
                    LinkedHashSet exact_types = new LinkedHashSet();
                    boolean has_nonexact_types = false;
                    for (Iterator i=concreteNodes.iterator(); i.hasNext(); ) {
                        Node n2 = (Node)i.next();
                        if (n2 instanceof ConcreteTypeNode) {
                            if (n2.getDeclaredType() != null)
                                exact_types.add(n2.getDeclaredType());
                        } else {
                            has_nonexact_types = true;
                        }
                    }
                    if (TRACE) out.println("exact types of concrete nodes: "+exact_types);
                    if (TRACE) out.println("has nonexact types="+has_nonexact_types);
                    int num_exact_types = exact_types.size();
                    if (!has_nonexact_types && num_exact_types <= 1) {
                        if (TRACE) out.println("only one type, skipping");
                        skip = true;
                    } else if (has_nonexact_types && num_exact_types == 0) {
                        if (TRACE) out.println("only nonexact types, skipping");
                        skip = true;
                    } else {
                        Set exact_targets = mc.getCallTargets(exact_types, true);
                        if (TRACE) out.println(n+" out edges: "+outEdges.size()+" exact call targets: "+exact_targets.size());
                        if (!has_nonexact_types && exact_targets.size() <= 1) {
                            if (TRACE) out.println("call target set too small, skipping");
                            skip = true;
                        } else {
                            for (Iterator i = outEdges.iterator(); i.hasNext(); ) {
                                Node n2 = (Node) i.next();
                                if (n2 instanceof OutsideNode)
                                    while (((OutsideNode)n2).skip != null)
                                        n2 = ((OutsideNode)n2).skip;
                                Object reason = pa.edgesToReasons.get(new Pair(n, n2));
                                if (reason instanceof ProgramLocation) {
                                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : from "+reason);
                                    ProgramLocation mc2 = (ProgramLocation) reason;
                                    Set concreteNodes2 = pa.getConcreteNodes(n2);
                                    if (concreteNodes.size() == concreteNodes2.size()) {
                                        if (TRACE) out.println("Same number of concrete nodes, skipping this edge.");
                                        continue;
                                    }
                                    LinkedHashSet exact_types2 = new LinkedHashSet();
                                    boolean has_nonexact_types2 = false;
                                    for (Iterator i2=concreteNodes2.iterator(); i2.hasNext(); ) {
                                        Node n3 = (Node)i2.next();
                                        if (n3 instanceof ConcreteTypeNode) {
                                            if (n3.getDeclaredType() != null)
                                                exact_types2.add(n3.getDeclaredType());
                                        } else {
                                            has_nonexact_types2 = true;
                                        }
                                    }
                                    if (TRACE) out.println("exact types of successor concrete nodes: "+exact_types2);
                                    if (TRACE) out.println("successor has nonexact types="+has_nonexact_types2);
                                    int num_exact_types2 = exact_types2.size();
                                    if (has_nonexact_types == has_nonexact_types2 &&
                                        num_exact_types == num_exact_types2) {
                                        if (TRACE) out.println("Same number of types, skipping this edge.");
                                        continue;
                                    }
                                    if (!has_nonexact_types && num_exact_types2 == 0) {
                                        if (TRACE) out.println("Not enough types, skipping this edge.");
                                        continue;
                                    }
                                    if (has_nonexact_types && !has_nonexact_types2) {
                                        if (TRACE) out.println("Edge "+n+"=>"+n2+" : loss of nonexact targets");
                                        AndersenMethod targetMethod = ((ParamNode) n).m;
                                        if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                            ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                                            MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                            CallSite cs = new CallSite(ms, mc2);
                                            if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                            Set targetMethods = (Set) toInline.get(cs);
                                            if (targetMethods == null) toInline.put(cs, targetMethods = new LinkedHashSet());
                                            targetMethods.add(targetMethod);
                                            found = true;
                                        } else {
                                            if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                                        }
                                    } else {
                                        Set exact_targets2 = mc.getCallTargets(exact_types2, true);
                                        if (exact_targets.size() > exact_targets2.size()) {
                                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_targets2.size()+" targets (smaller than "+exact_targets.size()+")");
                                            AndersenMethod targetMethod = ((ParamNode) n).m;
                                            if (targetMethod.getNameAndDesc().equals(mc2.getTargetMethod().getNameAndDesc())) {
                                                ControlFlowGraph caller_cfg = CodeCache.getCode((jq_Method)mc2.getMethod());
                                                MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                                CallSite cs = new CallSite(ms, mc2);
                                                if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                                Set targetMethods = (Set) toInline.get(cs);
                                                if (targetMethods == null) toInline.put(cs, targetMethods = new LinkedHashSet());
                                                targetMethods.add(targetMethod);
                                                found = true;
                                            } else {
                                                if (TRACE) System.out.println("Method on parameter node doesn't match edge reason! param node="+targetMethod+" reason="+reason);
                                            }
                                        }
                                    }
                                } else {
                                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : unknown reason "+reason);
                                }
                            }
                        }
                    }
                }
            }
            if (!skip && (FIND_ALL || !found)) {
                for (Iterator i=outEdges.iterator(); i.hasNext(); ) {
                    Node n2 = (Node) i.next();
                    if (n2 instanceof OutsideNode)
                        while (((OutsideNode)n2).skip != null)
                            n2 = ((OutsideNode)n2).skip;
                    if (visited.contains(n2)) continue;
                    visited.add(n2); worklist.add(n2);
                }
            }
        }
    }
    
}

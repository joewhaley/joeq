package Compil3r.Quad;

import java.io.PrintStream;
import java.util.*;

import Clazz.jq_Method;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.OutsideNode;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.PassedParameter;
import Compil3r.Quad.PointerExplorer.InlineSet;
import Main.jq;
import Util.Default;

/**
 * @author John Whaley
 */
public class SelectiveCloning {

    public static AndersenPointerAnalysis pa;
    
    public static boolean TRACE = false;
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
                if (exact_types == null) {
                    addNextToWorklist(n2, mc, visited, node_worklist);
                    continue;
                }
                Object reason = pa.edgesToReasons.get(Default.pair(n, n2));
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
                    ControlFlowGraph caller_cfg = CodeCache.getCode(mc2.getMethod());
                    MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                    CallSite cs = new CallSite(ms, mc2);
                    jq_Method targetMethod = ((ParamNode) n).m;
                    if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                    InlineSet targetMethods = (InlineSet) toInline.get(cs);
                    if (targetMethods == null) {
                        targetMethods = new InlineSet(new LinkedHashSet(), false);
                        toInline.put(cs, targetMethods);
                    }
                    targetMethods.backing_set.add(targetMethod);
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
                        jq.Assert(exact_target_set != null);
                        if (exact_target_set.size() > exact_target_set2.size()) {
                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_target_set2.size()+" targets (smaller than "+exact_target_set.size()+")");
                            ControlFlowGraph caller_cfg = CodeCache.getCode(mc2.getMethod());
                            MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                            CallSite cs = new CallSite(ms, mc2);
                            jq_Method targetMethod = ((ParamNode) n).m;
                            if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                            InlineSet targetMethods = (InlineSet) toInline.get(cs);
                            if (targetMethods == null) {
                                targetMethods = new InlineSet(new LinkedHashSet(), false);
                                toInline.put(cs, targetMethods);
                            }
                            targetMethods.backing_set.add(targetMethod);
                            addNextToWorklist(n2, new_mc2, visited, node_worklist);
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
                                Object reason = pa.edgesToReasons.get(Default.pair(n, n2));
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
                                        ControlFlowGraph caller_cfg = CodeCache.getCode(mc2.getMethod());
                                        MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                        CallSite cs = new CallSite(ms, mc2);
                                        jq_Method targetMethod = ((ParamNode) n).m;
                                        if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                        Set targetMethods = (Set) toInline.get(cs);
                                        if (targetMethods == null) toInline.put(cs, targetMethods = new LinkedHashSet());
                                        targetMethods.add(targetMethod);
                                        found = true;
                                    } else {
                                        Set exact_targets2 = mc.getCallTargets(exact_types2, true);
                                        if (exact_targets.size() > exact_targets2.size()) {
                                            if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+exact_targets2.size()+" targets (smaller than "+exact_targets.size()+")");
                                            ControlFlowGraph caller_cfg = CodeCache.getCode(mc2.getMethod());
                                            MethodSummary ms = MethodSummary.getSummary(caller_cfg);
                                            CallSite cs = new CallSite(ms, mc2);
                                            jq_Method targetMethod = ((ParamNode) n).m;
                                            if (TRACE) out.println("Inlining call graph edge "+cs+" to "+targetMethod);
                                            Set targetMethods = (Set) toInline.get(cs);
                                            if (targetMethods == null) toInline.put(cs, targetMethods = new LinkedHashSet());
                                            targetMethods.add(targetMethod);
                                            found = true;
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
                    if (visited.contains(n2)) continue;
                    if (n2 instanceof OutsideNode)
                        while (((OutsideNode) n2).skip != null)
                            n2 = ((OutsideNode) n2).skip;
                    visited.add(n2); worklist.add(n2);
                }
            }
        }
    }
    
}

package Compil3r.Quad;

import java.io.PrintStream;
import java.util.*;

import Clazz.jq_Method;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.OutsideNode;
import Compil3r.Quad.MethodSummary.ParamNode;
import Util.Default;

/**
 * @author John Whaley
 */
public class SelectiveCloning {

    public static AndersenPointerAnalysis pa;
    
    public static boolean TRACE = false;
    public static PrintStream out = System.out;
    
    public static boolean FIND_ALL = true;
    
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

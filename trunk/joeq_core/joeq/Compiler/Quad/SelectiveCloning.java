package Compil3r.Quad;

import java.io.PrintStream;
import java.util.*;

import Clazz.jq_Method;
import Compil3r.Quad.MethodSummary.CallSite;
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
        LinkedList worklist = new LinkedList();
        HashSet visited = new HashSet();
        worklist.add(on); visited.add(on);
        while (!worklist.isEmpty()) {
            Node n = (Node) worklist.removeFirst();
            Set outEdges = (Set) pa.nodeToInclusionEdges.get(n);
            if (outEdges == null) continue;
            boolean found = false;
            if (n instanceof ParamNode) {
                if (outEdges.size() > 1) {
                    Set concreteNodes = pa.getConcreteNodes(n);
                    Set targets = mc.getCallTargets(concreteNodes);
                    if (TRACE) out.println(n+" out edges: "+outEdges.size()+" call targets: "+targets.size());
                    if (targets.size() > 1) {
                        for (Iterator i = outEdges.iterator(); i.hasNext(); ) {
                            Node n2 = (Node) i.next();
                            Object reason = pa.edgesToReasons.get(Default.pair(n, n2));
                            if (reason instanceof ProgramLocation) {
                                if (TRACE) out.println("Edge "+n+"=>"+n2+" : from "+reason);
                                ProgramLocation mc2 = (ProgramLocation) reason;
                                Set concreteNodes2 = pa.getConcreteNodes(n2);
                                if (concreteNodes.size() == concreteNodes2.size()) continue;
                                Set targets2 = mc.getCallTargets(concreteNodes2);
                                if (targets.size() > targets2.size()) {
                                    if (TRACE) out.println("Edge "+n+"=>"+n2+" : "+targets2.size()+" targets (smaller than "+targets.size()+")");
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
                            } else {
                                if (TRACE) out.println("Edge "+n+"=>"+n2+" : unknown reason "+reason);
                            }
                        }
                    }
                }
            }
            if (FIND_ALL || !found) {
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

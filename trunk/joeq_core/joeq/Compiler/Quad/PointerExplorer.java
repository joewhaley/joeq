/*
 * PointerExplorer.java
 *
 * Created on August 18, 2002, 4:18 PM
 */

package Compil3r.Quad;

import Main.jq;
import Util.FilterIterator;
import Util.LinkedHashSet;
import Util.LinkedHashMap;
import Util.Default;
import Clazz.*;
import java.io.*;
import java.util.*;
import Compil3r.Quad.*;
import Compil3r.Quad.MethodSummary.MethodCall;
import Compil3r.Quad.MethodSummary.CallSite;
import Compil3r.Quad.MethodSummary.PassedParameter;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class PointerExplorer {

    public static final BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    
    public static jq_Method getMethod(Set/*<jq_Method>*/ set) throws IOException {
        int which, count = 0;
        for (Iterator i=set.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method)i.next();
            System.out.println((++count)+": "+m);
        }
        for (;;) {
            System.out.print("Which method? ");
            String s = in.readLine();
            try {
                which = Integer.parseInt(s);
                if ((which >= 1) && (which <= count))
                    break;
                System.out.println("Out of range: "+which);
            } catch (NumberFormatException x) {
                System.out.println("Not a number: "+s);
            }
        }
        for (Iterator i=set.iterator(); ; ) {
            jq_Method m = (jq_Method)i.next();
            if ((++count) == which) return m;
        }
    }
    
    public static jq_Method getMethod() throws IOException {
        return getMethod((String[])null);
    }
        
    public static jq_Method getMethod(String[] args) throws IOException {
        String mainClassName;
        if (args != null && args.length > 0) {
            mainClassName = args[0];
        } else {
            System.out.print("Enter the name of the class: ");
            mainClassName = in.readLine();
        }
        jq_Type t = jq.parseType(mainClassName);
        if (!(t instanceof jq_Class)) {
            System.out.println("Error, "+mainClassName+" ("+t+") is not a valid class.");
            System.exit(-1);
        }
        
        jq_Class klass = (jq_Class)t;
        klass.load(); klass.verify(); klass.prepare();
        String name = (args != null && args.length > 1) ? args[1] : null;
        return getMethod(klass, name);
    }
    
    public static jq_Method getMethod(jq_Class klass, String name) throws IOException {
        jq_Method method;
        if (name != null) {
            String methodName = name;
            boolean static_or_instance = false;
uphere1:
            for (;;) {
                jq_Method[] m = static_or_instance?(jq_Method[])klass.getDeclaredStaticMethods():(jq_Method[])klass.getDeclaredInstanceMethods();
                for (int i=0; i<m.length; ++i) {
                    if (methodName.equals(m[i].getName().toString())) {
                        method = m[i];
                        break uphere1;
                    }
                }
                if (static_or_instance) {
                    System.out.println("Error, no method named "+methodName+" is declared in class "+klass.getName());
                    System.exit(-1);
                }
                static_or_instance = true;
            }
        } else {
            boolean static_or_instance = true;
uphere2:
            for (;;) {
                System.out.println((static_or_instance?"Static":"Instance")+" methods:");
                jq_Method[] m = static_or_instance?(jq_Method[])klass.getDeclaredStaticMethods():(jq_Method[])klass.getDeclaredInstanceMethods();
                for (int i=0; i<m.length; ++i) {
                    System.out.println((i+1)+": "+m[i]);
                }
                int which;
                for (;;) {
                    System.out.print("Which method, or "+(static_or_instance?"'i' for instance":"'s' for static")+" methods: ");
                    String s = in.readLine();
                    try {
                        if (s.equalsIgnoreCase("s")) {
                            static_or_instance = true;
                            continue uphere2;
                        }
                        if (s.equalsIgnoreCase("i")) {
                            static_or_instance = false;
                            continue uphere2;
                        }
                        which = Integer.parseInt(s);
                        if ((which >= 1) && (which <= m.length))
                            break;
                        System.out.println("Out of range: "+which);
                    } catch (NumberFormatException x) {
                        System.out.println("Not a number: "+s);
                    }
                }
                method = m[which-1];
                break;
            }
        }
        return method;
    }
    
    public static SortedSet sortByNumberOfTargets(Map callGraph) {
        TreeSet ts = new TreeSet(
            new Comparator() {
                public int compare(Object o1, Object o2) {
                    Map.Entry e1 = (Map.Entry)o1;
                    CallSite cs1 = (CallSite)e1.getKey();
                    Set s1 = (Set)e1.getValue();
                    Map.Entry e2 = (Map.Entry)o2;
                    CallSite cs2 = (CallSite)e2.getKey();
                    Set s2 = (Set)e2.getValue();
                    int s1s = s1.size(); int s2s = s2.size();
                    if (s1s < s2s) return 1;
                    else if (s1s > s2s) return -1;
                    else return cs1.toString().compareTo(cs2.toString());
                }
            });
        ts.addAll(callGraph.entrySet());
        return ts;
    }
    
    public static AndersenPointerAnalysis apa;
    public static Map callGraph;
    public static Set rootSet = new LinkedHashSet();
    public static Set selectedCallSites = new LinkedHashSet();
    public static Map toInline = new LinkedHashMap();
    
    public static void selectCallSites(String desc, Iterator i, Iterator i2) throws IOException {
        System.out.println("Call sites with "+desc+": ");
        int count = 0;
        while (i2.hasNext()) {
            Map.Entry e = (Map.Entry)i2.next();
            Set s = (Set)e.getValue();
            System.out.println((++count)+": "+e.getKey()+"="+s.size()+" targets");
        }
        int which;
        for (;;) {
            System.out.print("Enter your selection, or 'a' for all: ");
            String input = in.readLine();
            if (input.equalsIgnoreCase("a")) {
                which = -1;
                break;
            } else { 
                try {
                    which = Integer.parseInt(input);
                    if ((which >= 1) && (which <= count))
                        break;
                } catch (NumberFormatException x) {
                    System.out.println("Cannot parse number: "+input);
                }
            }
        }
        for (int j=0; j<count; ++j) {
            Map.Entry e = (Map.Entry)i.next();
            if (which == j+1 || which == -1)
                selectedCallSites.add(e.getKey());
            if (which == j+1) {
                System.out.println("Selected "+e);
            }
        }
    }
    
    static void printAllInclusionEdges(HashSet visited, MethodSummary.Node pnode, MethodSummary.Node node, String indent, boolean all, jq_Field f, boolean verbose) throws IOException {
        if (verbose) System.out.print(indent+"Node: "+node);
        if (pnode != null) {
            Quad q = (Quad)apa.edgesToQuads.get(Default.pair(pnode, node));
            if (q != null)
                if (verbose) System.out.print(" from instruction "+q);
        }
        if (visited.contains(node)) {
            if (verbose) System.out.println(" <duplicate>, skipping.");
            return;
        }
        visited.add(node);
        if (verbose) System.out.println();
        if (node instanceof MethodSummary.OutsideNode) {
            MethodSummary.OutsideNode onode = (MethodSummary.OutsideNode)node;
            while (onode.skip != null) {
                if (verbose) System.out.println(indent+onode+" equivalent to "+onode.skip);
                onode = onode.skip;
            }
            if (onode instanceof MethodSummary.FieldNode) {
                MethodSummary.FieldNode fnode = (MethodSummary.FieldNode)onode;
                jq_Field field = fnode.f;
                Set inEdges = fnode.getAccessPathPredecessors();
                System.out.println(indent+"Field "+field+" Parent nodes: "+inEdges);
                System.out.print(indent+"Type 'w' to find matching writes to parent nodes, 'u' to go up: ");
                String s = in.readLine();
                if (s.equalsIgnoreCase("u")) {
                    for (Iterator it3 = inEdges.iterator(); it3.hasNext(); ) {
                        MethodSummary.Node node4 = (MethodSummary.Node)it3.next();
                        printAllInclusionEdges(visited, null, node4, indent+"<", all, field, true);
                    }
                } else if (s.equalsIgnoreCase("w")) {
                    for (Iterator it3 = inEdges.iterator(); it3.hasNext(); ) {
                        MethodSummary.Node node4 = (MethodSummary.Node)it3.next();
                        printAllInclusionEdges(visited, null, node4, indent+"<", all, field, false);
                    }
                }
            }
            Set outEdges = (Set)apa.nodeToInclusionEdges.get(onode);
            if (outEdges != null) {
                boolean yes = all || !verbose;
                if (!yes) {
                    System.out.print(indent+outEdges.size()+" out edges, print them? ('y' for yes, 'a' for all) ");
                    String s = in.readLine();
                    if (s.equalsIgnoreCase("y")) yes = true;
                    else if (s.equalsIgnoreCase("a")) all = yes = true;
                }
                if (yes) {
                    for (Iterator it3 = outEdges.iterator(); it3.hasNext(); ) {
                        MethodSummary.Node node2 = (MethodSummary.Node)it3.next();
                        printAllInclusionEdges(visited, onode, node2, indent+" ", all, null, verbose);
                    }
                }
            }
        } else {
            Set s = node.getNonEscapingEdges(f);
            if (s.size() > 0) {
                boolean yes = all;
                System.out.println(indent+s.size()+" write edges match field "+((f==null)?"[]":f.getName().toString()));
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    MethodSummary.Node node2 = (MethodSummary.Node)i.next();
                    Quad quad = node.getSourceQuad(f, node2);
                    if (quad != null)
                        System.out.println(indent+"From instruction: "+quad);
                    printAllInclusionEdges(visited, null, node2, indent+">", all, null, verbose);
                }
            }
        }
    }
    
    public static void doInlining() {
        System.out.println("Inlining "+toInline.size()+" call sites.");
        for (Iterator it = toInline.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry e = (Map.Entry)it.next();
            CallSite cs = (CallSite)e.getKey();
            MethodSummary caller = MethodSummary.getSummary(CodeCache.getCode(cs.caller.method));
            MethodCall mc = cs.m;
            Set targets = (Set)e.getValue();
            Iterator it2 = targets.iterator();
            if (!it2.hasNext()) {
                System.out.println("No targets to inline for "+cs);
            } else {
                for (;;) {
                    jq_Method target_m = (jq_Method)it2.next();
                    if (target_m.getBytecode() == null) {
                        System.out.println("Cannot inline target "+target_m+": target has no bytecode");
                    } else {
                        MethodSummary callee = MethodSummary.getSummary(CodeCache.getCode(target_m));
                        if (caller == callee) {
                            System.out.println("Inlining of recursive call not supported yet: "+cs);
                        } else {
                            MethodSummary.instantiate(caller, mc, callee, !it2.hasNext());
                        }
                    }
                    if (!it2.hasNext()) break;
                }
            }
        }
    }
    
    public static void main(String[] args) throws IOException {
        jq.initializeForHostJVMExecution();
        
        System.out.println("Select the root method.");
        jq_Method m = getMethod(args);
        rootSet.add(m);
        ControlFlowGraph cfg = CodeCache.getCode(m);
        apa = new AndersenPointerAnalysis(false);
        apa.addToRootSet(cfg);
        System.out.println("Performing initial context-insensitive analysis...");
        long time = System.currentTimeMillis();
        apa.iterate();
        time = System.currentTimeMillis() - time;
        System.out.println("Time to complete: "+time);
        callGraph = apa.getCallGraph();
        SortedSet sorted = sortByNumberOfTargets(callGraph);
        
        for (;;) {
            System.out.print("Enter command: ");
            String s = in.readLine();
            if (s == null) {
                System.out.println("Exiting.");
                System.exit(0);
            }
            if (s.startsWith("histogram")) {
                System.out.println(AndersenPointerAnalysis.computeHistogram(callGraph));
                continue;
            }
            if (s.startsWith("addroot")) {
                m = getMethod();
                rootSet.add(m);
                //cfg = CodeCache.getCode(m);
                //apa.addToRootSet(cfg);
                continue;
            }
            if (s.startsWith("trace summary ")) {
                boolean b = s.substring(14).equals("on");
                MethodSummary.TRACE_INTRA = b;
                System.out.println("Trace summary: "+b);
                continue;
            }
            if (s.startsWith("trace inline ")) {
                boolean b = s.substring(13).equals("on");
                MethodSummary.TRACE_INTER = b;
                System.out.println("Trace inline: "+b);
                continue;
            }
            if (s.startsWith("trace andersen ")) {
                boolean b = s.substring(15).equals("on");
                AndersenPointerAnalysis.TRACE = b;
                System.out.println("Trace Andersen: "+b);
                continue;
            }
            if (s.startsWith("inline")) {
                System.out.println("Marking "+selectedCallSites.size()+" call sites for inlining.");
                int size=0;
                for (Iterator it = selectedCallSites.iterator(); it.hasNext(); ) {
                    CallSite cs = (CallSite)it.next();
                    Set set = (Set)callGraph.get(cs);
                    if (set == null) {
                        System.out.println("Error: call site "+cs+" not found in call graph.");
                    } else {
                        toInline.put(cs, set);
                        size += set.size();
                    }
                }
                System.out.println("Total number of target methods: "+size);
                continue;
            }
            if (s.startsWith("run")) {
                MethodSummary.clearSummaryCache();
                doInlining();
                selectedCallSites.clear();
                System.gc();
                apa = new AndersenPointerAnalysis(false);
                for (Iterator it = rootSet.iterator(); it.hasNext(); ) {
                    m = (jq_Method)it.next();
                    cfg = CodeCache.getCode(m);
                    apa.addToRootSet(cfg);
                }
                System.out.println("Re-running context-insensitive analysis...");
                time = System.currentTimeMillis();
                apa.iterate();
                time = System.currentTimeMillis() - time;
                System.out.println("Time to complete: "+time);
                callGraph = apa.getCallGraph();
                sorted = sortByNumberOfTargets(callGraph);
                continue;
            }
            if (s.startsWith("source")) {
                final jq_Method m2 = getMethod();
                FilterIterator.Filter f = new FilterIterator.Filter() {
                        public boolean isElement(Object o) {
                            Map.Entry e = (Map.Entry)o;
                            CallSite cs = (CallSite)e.getKey();
                            return (cs.caller.method == m2);
                        }
                };
                FilterIterator it1 = new FilterIterator(sorted.iterator(), f);
                FilterIterator it2 = new FilterIterator(sorted.iterator(), f);
                selectCallSites("caller="+m, it1, it2);
                continue;
            }
            if (s.startsWith("targets")) {
                m = getMethod();
                int total = 0;
                for (Iterator it = callGraph.entrySet().iterator(); it.hasNext(); ) {
                    Map.Entry e = (Map.Entry)it.next();
                    Set set = (Set)e.getValue();
                    if (set.contains(m)) {
                        selectedCallSites.add(e.getKey());
                        ++total;
                    }
                }
                System.out.println("Selected "+total+" call sites.");
                continue;
            }
            if (s.startsWith("basepointers")) {
                for (Iterator it = selectedCallSites.iterator(); it.hasNext(); ) {
                    CallSite cs = (CallSite)it.next();
                    System.out.println("For call site: "+cs);
                    MethodSummary ms = cs.caller;
                    LinkedHashSet set = new LinkedHashSet();
                    PassedParameter pp = new PassedParameter(cs.m, 0);
                    ms.getNodesThatCall(pp, set);
                    for (Iterator it2 = set.iterator(); it2.hasNext(); ) {
                        MethodSummary.Node node = (MethodSummary.Node)it2.next();
                        printAllInclusionEdges(new HashSet(), null, node, "", false, null, true);
                    }
                }
                continue;
            }
            if (s.startsWith("clearselection")) {
                selectedCallSites.clear();
                continue;
            }
            if (s.startsWith("printselection")) {
                System.out.println(selectedCallSites);
                continue;
            }
            if (s.startsWith("summary")) {
                m = getMethod();
                MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
                System.out.println(ms);
                continue;
            }
            if (s.startsWith("printsize ")) {
                try {
                    final int size = Integer.parseInt(s.substring(10));
                    FilterIterator.Filter f = new FilterIterator.Filter() {
                            public boolean isElement(Object o) {
                                Map.Entry e = (Map.Entry)o;
                                Set set = (Set)e.getValue();
                                return set.size() >= size;
                            }
                    };
                    FilterIterator it1 = new FilterIterator(sorted.iterator(), f);
                    while (it1.hasNext()) {
                        System.out.println(it1.next());
                    }
                } catch (NumberFormatException x) {
                    System.out.println("Invalid size: "+s.substring(5));
                }
                continue;
            }
            if (s.startsWith("size ")) {
                try {
                    final int size = Integer.parseInt(s.substring(5));
                    FilterIterator.Filter f = new FilterIterator.Filter() {
                            public boolean isElement(Object o) {
                                Map.Entry e = (Map.Entry)o;
                                Set set = (Set)e.getValue();
                                return set.size() == size;
                            }
                    };
                    FilterIterator it1 = new FilterIterator(sorted.iterator(), f);
                    FilterIterator it2 = new FilterIterator(sorted.iterator(), f);
                    selectCallSites("size="+size, it1, it2);
                } catch (NumberFormatException x) {
                    System.out.println("Invalid size: "+s.substring(5));
                }
                continue;
            }
            if (s.startsWith("exit") || s.startsWith("quit")) {
                System.exit(0);
            }
            System.out.println("Unknown command: "+s);
        }
        
    }
    
}

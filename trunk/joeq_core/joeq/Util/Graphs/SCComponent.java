package Util.Graphs;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import Clazz.jq_Method;
import Compil3r.Quad.CallGraph;
import Util.Assert;

/**
 * <code>SCComponent</code> models a <i>Strongly connected component</i> \
 of a graph.
 * The only way to split a graph into <code>SCComponent</code>s is though
 * <code>buildSSC</code>.
 * That method is quite flexible: all it needs is a root node (or a set of
 * root nodes) and a <i>Navigator</i>: an object implementing the 
 * <code>Navigator</code> interface that provides the 
 * edges coming into/going out of a given <code>Object</code>. So, it can
 * build strongly connected components even for graphs that are not built
 * up from <code>CFGraphable</code> nodes, a good example being the set of
 * methods where the edges represent the caller-callee relation (in this
 * case, the strongly connected components group together sets of mutually
 * recursive methods).
 * 
 * @author  Alexandru SALCIANU <salcianu@retezat.lcs.mit.edu>
 * @version $Id$
 */
public final class SCComponent implements Comparable, Serializable {

    // THE FIRST PART CONTAINS JUST SOME STATIC METHODS & FIELDS

    // The internal version of a SCC: basically the same as the external
    // one, but using Sets instead of arrays; the idea is to use the most
    // convenient format during the generation and next to convert everything
    // to the small one to save some memory ...
    private static class SCComponentInt{
	// the nodes of this SCC
	public Vector nodes = new Vector();
	// the successors; kept as both a set (for quick membership testing)
	// and a vector for uniquely ordered and fast iterations. 
	public Set next = new HashSet();
	public Vector next_vec = new Vector();
	// the predecessors; similar to next, next_vec
	public Set prev = new HashSet();
	public Vector prev_vec = new Vector();
	// the "economic format" component
	public SCComponent comp = new SCComponent();
	// is there any edge to itself?
	public boolean loop = false;
    }
    
    // the set of nodes that are reachable from the root object
    private static Set reachable_nodes;
    // the set of reached nodes (to avoid reanalyzing them "ad infinitum")
    private static Set analyzed_nodes;  
    // Mapping node (Object) -> Strongly Connected Component (SCComponentInt)
    private static Hashtable node2scc;
    // The vector of the reached nodes, in the order DFS finished them
    private static Vector nodes_vector;
    private static SCComponentInt current_scc_int;
    // the navigator used in the DFS algorithm
    private static Navigator nav;
    // vector to put the generated SCCs in.
    private static Vector scc_vector;


    /** Convenient version for the single root case (see the other 
	<code>buildSCC</code> for details). Returns the single element of
	the set of top level SCCs. */
    public static final SCComponent buildSCC(final Object root,
					     final Navigator navigator) {
	Set set = buildSCC(Collections.singleton(root), navigator);
	if((set == null) || set.isEmpty()) return null;
        Assert._assert(set.size() <= 1, "More than one root SCComponent in a call with a a single root");
	// return the single element of the set of root SCComponents.
	return (SCComponent)(set.iterator().next());
    }


    /** Constructs the strongly connected components of the graph containing
	all the nodes reachable on paths that originate in nodes from
	<code>roots</code>. The edges are indicated by <code>navigator</code>.
	Returns the set of the root <code>SCComponent</code>s, the components
	that are not pointed by any other component. This constraint is
	actively used in the topological sorting agorithm (see
	<code>SCCTopSortedGraph</code>). */
    public static final Set buildSCC(final Collection roots,
				     final Navigator navigator) {
	scc_vector = new Vector();
	// STEP 1: compute the finished time of each node in a DFS exploration.
	// At the end of this step, nodes_vector will contain all the reached
	// nodes, in the order of their "finished" time. 
	nav = navigator;
	analyzed_nodes = new HashSet();
	nodes_vector   = new Vector();
        for(Iterator i=roots.iterator(); i.hasNext(); ) {
            Object root = i.next();
	    // avoid reanalyzing nodes
	    if(!analyzed_nodes.contains(root))
		DFS_first(root);
	}

	// STEP 2. build the SCCs by doing a DFS in the reverse graph.
	node2scc = new Hashtable();
	// "in reverse" navigator
	nav = new Navigator() {
		public Collection next(Object node) {
		    return navigator.prev(node);
		}
		public Collection prev(Object node) {
		    return navigator.next(node);
		}
	    };

	// Explore the nodes in the decreasing order of their finishing time.
	// This phase will create the SCCs (big format) and initialize the
	// node2scc mapping (but it won't set the inter-SCC edges). Also,
	// the SCCs are put in scc_vector.

	// only the nodes reachable from root nodes count! :
	// we make sure that navigator.prev cannot take us to strange places!
	reachable_nodes = analyzed_nodes; 
	analyzed_nodes = new HashSet();
	for(int i = nodes_vector.size() - 1; i >= 0; i--){
	    Object node = nodes_vector.elementAt(i);
	    // explore nodes that are still unanalyzed
	    if(node2scc.get(node) == null) {
		current_scc_int = new SCComponentInt();
		scc_vector.add(current_scc_int);
		DFS_second(node);
	    }
	}

	// Put the edges between the SCCs.
	put_the_edges(navigator);

	// Convert the big format SCCs into the compressed format SCCs.
	build_compressed_format();

	// Save the root SCComponents somewhere before activating the GC.
	Set root_sccs = new HashSet();
	Vector root_sccs_vec = new Vector();
        for(Iterator i=roots.iterator(); i.hasNext(); ) {
            Object root = i.next();
	    SCComponent root_scc = ((SCComponentInt) node2scc.get(root)).comp;
	    if(root_scc.prevLength() == 0) {
		if(root_sccs.add(root_scc)) 
		    root_sccs_vec.add(root_scc);
	    }
	}

	nav             = null; // enable the GC
	nodes_vector    = null;
	reachable_nodes = null;
	analyzed_nodes  = null;
	node2scc        = null;
	current_scc_int = null;
	scc_vector      = null;
	return root_sccs;
    }

    // DFS for the first step: the "forward" navigation
    private static final void DFS_first(Object node) {
	// do not analyze nodes already reached
	if(analyzed_nodes.contains(node)) return;

	analyzed_nodes.add(node);

	Collection next = nav.next(node);
        for (Iterator i=next.iterator(); i.hasNext(); )
	    DFS_first(i.next());

	nodes_vector.add(node);
    }

    // DFS for the second step: the "backward" navigation.
    private static final void DFS_second(Object node) {
	if(analyzed_nodes.contains(node) ||
	   !reachable_nodes.contains(node)) return;

	analyzed_nodes.add(node);
	node2scc.put(node, current_scc_int);
	current_scc_int.nodes.add(node);

        Collection next = nav.next(node);
        for (Iterator i=next.iterator(); i.hasNext(); )
            DFS_second(i.next());
    }
    
    // put the edges between the SCCs: there is an edge from scc1 to scc2 iff
    // there is at least one pair of nodes n1 in scc1 and n2 in scc2 such that
    // there exists an edge from n1 to n2.
    private static final void put_the_edges(final Navigator navigator){
	int nb_scc = scc_vector.size();
	for(int i = 0; i < scc_vector.size(); i++){
	    SCComponentInt compi = (SCComponentInt) scc_vector.elementAt(i);
	    for(Iterator it = compi.nodes.iterator(); it.hasNext(); ) {
		Object node = it.next();
                Collection edges = navigator.next(node);
                for (Iterator j=edges.iterator(); j.hasNext(); ) {
		    Object node2 = j.next();
		    SCComponentInt compi2 = 
			(SCComponentInt) node2scc.get(node2);
		    
		    if(compi2 == compi) compi.loop = true; 
		    else {
			if(compi.next.add(compi2.comp))
			    compi.next_vec.add(compi2.comp);
			if(compi2.prev.add(compi.comp))
			    compi2.prev_vec.add(compi.comp);
		    }
		}
	    }
	}
    }

    // Build the compressed format attached to each "fat" SCComponentInt.
    // This requires converting some sets to arrays (and sorting them in
    // the deterministic case). 
    private static final void build_compressed_format() {
	for(int i = 0; i < scc_vector.size(); i++){
	    SCComponentInt compInt = (SCComponentInt) scc_vector.elementAt(i);
	    SCComponent comp = compInt.comp;
	    comp.loop  = compInt.loop;
	    comp.nodes = new HashSet(compInt.nodes);
	    comp.nodes_array = 
		compInt.nodes.toArray(new Object[compInt.nodes.size()]);
	    comp.next  = (SCComponent[]) compInt.next_vec.toArray(
			      new SCComponent[compInt.next_vec.size()]);
	    comp.prev  = (SCComponent[]) compInt.prev_vec.toArray(
			      new SCComponent[compInt.prev_vec.size()]);
	}
    }


    // HERE STARTS THE REAL (i.e. NON STATIC) CLASS

    private static int count = 0;
    private int id;

    /** Returns the numeric ID of <code>this</code> <code>SCComponent</code>.
	Just for debug purposes ... */
    public int getId(){ return id; }

    // The nodes of this SCC (Strongly Connected Component).
    Set nodes;
    Object[] nodes_array;
    // The successors.
    private SCComponent[] next;
    // The predecessors.
    private SCComponent[] prev;

    // is there any arc to itself?
    private boolean loop;
    /** Checks whether <code>this</code> strongly connected component
	corresponds to a loop, <i>ie</i> it has at least one arc to
	itself. */
    public final boolean isLoop() { return loop; }

    //The only way to produce SCCs is through SCComponent.buildSSC !
    SCComponent() { id = count++; }

    public int compareTo(Object o) {
	SCComponent scc2 = (SCComponent) o;
	int id2 = scc2.id;
	if(id  < id2) return -1;
	if(id == id2) return 0;
	return 1;
    }
    
    /** Returns the number of successors. */
    public final int nextLength() { return next.length; }

    /** Returns the <code>i</code>th successor. */
    public final SCComponent next(int i) { return next[i]; }

    /** Returns the number of predecessors. */
    public final int prevLength() { return prev.length; }

    /** Returns the <code>i</code>th predecessor. */
    public final SCComponent prev(int i) { return prev[i]; }

    /** Returns the nodes of <code>this</code> strongly connected component
	(set version). */
    public final Set nodeSet() { return nodes; }

    /** Returns the nodes of <code>this</code> strongly connected component;
	array version - good for iterating over the elements of the SCC. */
    public final Object[] nodes() { return nodes_array; }

    /** Returns the number of nodes in <code>this</code> strongly connected
	component. */
    public final int size() { return nodes_array.length; }

    /** Checks whether <code>node</code> belongs to <code>this</code> \
	strongly connected component. */
    public final boolean contains(Object node){
	return nodes.contains(node);
    }

    // the next and prev links in the double linked list of SCCs in
    // decreasing topological order
    SCComponent nextTopSort = null;
    SCComponent prevTopSort = null;

    /** Returns the next <code>SCComponent</code> according to the decreasing
     * topological order */
    public final SCComponent nextTopSort() { return nextTopSort; }

    /** Returns the previous <code>SCComponent</code> according to the
     * decreasing topological order */
    public final SCComponent prevTopSort() { return prevTopSort; }

    public final List/*<SCComponent>*/ listTopSort() {
        int n = 1;
        SCComponent c = this;
        while ((c = c.nextTopSort()) != null) ++n;
        SCComponent[] a = new SCComponent[n];
        c = this;
        for (int i=0; i<n; ++i) {
            a[i] = c;
            c = c.nextTopSort();
        }
        Assert._assert(c == null);
        return Arrays.asList(a);
    }
    
    /** Pretty print debug function. */
    public final String toString() {
	StringBuffer buffer = new StringBuffer();
	buffer.append("SCC" + id + " (size " + size() + ") {\n");
	for(int i = 0; i < nodes_array.length; i++) {
	    buffer.append(nodes_array[i]);
	    buffer.append("\n");
	}
	buffer.append("}\n");
	buffer.append(prevStringRepr());
	buffer.append(nextStringRepr());
	return buffer.toString();
    }

    // Returns a string representation of the "prev" links.
    private String prevStringRepr() {
	StringBuffer buffer = new StringBuffer();
	int nb_prev = prevLength();
	if(nb_prev > 0){
	    buffer.append("Prev:");
	    for(int i = 0; i < nb_prev ; i++){
		buffer.append(" SCC" + prev(i).id);
	    }
	    buffer.append("\n");
	}
	return buffer.toString();
    }

    // Returns a string representation of the "next" links.
    private String nextStringRepr() {
	StringBuffer buffer = new StringBuffer();
	int nb_next = nextLength();
	if(nb_next > 0){
	    buffer.append("Next:");
	    for(int i = 0; i < nb_next ; i++){
		buffer.append(" SCC" + next(i).id);
	    }
	    buffer.append("\n");
	}
	return buffer.toString();
    }

    /** Pretty print debug function for SCC's of a call graph. */
    public final String toString(CallGraph mcg) {
	StringBuffer buffer = new StringBuffer();

	boolean extended = size() > 1;

	buffer.append("SCC" + id + " (size " + size() + ") {\n");

	for(int i = 0; i < nodes_array.length; i++) {
	    Object o = nodes_array[i];
	    buffer.append(o);
	    buffer.append("\n");
	    if(extended){
		Collection next = mcg.getCallees((jq_Method) o);
		for(Iterator j = next.iterator(); j.hasNext(); ) {
                    Object p = j.next();
		    if(nodes.contains(p))
		       buffer.append("  " + p + "\n");
		}
		buffer.append("\n");
	    }
	}
	buffer.append("}\n");
	buffer.append(prevStringRepr());
	buffer.append(nextStringRepr());
	return buffer.toString();
    }

}

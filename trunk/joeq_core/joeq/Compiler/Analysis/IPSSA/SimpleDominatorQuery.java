package Compil3r.Analysis.IPSSA;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import Util.Assert;

import Clazz.jq_Method;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.BasicBlockVisitor;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.Dominators;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;
import Compil3r.Quad.Dominators.DominatorNode;
import Util.SyntheticGraphs.Graph;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;

/**
 * @author Vladimir Livshits
 *  */
interface DominatorQuery {
	/** The result is null for the top node of the CFG. */
	public Quad getImmediateDominator(Quad q);
	/** Checks if the node is the top node of the CFG. */
	public boolean isTop(Quad q);
	/** Fills set with the dominance frontier of q */
	public void getDominanceFrontier(Quad q, Set/*<Quad>*/ set);
	/** Fills set with the iterated dominance frontier of q */
	public void getIteratedDominanceFrontier(Quad q, Set/*<Quad>*/ set);
	/** Prints the dominator tree on Quads in dot format. */	
	public void printDot(PrintStream out); 
};

/**
 * A pretty obvious implementation of DominatorQuery, nothing fancy here. 
 * Needs to be optimized for future use.
 * @see DominatorQuery
 * */
public class SimpleDominatorQuery implements DominatorQuery {
	private jq_Method _m;	
	private ControlFlowGraph _cfg;

	// Maps we create to answer the queries
	private HashMap _bb2nodeMap;	
	private HashMap _quad2BBMap;

	public SimpleDominatorQuery(jq_Method m){
		this._m = m;
		this._cfg = CodeCache.getCode(m);
		
		// build BB-level dominators
		Dominators dom = new Dominators(true);
		dom.visitMethod(m);
		DominatorNode root = dom.computeTree();
		
		// create lookup maps
		_bb2nodeMap = new HashMap();
		buildBB2NodeMap(root, _bb2nodeMap);
		
		_quad2BBMap = new HashMap();
		buildQuad2BBMap(_quad2BBMap);
		
		//System.out.println("_bb2nodeMap: " + _bb2nodeMap.size() + ", _quad2BBMap: " + _quad2BBMap.size() + "\n");
	}
		
	private void buildBB2NodeMap(DominatorNode root, HashMap map) {
		BasicBlock bb = root.getBasicBlock();
		Assert._assert(bb != null);
		map.put(bb, root);		
		
		List children = root. getChildren();
		//System.out.println(children.size() + " children \n");
		for(Iterator i = children.iterator(); i.hasNext();){
			DominatorNode child = (DominatorNode)i.next();
			
			buildBB2NodeMap(child, map);
		}
	}
	
	private void buildQuad2BBMap(final HashMap map) {
		_cfg.visitBasicBlocks(new BasicBlockVisitor(){
			public void visitBasicBlock(BasicBlock bb){
				//System.out.println("Block " + bb.toString() + "\n");
				Quad lastQuad = bb.getLastQuad();
				if(lastQuad == null) return;
				for(int i = 0; i <= bb.getQuadIndex(lastQuad); i++){
					Quad q = (Quad)bb.getQuad(i);
					
					map.put(q, bb);
				}	
			}
		});
	}

	public Quad getImmediateDominator(Quad q){
		BasicBlock bb = (BasicBlock)_quad2BBMap.get(q);
		Assert._assert(bb != null);
		
		int pos = bb.getQuadIndex(q);
		if(pos > 0){
			return bb.getQuad(pos - 1);
		}else{
			DominatorNode node = (DominatorNode)_bb2nodeMap.get(bb);
			Assert._assert(node != null);
			
			DominatorNode dom = node.getParent();
			if(dom != null){
				return dom.getBasicBlock().getLastQuad(); 	
			}else{
				return null;
			}
		}
	}
	
	public boolean isTop(Quad q){
		return getImmediateDominator(q) == null;
	}
	
	public void getDominanceFrontier(Quad q, Set/*<Quad>*/ set){
		/*
		 * The idea is to get a dominance frontier from the dominance tree.
		 * DF(n) = {m | n dom pred(m), but n ~dom m}, so, there's a dominated predecessor. 
		 * */
		Assert._assert(set != null);	 
		DominatorNode node = getNode(q);
		processChildren(node.getBasicBlock(), node, set);
	}
	
	private void processChildren(BasicBlock root, DominatorNode node, Set set) {
		BasicBlock bb = node.getBasicBlock();
		Util.Templates.ListIterator.BasicBlock predecessors = bb.getPredecessors().basicBlockIterator();
		while(predecessors.hasNext()){
			BasicBlock pred = predecessors.nextBasicBlock();
			
			if(!dominates(root, pred)){
				set.add(bb.getQuad(0));
				// AT LEAST one predecessor is not dominated -- no need to look at the others
				break;
			}
		}
		for(Iterator iter = node.getChildren().iterator(); iter.hasNext(); ){
			DominatorNode child = (DominatorNode)iter.next();
			
			// only the first node of a BB is suspect
			processChildren(root, child, set);									
		}	
	}

	private boolean dominates(BasicBlock root, BasicBlock pred) {
		DominatorNode root_node = (DominatorNode) _bb2nodeMap.get(root);
		DominatorNode node = (DominatorNode) _bb2nodeMap.get(pred);		
		
		while(node != null){
			node = node.getParent();
			if(node == root_node){
				return true;
			}
		}
		
		return false;
	}

	private DominatorNode getNode(Quad q) {
		BasicBlock bb = (BasicBlock)_quad2BBMap.get(q);
		Assert._assert(bb != null);
		DominatorNode node = (DominatorNode)_bb2nodeMap.get(bb);
		Assert._assert(node != null);

		return node;
	}

	public void getIteratedDominanceFrontier(Quad q, Set/*<Quad>*/ set){
		getDominanceFrontier(q, set);
		
		boolean change = false;
		do {
			change = false;
			for(Iterator iter = set.iterator(); iter.hasNext(); ){
				Quad domFrontierQuad = (Quad)iter.next();
				
				int oldSetSize = set.size();				
				getDominanceFrontier(domFrontierQuad, set);				
				Assert._assert(set.size() >= oldSetSize);
				
				if(set.size() != oldSetSize){
					// change detected
					change = true;
				}
			}
		} while (change);
	}
		
	/**
	 * Prints the dominator tree on Quads in dot format.
	 * */
	public void printDot(PrintStream out){
		Graph g = new Graph(_m.toString(), new Graph.Direction(Graph.Direction.LR));
		for(Iterator iter = new QuadIterator(_cfg); iter.hasNext();){
			Quad q = (Quad)iter.next();
			
			// these IDs should be unique, I hope
			ProgramLocation loc = new QuadProgramLocation(_m, q);
			String src_loc = loc.getSourceFile().toString() + ":" + loc.getLineNumber();
			g.addNode(q.getID(), q.toString_short() + "\\l" + src_loc);
			Quad dom = getImmediateDominator(q);
			if(dom != null){
				g.addNode(dom.getID(), dom.toString_short());
				g.addEdge(q.getID(), dom.getID());
			}
		}
		
		// graph creation is complete
		g.printDot(out);
	}
	
	public static class TestSimpleDominatorQuery implements ControlFlowGraphVisitor {
		public TestSimpleDominatorQuery(){
			CodeCache.AlwaysMap = true;
		}
		
		public void visitCFG(ControlFlowGraph cfg) {
			SimpleDominatorQuery q = new SimpleDominatorQuery(cfg.getMethod());
			q.printDot(System.out);	
		}
		
		public static void Main(String argv[]){
			for(int i = 0; i < argv.length; i++){
				String arg = argv[i];
				
				if(arg == "-v"){
					// TOOD
				}
			}
		}
	}
};


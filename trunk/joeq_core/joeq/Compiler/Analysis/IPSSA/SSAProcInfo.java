package Compil3r.Analysis.IPSSA;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import Util.Assert;
import Util.Templates.ListIterator;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.BasicBlockVisitor;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.DotGraph;
import Compil3r.Quad.ExceptionHandler;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;


public final class SSAProcInfo {
	protected static HashMap /*<Query,  SSABindingAnnote>*/ 	_queryMap  = new HashMap();
	protected static HashMap /*<Helper, SSABindingAnnote>*/ 	_helperMap = new HashMap();
	
	public static Query retrieveQuery(jq_Method method){
		if(_queryMap.containsKey(method)){
			return (Query)_queryMap.get(method);
		}else{
			Query q = new Query(method);
			_queryMap.put(method, q);
			
			return q;
		}
	}
	public static Helper retrieveHelper(jq_Method method){
		if(_queryMap.containsKey(method)){
			return (Helper)_helperMap.get(method);
		}else{
			Helper q = new Helper(method);
			_helperMap.put(method, q);
			
			return q;
		}
	}
	
	// TODO: this is pretty lame
	static Iterator emptyIterator(){
		return new HashSet().iterator();
	}
	
	/**
	 * This class is used to get information about the IPSSA representation.
	 * Use SSAProcInfo.retreiveQuery to get an appropriate query.
	 * */
	public static class Query {
		jq_Method 		  		     					_method;
		protected ControlFlowGraph 	 					_cfg;
		protected DominatorQuery 	 					_dom_query; 	
		protected HashMap /*<Quad, SSABindingAnnote>*/  _bindingMap;
				
		protected Query(jq_Method method){			
			this._method 	 = method;
			this._cfg    	 = CodeCache.getCode(method);
			this._bindingMap = new HashMap();
			this._dom_query  = new SimpleDominatorQuery(_method);		
		}
		
		public String toString(){
			return "Query for " + _method.toString();
		}
		
		public SSADefinition getDefinitionFor(SSALocation loc, Quad q){
			SSABindingAnnote ba = (SSABindingAnnote)_bindingMap.get(q);
			if(ba == null) return null;

			return ba.getDefinitionFor(loc);
		}
				
		public SSADefinition getLastDefinitionFor(SSALocation loc, Quad q, boolean strict){
			if(strict){
				q = _dom_query.getImmediateDominator(q);				
			}
			
			while(q != null){
				SSADefinition def = getDefinitionFor(loc, q);
				if(def != null){
					return def;
				}
			}
			
			return null;
		}
		
		public Iterator getBindingIterator(Quad q){
			if(_bindingMap.containsKey(q)){
				return ((SSABindingAnnote)_bindingMap.get(q)).getBindingIterator(); 
			}else{
				return emptyIterator();	
			}
		}
		
		public int getBindingCount(Quad quad) {
			if(!_bindingMap.containsKey(quad)){
				return 0;
			}else{				
				SSABindingAnnote ba = ((SSABindingAnnote)_bindingMap.get(quad));
				return ba.size();
			}
		}
	
		/**
		 * An iterator for all bindings in method.
		 * */	
		public Iterator getBindingIterator(jq_Method method){
			class BindingIterator implements Iterator {
				protected jq_Method _method;
				protected Iterator 	_bindingIter;
				protected Iterator 	_quadIter;
				protected Query 	_query;
				
				public BindingIterator(jq_Method method){
					this._method      = method; 
					this._quadIter 	  = new QuadIterator(CodeCache.getCode(_method));
					this._bindingIter = emptyIterator();
					this._query       = retrieveQuery(_method);										 
				}
				public boolean hasNext(){
					if(_bindingIter.hasNext()) return true;
					
					while(_quadIter.hasNext()){
						Quad quad = (Quad)_quadIter.next();
						if(_query.getBindingCount(quad) > 0){
							_bindingIter = _query.getBindingIterator(quad);
							
							return true;
						}
					}
					
					return false;
				}
				public Object next(){
					if(_bindingIter.hasNext()){
						return _bindingIter.next();
					}else{
						Quad quad = (Quad)_quadIter.next();						
						_bindingIter = _query.getBindingIterator(quad);
						
						return _bindingIter.next();						 
					}
				}				
				public void remove(){
					Assert._assert(false, "Don't call this method");
				}	
			}
			return new BindingIterator(method);
		}
		
		public void print(PrintStream out){
			for (QuadIterator j=new QuadIterator(_cfg, true); j.hasNext(); ) {
				Quad q = j.nextQuad();
			
				SSABindingAnnote ba = (SSABindingAnnote)_bindingMap.get(q);
				if(ba == null) continue;
				out.println(q.toString() + "\n" + ba.toString("\t"));					
			}
		}
		
		public void printDot(PrintStream out){
			new DotGraph(){
			public void visitCFG(ControlFlowGraph cfg) {
				try {
					String filename = "joeq-" + cfg.getMethod().toString() + ".ssa.dot";
					filename = filename.replace('/', '_');
					filename = filename.replace(' ', '_');
					filename = filename.replace('<', '_');
					filename = filename.replace('>', '_');
					dot.openGraph(filename);
					
					cfg.visitBasicBlocks(new BasicBlockVisitor() {
						public void visitBasicBlock(BasicBlock bb) {
							if (bb.isEntry()) {
								if (bb.getNumberOfSuccessors() != 1)
									throw new Error("entry bb has != 1 successors " + bb.getNumberOfSuccessors());
								dot.addEntryEdge(bb.toString(), bb.getSuccessors().iterator().next().toString(), null);
							} else
							if (!bb.isExit()) {
								ListIterator.Quad qit = bb.iterator();
								StringBuffer l = new StringBuffer(" " + bb.toString() + "\\l");
								HashSet allExceptions = new HashSet();
								while (qit.hasNext()) {
									// This is where the text of the bb is created
									l.append(" ");
									Quad quad = qit.nextQuad();
									//l.append(dot.escape(quad.toString()));
									
									SSAProcInfo.Query q = SSAProcInfo.retrieveQuery(_cfg.getMethod());
									for(Iterator iter = q.getBindingIterator(quad); iter.hasNext();){
										SSABinding b = (SSABinding)iter.next();
										l.append(b.toString() + "\\l");
									}
									
									l.append(dot.escape(quad.toString()));
									l.append("\\l");
									ListIterator.jq_Class exceptions = quad.getThrownExceptions().classIterator();
									while (exceptions.hasNext()) {
										allExceptions.add(exceptions.nextClass());
									}
								}
								dot.userDefined("\t" + bb.toString() + " [shape=box,label=\"" + l + "\"];\n");

								ListIterator.BasicBlock bit = bb.getSuccessors().basicBlockIterator();
								while (bit.hasNext()) {
									BasicBlock nextbb = bit.nextBasicBlock();
									if (nextbb.isExit()) {
										dot.addLeavingEdge(bb.toString(), nextbb.toString(), null);
									} else {
										dot.addEdge(bb.toString(), nextbb.toString());
									}
								}

								Iterator eit = allExceptions.iterator();
								while (eit.hasNext()) {
									jq_Class exc = (jq_Class)eit.next();
									ListIterator.ExceptionHandler mayCatch;
									mayCatch = bb.getExceptionHandlers().mayCatch(exc).exceptionHandlerIterator();
									while (mayCatch.hasNext()) {
										ExceptionHandler exceptionHandler = mayCatch.nextExceptionHandler();
										BasicBlock nextbb = exceptionHandler.getEntry();
										dot.addEdge(bb.toString(), nextbb.toString(), exceptionHandler.getExceptionType().toString());
									}
									// if (bb.getExceptionHandlers().mustCatch(exc) == null) { }
								}
							}
						}
					});
				} finally {
					dot.closeGraph();
				}
			}}.visitCFG(_cfg);		 
		}

		public DominatorQuery getDominatorQuery() {
			return _dom_query;			
		}
	}
		
	/**
	 * This class is used to make modifications to the IPSSA representation.
	 * */
	public static class Helper {
		jq_Method _method;
		Query     _query;
		
		protected Helper(jq_Method method){
			this._method = method;
			this._query  = SSAProcInfo.retrieveQuery(_method);
		}
		
		public static SSADefinition create_ssa_definition(SSALocation loc, Quad quad) {
			return SSADefinition.Helper.create_ssa_definition(loc, quad);
		}
	}
	
	static class SSABindingAnnote {
		protected LinkedList _bindings;
		
		SSABindingAnnote(){
			_bindings = new LinkedList();
		}
				
		public SSADefinition getDefinitionFor(SSALocation loc) {
			for(Iterator iter = _bindings.iterator(); iter.hasNext();){
				SSABinding b = (SSABinding)iter.next();
				SSADefinition def = b.getDestination();
				if(def.getLocation() == loc){
					return def;
				}
			}			
			return null;
		}

		public SSADefinition addBinding(SSALocation loc, SSAValue value, Quad quad) {
			SSABinding b = new SSABinding(quad, loc, value);
			Assert._assert(quad == value.getQuad());
			Assert._assert(quad == b.getDestination().getQuad());
			
			this._bindings.addLast(b);
			
			return b.getDestination(); 		
		}

		public void addBinding(Quad quad, SSALocation loc, SSAValue value){
			SSABinding b = new SSABinding(quad, loc, value);
			Assert._assert(quad == value.getQuad());
			Assert._assert(quad == b.getDestination().getQuad());
						
			this._bindings.addLast(b);
		}
		
		public Iterator getBindingIterator(){
			return _bindings.iterator();
		}
		
		public int size(){return _bindings.size();}
		
		public String toString(String prepend){
			String result = "";
			for(Iterator iter = _bindings.iterator(); iter.hasNext();){
				SSABinding b = (SSABinding)iter.next();
				result += prepend + b.toString() + "\n";
			}
			
			return result;
		}
		
		public String toString(){return toString("");}
	}
}

package Compil3r.Analysis.IPSSA;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import Util.Assert;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.PAResults;
import Compil3r.Analysis.IPA.PointerAnalysisResults;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operator;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.Quad.QuadIterator;
import Compil3r.Quad.QuadVisitor;
import Compil3r.Quad.CodeCache;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.Analysis.IPSSA.SSAProcInfo.Helper;
import Compil3r.Analysis.IPSSA.SSAProcInfo.Query;
import Compil3r.Analysis.IPSSA.SSAProcInfo.SSABindingAnnote;
import Compil3r.Analysis.IPSSA.Utils.SSAGraphPrinter;
import Compil3r.Analysis.IPA.ContextSet;
import Util.Templates.ListIterator;

/**
 * This is where the main action pertaining to IPSSA construction happens. 
 * A subclass is SSABuilder, which is responsible for intraprocedural IPSSA
 * construction.
 * */
public class IPSSABuilder implements ControlFlowGraphVisitor {
	protected int      			   _verbosity;
	private static HashMap 		   _builderMap = new HashMap();
	private PointerAnalysisResults _ptr = null;
	
	boolean PRINT_CFG 		= !System.getProperty("ipssa.print_cfg", "no").equals("no");
	boolean PRINT_SSA_GRAPH = !System.getProperty("ipssa.print_ssa", "no").equals("no");

	public IPSSABuilder(int verbosity){
		CodeCache.AlwaysMap = true;
		this._verbosity     = verbosity;
		// get pointer analysis results			
		try {
			_ptr = PAResults.loadResults(null, null);
		} catch (IOException e) {
			System.err.println("Caught an exception: " + e.toString());
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	/*
	 * Default constructor with verbosity=2.
	 **/
	public IPSSABuilder(){
		this(2);
	}

	// TODO: what's the order in the CFGs are visited? Is there a BU visitor?
	public void visitCFG(ControlFlowGraph cfg) {
		jq_Method method = cfg.getMethod();
	
		SSABuilder builder = new SSABuilder(method, _ptr, _verbosity);
		_builderMap.put(method, builder);		// TODO: do we really need to hash them?
		builder.run(); 
	}
	
	/** The return result may be NULL */
	public static SSABuilder getBuilder(jq_Method m){
		return (SSABuilder)_builderMap.get(m);
	}
	
	class SSABuilder implements Runnable {
		protected int      				_verbosity;
		protected jq_Method 			_method;
		protected ControlFlowGraph 		_cfg;
		protected SSAProcInfo.Query 	_q;
		private PointerAnalysisResults 	_ptr;
		
		SSABuilder(jq_Method method, PointerAnalysisResults ptr, int verbosity){
			this._method 	= method;
			this._cfg 		= CodeCache.getCode(_method);
			this._verbosity = verbosity;
			this._q         = SSAProcInfo.retrieveQuery(_method); 
			this._ptr    	= ptr;
		}		

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/***************************************** Auxilary routines ************************************/
		//////////////////////////////////////////////////////////////////////////////////////////////////	
		protected int addBinding(Quad quad, SSALocation loc, SSAValue value){
			if(_ptr.hasAliases(_method, loc)){
				// add the binding to potential aliased locations
				int i = 0;
				for(Iterator iter = _ptr.getAliases(_method, loc).iterator(); iter.hasNext();){
					ContextSet.ContextLocationPair clPair = (ContextSet.ContextLocationPair)iter.next();
						
					// process aliasedLocation
					i += addBinding(quad, clPair.getLocation(), value, clPair.getContext());									
				}
				return i;
			}else{
				addBinding(quad, loc, value, null);
				return 1;
			}					
		}
			
		/**
		 * This is used by addBinding(Quad quad, SSALocation loc, SSAValue value) and
		 * should never be called directly.
		 * */
		private int addBinding(Quad quad, SSALocation loc, SSAValue value, ContextSet context){
			// initialize the location
			if(quad != _q.getFirstQuad()){
				initializeLocation(loc);
			}
	
			SSABindingAnnote ba = (SSABindingAnnote)_q._bindingMap.get(quad);
			if(ba == null){
				ba = new SSABindingAnnote();
				_q._bindingMap.put(quad, ba);
			}
			
			int result = 0;
			if(context == null){
				ba.addBinding(loc, value, quad);
				result++;
				if(quad != _q.getFirstQuad()){
					result += markIteratedDominanceFrontier(loc, quad);					
				}
			}else{
				// TODO: if(quad != _firstQuad)
				SSADefinition tmpForValue = makeTemporary(value, quad, context);
				result++;
				SSADefinition lastDef = _q.getLastDefinitionFor(loc, quad, true);
				
				SSAValue.SigmaPhi sigma = new SSAValue.SigmaPhi(context, tmpForValue, lastDef);
				ba.addBinding(loc, sigma, quad);
				result++;
				result += markIteratedDominanceFrontier(loc, quad);
			}
			
			return result;
		}
			
		/**
		 * This is used by addBinding(...) routines and should not be called directly.
		 * */
		private int initializeLocation(SSALocation loc) {			
			if(_q.getDefinitionFor(loc, _q.getFirstQuad()) == null){
				if(loc instanceof LocalLocation){
					// no previous value to speak of for the locals
					return addBinding(_q.getFirstQuad(), loc, null, null);
				}else{
					// the RHS is always a FormalIn
					return addBinding(_q.getFirstQuad(), loc, new SSAValue.FormalIn(), null);
				}
			}else{
				return 0;
			}								
		}
		
		/**
		 * Creates new empty definitions at the dominance frontier of quad for 
		 * location loc.
		 */
		private int markIteratedDominanceFrontier(SSALocation loc, Quad quad) {
			if(loc instanceof SSALocation.Unique){
				// don't create Gamma nodes for unique locations
				return 0;
			}
			int result = 0;
			HashSet set = new HashSet();
			_q.getDominatorQuery().getIteratedDominanceFrontier(quad, set);
			if(_verbosity > 2) System.err.println("There are " + set.size() + " element(s) on the frontier");
			
			for(Iterator iter = set.iterator(); iter.hasNext();){
				Quad dom = (Quad)iter.next();
				//if(dom == quad) continue;
				if(_q.getDefinitionFor(loc, dom) == null){				
					SSAValue.Gamma gamma = new SSAValue.Gamma();
					
					// to be filled in later
					result += addBinding(dom, loc, gamma, null);
					if(_verbosity > 1) System.err.println("Created a gamma function for " + loc + " at " + dom);
				}else{
					// TODO: fill the gamma?
				}
			}
			
			return result;		
		}
			
		/**
		 * Creates a temporary definition at quad with the RHS value in 
		 * the given context.
		 * */
		private SSADefinition makeTemporary(SSAValue value, Quad quad, ContextSet context) {
			// TODO We need to create a temporary definition at quad
			SSALocation.Temporary temp = SSALocation.Temporary.FACTORY.get();
				
			addBinding(quad, temp, value, context);
			
			SSADefinition def = _q.getDefinitionFor(temp, quad);
			Assert._assert(def != null);
			
			return def; 
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/******************************************** Stages ********************************************/
		//////////////////////////////////////////////////////////////////////////////////////////////////		
		public void run(){
			SSAProcInfo.Query q = SSAProcInfo.retrieveQuery(_method);
			if(_verbosity>2) System.out.println("Created query: " + q.toString());
			if(_verbosity > 0){
				String name = _method.toString();
				if(name.length() > 40){
					name = name.substring(40);
				}else{
					name = repeat(" ", 40-name.length())+name;
				}
				System.out.println("============= Processing method " + name + " in IPSSABuilder =============");
			}
			
			/*
			 * Stages of intraprocedural processing:
			 * 	Stage 1     : Process all statements in turn and create slots for each modified location.
			 *  Invariant 1 : All necessary assignments are created by this point and all definitions are numbered.
			 *  
			 * 	Stage 2     : Walk over and fill in all RHSs that don't require dereferencing.
			 *  Invariant 2 : All remaining RHSs that haven't been filled in require dereferencing.
			 * 
			 *  Stage 3     : Walk over and do all remaining pointer resolution.
			 *  Invariant 3 : All RHSs are filled in.
			 * */
			// 1. 			
			Stage1Visitor vis1 = new Stage1Visitor(_method);  
			for (QuadIterator j=new QuadIterator(_cfg, true); j.hasNext(); ) {
				Quad quad = j.nextQuad();
				quad.accept(vis1);
			}			
			if(_verbosity > 1){
				System.err.println("Created a total of " + vis1.getBindingCount() + " bindings");
			}
			vis1 = null;

/*
			//	2.
			Stage2Visitor vis2 = new Stage2Visitor();
			vis2.visitCFG(_cfg);
			
			//	3.			
			Stage3Visitor vis3 = new Stage3Visitor();  
			vis3.visitCFG(_cfg);
*/			
			Stage2PrimeVisitor vis2 = new Stage2PrimeVisitor(_method);  
			for (QuadIterator j=new QuadIterator(_cfg, true); j.hasNext(); ) {
				Quad quad = j.nextQuad();
				quad.accept(vis2);
			}
						
			/** Now print the results */
			if(PRINT_CFG){
				// print the CFG annotated with SSA information
				_q.printDot();	
			}
			
			if(PRINT_SSA_GRAPH) {
				try {
					FileOutputStream file = new FileOutputStream("ssa.dot");
					PrintStream out = new PrintStream(file);
					SSAGraphPrinter.printAllToDot(out);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(2);
				}				
			}
		}
		/** 
		 * Stage 1     : Process all statements in turn and create slots for each modified location. 
		 * Invariant 1 : All necessary assignments are created by this point and all definitions are numbered.
		 * */
		class Stage1Visitor extends QuadVisitor.EmptyVisitor {
			jq_Method _method;
			SSAProcInfo.Helper _h;
			SSAProcInfo.Query  _q;
			private int        _bindings;
			
			Stage1Visitor(jq_Method method){
				this._method   = method;
				this._h 	   = SSAProcInfo.retrieveHelper(_method);
				this._q 	   = SSAProcInfo.retrieveQuery(_method);
				this._bindings = 0;
			}
			
			int getBindingCount(){
				return _bindings;
			}		
			
			/**************************** Begin handlers ****************************/
			/** A get static field instruction. */
			public void visitGetstatic(Quad quad) {
				processLoad(quad);
			}
			/** A get instance field instruction. */
			public void visitGetfield(Quad quad) {
				processLoad(quad);
			}
			private void processLoad(Quad quad) {
				markDestinations(quad);				
			}			
			/** A put instance field instruction. */
			public void visitPutfield(Quad quad) {
				processStore(quad);
			}
			/** A put static field instruction. */
			public void visitPutstatic(Quad quad) {
				processStore(quad);
			}
			/** A register move instruction. */
			public void visitMove(Quad quad) {
				markDestinations(quad);
			}
			/** An array load instruction. */
			public void visitALoad(Quad quad) {
				processLoad(quad);
			}
			/** An array store instruction. */
			public void visitAStore(Quad quad) {
				print(quad);
			}
			/** An quadect allocation instruction. */
			public void visitNew(Quad quad) {
				markDestinations(quad);
			}
			/** An array allocation instruction. */
			public void visitNewArray(Quad quad) {
				markDestinations(quad);
			}
			/** A return from method instruction. */
			public void visitReturn(Quad quad) {
				// TODO: make up a location for return?
				print(quad);
			}			
			public void visitInvoke(Quad quad) {
				//printAlways(quad);
				processDefs(quad);	
			}
			/**************************** End of handlers ****************************/ 
			
			private void markDestinations(Quad quad) {				
				Register reg = getOnlyDefinedRegister(quad); 
				Assert._assert(reg != null);
				LocalLocation loc = LocalLocation.FACTORY.createLocalLocation(reg);

				addBinding(quad, loc, null, null);
			}
			private void processStore(Quad quad) {
				processDefs(quad);
			}
	
			private void processDefs(Quad quad) {
				QuadProgramLocation pl = new QuadProgramLocation(_method, quad);
				Assert._assert(isCall(quad) || isStore(quad));
				Set mods = _ptr.mod(pl);

				// create bindingins for all modified locations
				if(mods != null && mods.size() > 0){
					if(_verbosity > 2) System.out.print("Found " + mods.size() + " mods at " + pl.toString() + ": [ ");
					Iterator iter = mods.iterator();
					while(iter.hasNext()){
						SSALocation loc = (SSALocation)iter.next();
						if(_verbosity > 2) System.out.print(loc.toString(_ptr.getPAResults()) + " ");
						if(isCall(quad)){
							_bindings += addBinding(quad, loc, new SSAValue.ActualOut(), null);
						}else
						if(isStore(quad)){
							_bindings += addBinding(quad, loc, null, null);
						}else{
							Assert._assert(false);
						}
					}
					if(_verbosity > 2) System.out.println("]\n");
				}
			}

			/** Any quad */
			public void visitQuad(Quad quad) {print(quad);}
			
			protected void print(Quad quad, boolean force){
				if(!force) return;
				ProgramLocation loc = new QuadProgramLocation(_method, quad);
				String loc_str = null;
				
				try {
					loc_str = loc.getSourceFile() + ":" + loc.getLineNumber();
				}catch(Exception e){
					loc_str = "<unknown>";
				}
				
				System.out.println("Visited quad # " + quad.toString() + "\t\t\t at " + loc_str);
			}
			
			protected void printAlways(Quad quad){
				print(quad, true);
			}
			
			protected void print(Quad quad){
				print(quad, false);
			}
						
			protected void warn(String s){
				System.err.println(s);
			}
		}
		
		/** 
		 * Stage 2     : Walk over and fill in all RHSs that don't require dereferencing. 
		 * Invariant 2 : All remaining RHSs that haven't been filled in require dereferencing.		
		 * */
		class Stage2Visitor implements ControlFlowGraphVisitor {
			public void visitCFG(ControlFlowGraph cfg) {
				SSAProcInfo.Query q = SSAProcInfo.retrieveQuery(cfg.getMethod()); 
				for(Iterator iter = new QuadIterator(cfg); iter.hasNext();){
					Quad quad = (Quad)iter.next();
					for(Iterator bindingIter = q.getBindingIterator(quad); bindingIter.hasNext(); ){
						SSABinding b = (SSABinding)bindingIter.next();
						
						if(!isStore(quad) && !isLoad(quad) && !isCall(quad)){
							specialize(quad);
						}
					} 
				}
			}

			void specialize(Quad quad) {
							
			}
		}
		
		/** 
		 * Stage 3	   : Walk over and do all remaining pointer resolution. 
		 * Invariant 3 : All RHSs are filled in.
		 * */
		final class Stage3Visitor extends SSABindingVisitor {
			public void visit(SSABinding b) {
				Quad quad = b.getQuad();
				
				if(isStore(quad)){
					// rewrite a store
					processStore(quad);
				}else
				if(isLoad(quad)){
					// rewrite a load
					processLoad(quad);
				}
			}

			private void processStore(Quad quad) {
				// TODO Auto-generated method stub				
			}

			private void processLoad(Quad quad) {
				// TODO Auto-generated method stub				
			}			
		}
	
		/** 
		 * Stage 2     : Walk over and fill in all RHSs that don't require dereferencing. 
		 * Invariant 2 : Update RHSs referring to heap objects to refer to the right locations.
		 * */
		class Stage2PrimeVisitor extends QuadVisitor.EmptyVisitor {
			private jq_Method _method;
			private Query     _q;
			private Helper    _h;

			Stage2PrimeVisitor(jq_Method method){
				this._method   = method;
				this._h 	   = SSAProcInfo.retrieveHelper(_method);
				this._q 	   = SSAProcInfo.retrieveQuery(_method);
			}
			
			/**************************** Begin handlers ****************************/
			/** A get static field instruction. */
			public void visitGetstatic(Quad quad) {
				processLoad(quad);
			}
			/** A get instance field instruction. */
			public void visitGetfield(Quad quad) {
                processLoad(quad);
			}
			/** A put instance field instruction. */
			public void visitPutfield(Quad quad) {
				processStore(quad);
			}
			/** A put static field instruction. */
			public void visitPutstatic(Quad quad) {
				processStore(quad);
			}
			/** A register move instruction. */
			public void visitMove(Quad quad) {
				// there is only one binding at this quad
				Assert._assert(_q.getBindingCount(quad) == 1);
				SSABinding b = (SSABinding) _q.getBindingIterator(quad).next();
				Assert._assert(b.getValue() == null);
				b.setValue(markUses(quad));
			}
			/** An array load instruction. */
			public void visitALoad(Quad quad) {
                processLoad(quad);
			}
			/** An array store instruction. */
			public void visitAStore(Quad quad) {
				processStore(quad);
			}
			/** An quadect allocation instruction. */
			public void visitNew(Quad quad) {
				 // there is only one binding at this quad
				 Assert._assert(_q.getBindingCount(quad) == 1);
				 SSABinding b = (SSABinding) _q.getBindingIterator(quad).next();
				 Assert._assert(b.getValue() == null);
				 b.setValue(makeAlloc(quad));
			}
			/** An array allocation instruction. */
			public void visitNewArray(Quad quad) {
				// there is only one binding at this quad
				 Assert._assert(_q.getBindingCount(quad) == 1);
				 SSABinding b = (SSABinding) _q.getBindingIterator(quad).next();
				 Assert._assert(b.getValue() == null);
				 b.setValue(makeAlloc(quad));
			}
			/** A return from method instruction. */
			public void visitReturn(Quad quad) {
				// TODO: make up a location for return?
			}			
			public void visitInvoke(Quad quad) {
				processCall(quad);	
			}
			/**************************** End of handlers ****************************/ 
			private void processStore(Quad quad) {
				// the destinations have been marked at this point
				// need to fill in the RHSs
				for(Iterator iter = _q.getBindingIterator(quad); iter.hasNext();) {  					
					SSABinding b = (SSABinding) iter.next();
					Assert._assert(b.getValue() == null);
					b.setValue(markUses(quad));
				}
			}
			
			public void visitCFG(ControlFlowGraph cfg) {
				// fill in all the gammas
				for(Iterator iter = new QuadIterator(cfg); iter.hasNext();){
					Quad quad = (Quad)iter.next();
					
					Iterator bindingIter = _q.getBindingIterator(quad);
					while(bindingIter.hasNext()){
						SSABinding b = (SSABinding) bindingIter.next();
						SSAValue value = b.getValue();
						
						if(value != null && value instanceof SSAValue.Gamma){
							SSAValue.Gamma gamma = (SSAValue.Gamma)value;
							fillInGamma(quad, gamma);
						}
					}
				}
			}

			/**
			 * Fill in the gamma function with reaching definitions
			 * */
			private void fillInGamma(Quad quad, SSAValue.Gamma gamma) {
				SSALocation loc = gamma.getDestination().getLocation();				
				
				BasicBlock basicBlock = _q.getDominatorQuery().getBasicBlock(quad);
				Assert._assert(basicBlock != null);
				Assert._assert(basicBlock.size() > 0);
				Assert._assert(basicBlock.getQuad(0) == quad);
				ListIterator.BasicBlock predIter = basicBlock.getPredecessors().basicBlockIterator();
				while(predIter.hasNext()){
					BasicBlock predBlock = predIter.nextBasicBlock();
					Quad predQuad = predBlock.isEntry() ? _q.getFirstQuad() : predBlock.getLastQuad();
					SSADefinition predDef = _q.getLastDefinitionFor(loc, predQuad, false);
					gamma.add(predDef, null);
				}
			}

            /**
             * This method fills in the RHS of loads.
             * */
			private void processLoad(Quad quad) {
				QuadProgramLocation pl = new QuadProgramLocation(_method, quad);
				Assert._assert(isLoad(quad));
				Set refs = _ptr.ref(pl);

                SSAValue.OmegaPhi value = new SSAValue.OmegaPhi(); 

				// create bindingins for all modified locations
				if(refs != null && refs.size() > 0){
					if(_verbosity > 2) System.out.print("Found " + refs.size() + " refs at " + pl.toString() + ": [ ");
					Iterator iter = refs.iterator();
					while(iter.hasNext()){
						SSALocation loc = (SSALocation)iter.next();
						if(_verbosity > 2) System.out.print(loc.toString(_ptr.getPAResults()) + " ");
						// figure out the reaching definition for loc
						initializeLocation(loc);
						SSADefinition def = _q.getLastDefinitionFor(loc, quad, true);
						Assert._assert(def != null);						
						if(_verbosity > 1) System.out.println("Using " + def + " at " + quad);
						value.addUsedDefinition(def);
					}
					if(_verbosity > 2) System.out.println("]\n");
                }
                
                Assert._assert(_q.getBindingCount(quad) == 1, "Have " + _q.getBindingCount(quad) + " bindings at " + quad);
                SSABinding b = (SSABinding) _q.getBindingIterator(quad).next();
                Assert._assert(b.getValue() == null);
                Assert._assert(b.getDestination().getLocation() instanceof LocalLocation);
                LocalLocation loc = (LocalLocation) b.getDestination().getLocation();
                Assert._assert(loc.getRegister() == getOnlyDefinedRegister(quad));
                b.setValue(value);
			}
            
            private void processCall(Quad quad) {
                Assert._assert(isCall(quad));   
            }
			
			private SSAValue.Normal markUses(Quad quad) {
				SSAValue.UseCollection value = SSAValue.UseCollection.FACTORY.createUseCollection();
				ListIterator.RegisterOperand iter = quad.getUsedRegisters().registerOperandIterator();
				while(iter.hasNext()) {
					Register reg = iter.nextRegisterOperand().getRegister();
					SSALocation loc = LocalLocation.FACTORY.createLocalLocation(reg);
					initializeLocation(loc);
					SSADefinition  def =_q.getLastDefinitionFor(loc, quad, true);
					Assert._assert(def != null);
					
					value.addUsedDefinition(def);
				}
								
				return value;
			}
			
			private SSAValue makeAlloc(Quad quad) {
				return SSAValue.Alloc.FACTORY.createAlloc(quad);
			}
		}
	} // End of SSABuilder
	
	static boolean isLoad(Quad quad) {
		return 
			(quad.getOperator() instanceof Operator.Getfield) ||
			(quad.getOperator() instanceof Operator.Getstatic);
	}
	static boolean isStore(Quad quad) {
		return
			(quad.getOperator() instanceof Operator.Putfield) ||
			(quad.getOperator() instanceof Operator.Putstatic);
	}
	boolean isCall(Quad quad) {
		return (quad.getOperator() instanceof Operator.Invoke);	// TODO: anything else?
	}
	private static String repeat(String string, int n) {
		StringBuffer result = new StringBuffer();
		for(int i = 0; i<n; i++) result.append(string);
		
		return result.toString();
	}
	private static Register getOnlyDefinedRegister(Quad quad) {
		Util.Templates.ListIterator.RegisterOperand iter = quad.getDefinedRegisters().registerOperandIterator();
		if(!iter.hasNext()){
			// no definition here
			return null;
		}
		Register reg = iter.nextRegisterOperand().getRegister();
		Assert._assert(!iter.hasNext(), "More than one defined register");
			
		return reg;
	}
    private static Register getOnlyUsedRegister(Quad quad) {
        Util.Templates.ListIterator.RegisterOperand iter = quad.getUsedRegisters().registerOperandIterator();
        if(!iter.hasNext()){
            // no definition here
            return null;
        }
        Register reg = iter.nextRegisterOperand().getRegister();
        Assert._assert(!iter.hasNext(), "More than one used register");
        
        return reg;
    }
};

			/********************************************************/
			/** The stuff below is not intended to be implemented  **/
			/********************************************************/ 
	
//			/** A potentially excepting instruction. */
//			public void visitExceptionThrower(Quad obj) {}
//			/** An instruction that may branch (not including exceptional control flow). */
//			public void visitBranch(Quad obj) {}
//			/** A conditional branch instruction. */
//			public void visitCondBranch(Quad obj) {}
//			/** An exception check instruction. */
//			public void visitCheck(Quad obj) {}
//			/** An instruction.that accesses an array. */
//			public void visitArray(Quad obj) {}
//			/** An instruction.that does an allocation. */
//			public void visitAllocation(Quad obj) {}
//			
//			/** An instruction.that does a type check. */
//			public void visitTypeCheck(Quad obj) {/* NOOP */}
//			
//			/** An array length instruction. */
//			public void visitALength(Quad obj) {/* NOOP */}
//			/** A binary operation instruction. */
//			public void visitBinary(Quad obj) {}
//			/** An array bounds check instruction. */
//			public void visitBoundsCheck(Quad obj) {/* NOOP */}
//			/** A type cast check instruction. */
//			public void visitCheckCast(Quad obj) {/* NOOP */}
//			/** A goto instruction. */
//			public void visitGoto(Quad obj) {/* NOOP */}
//			/** A type instance of instruction. */
//			public void visitInstanceOf(Quad obj) {/* NOOP */}
//			/** A compare and branch instruction. */
//			public void visitIntIfCmp(Quad obj) {/* NOOP */}
//			/** An invoke instruction. */
//			public void visitInvoke(Quad obj) {}
//			/** A jump local subroutine instruction. */
//			public void visitJsr(Quad obj) {}
//			/** A lookup switch instruction. */
//			public void visitLookupSwitch(Quad obj) {}		
//			
//			/** A raw memory load instruction. */
//			
//			public void visitMemLoad(Quad obj) {}
//			/** A raw memory store instruction. */
//			public void visitMemStore(Quad obj) {}
//			
//			/** An object monitor lock/unlock instruction. */
//			public void visitMonitor(Quad obj) {}
//			/** A null pointer check instruction. */
//			public void visitNullCheck(Quad obj) {/* NOOP */}
//			/** A return from local subroutine instruction. */
//			public void visitRet(Quad obj) {}
//			/** A special instruction. */
//			public void visitSpecial(Quad obj) {}
//			/** An object array store type check instruction. */
//			public void visitStoreCheck(Quad obj) {}
//			/** A jump table switch instruction. */
//			public void visitTableSwitch(Quad obj) {}
//			/** A unary operation instruction. */
//			public void visitUnary(Quad obj) {}
//			/** A divide-by-zero check instruction. */
//			public void visitZeroCheck(Quad obj) {}
//			
//			/** An instruction that loads from memory. */
//			public void visitLoad(Quad obj) {}
//			/** An instruction that stores into memory. */
//			public void visitStore(Quad obj) {}
//			/** An instruction.that accesses a static field. */
//			public void visitStaticField(Quad obj) {}
//			/** An instruction that accesses an instance field. */
//			public void visitInstanceField(Quad obj) {}      

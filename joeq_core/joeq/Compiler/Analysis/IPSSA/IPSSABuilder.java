package Compil3r.Analysis.IPSSA;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import Clazz.jq_Field;
import Clazz.jq_Method;
import Compil3r.Analysis.IPA.PointerAnalysisResults;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operator;
import Compil3r.Quad.Operand;
import Compil3r.Quad.Operator.Getfield;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.Quad.QuadIterator;
import Compil3r.Quad.QuadVisitor;
import Compil3r.Quad.CodeCache;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.Analysis.IPSSA.SSAProcInfo.SSABindingAnnote;
import Compil3r.Analysis.IPA.ContextSet;

/**
 * This is where the main action pertaining to IPSSA construction happens. 
 * A subclass is SSABuilder, which is responsible for intraprocedural IPSSA
 * construction.
 * */
public class IPSSABuilder implements ControlFlowGraphVisitor {
	protected int      			_verbosity;
	//protected jq_Method 		_method;
	private static HashMap 		_builderMap = new HashMap();

	public IPSSABuilder(int verbosity){
		CodeCache.AlwaysMap = true;
		this._verbosity     = verbosity;
	}
		
	public IPSSABuilder(){
		this(1);
	}

	// TODO: what's the order in the CFGs are visited? Is there a BU visitor?
	public void visitCFG(ControlFlowGraph cfg) {
		jq_Method method = cfg.getMethod();
	
		SSABuilder builder = new SSABuilder(method, _verbosity);
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
		
		SSABuilder(jq_Method method, int verbosity){
			this._method 	= method;
			this._cfg 		= CodeCache.getCode(_method);
			this._verbosity = verbosity;
			this._q         = SSAProcInfo.retrieveQuery(_method); 
			this._ptr    	= null;		// TODO!
		}		

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/***************************************** Auxilary routines ************************************/
		//////////////////////////////////////////////////////////////////////////////////////////////////	
		protected int addBinding(Quad quad, SSALocation loc, SSAValue value){
			if(_ptr.hasAliases(_method, loc)){
				// add the binding to potential aliased locations
				int i = 0;
				for(Iterator iter = _ptr.getAliases(_method, loc).iterator(); iter.hasNext(); i++){
					ContextSet.ContextLocationPair clPair = (ContextSet.ContextLocationPair)iter.next();
						
					// process aliasedLocation
					addBinding(quad, clPair.getLocation(), value, clPair.getContext());									
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
		private SSADefinition addBinding(Quad quad, SSALocation loc, SSAValue value, ContextSet context){
			// initialize the location
			initializeLocation(loc);
	
			SSABindingAnnote ba = (SSABindingAnnote)_q._bindingMap.get(quad);
			if(ba == null){
				ba = new SSABindingAnnote();
			}
			
			SSADefinition result = null;
			if(context == null){
				result = ba.addBinding(loc, value, quad);
				markIteratedDominanceFrontier(loc, quad);					
			}else{
				SSADefinition tmpForValue = makeTemporary(value, quad, context);
				SSADefinition lastDef = _q.getLastDefinitionFor(loc, quad, true);
				
				SSAValue.SigmaPhi sigma = new SSAValue.SigmaPhi(context, tmpForValue, lastDef);
				result = ba.addBinding(loc, sigma, quad);
				markIteratedDominanceFrontier(loc, quad);
			}
			
			return result;
		}
			
		/**
		 * This is used by addBinding(...) routines and should not be called directly.
		 * */
		private void initializeLocation(SSALocation loc) {
			Quad firstQuad = CodeCache.getCode(_method).entry().getQuad(0);
			if(_q.getDefinitionFor(loc, firstQuad) == null){
				addBinding(firstQuad, loc, new SSAValue.FormalIn(), null);
			}								
		}
		
		/**
		 * Creates new empty definitions at the dominance frontier of quad for 
		 * location loc.
		 */
		private void markIteratedDominanceFrontier(SSALocation loc, Quad quad) {
			HashSet set = new HashSet();
			_q.getDominatorQuery().getIteratedDominanceFrontier(quad, set);
			
			for(Iterator iter = set.iterator(); iter.hasNext();){
				Quad dom = (Quad)iter.next();
				
				SSAValue.Gamma gamma = new SSAValue.Gamma();
				
				// to be filled in later
				addBinding(dom, loc, gamma, null); 
			}			
		}
			
		/**
		 * Creates a temporary definition at quad with the RHS value in 
		 * the given context.
		 * */
		private SSADefinition makeTemporary(SSAValue value, Quad quad, ContextSet context) {
			// TODO We need to create a temporary definition at quad
			SSALocation.Temporary temp = SSALocation.Temporary.FACTORY.get();
				
			return addBinding(quad, temp, value, context); 
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
			PointerAnalysisResults ptrResults = null;	// TODO: need to add results
			Stage1Visitor vis1 = new Stage1Visitor(_method);  
			for (QuadIterator j=new QuadIterator(_cfg, true); j.hasNext(); ) {
				Quad quad = j.nextQuad();
				quad.accept(vis1);
			}
			//	2.
			Stage2Visitor vis2 = new Stage2Visitor();
			vis2.visitCFG(_cfg);
			
			//	3.			
			Stage3Visitor vis3 = new Stage3Visitor();  
			vis3.visitCFG(_cfg);
		} 
		/** 
		 * Stage 1     : Process all statements in turn and create slots for each modified location. 
		 * Invariant 1 : All necessary assignments are created by this point and all definitions are numbered.
		 * */
		class Stage1Visitor extends QuadVisitor.EmptyVisitor {
			jq_Method _method;
			SSAProcInfo.Helper _h;
			SSAProcInfo.Query  _q;
			
			Stage1Visitor(jq_Method method){
				this._method = method;
				this._h 	 = SSAProcInfo.retrieveHelper(_method);
				this._q 	 = SSAProcInfo.retrieveQuery(_method);
			}		
			
			/** A get static field instruction. */
			public void visitGetstatic(Quad obj) {
				processLoad(obj);
			}
	
			/** A get instance field instruction. */
			public void visitGetfield(Quad obj) {
				processLoad(obj);
				
				/*
				if (obj.getOperator() instanceof Operator.Getfield.GETFIELD_A
			     || obj.getOperator() instanceof Operator.Getfield.GETFIELD_P) 
				{
					Register r = Getfield.getDest(obj).getRegister();
					Operand o = Getfield.getBase(obj);
					Getfield.getField(obj).resolve();
					jq_Field f = Getfield.getField(obj).getField();
					//System.out.println("\nField = " + f.toString());
					
					if (o instanceof RegisterOperand) {
						Register b = ((RegisterOperand)o).getRegister();
						//ProgramLocation pl = new QuadProgramLocation(method, obj);
						//heapLoad(pl, r, b, f);
					} else {
						// base is not a register?!
						warn("1");
					}
				}else{
					warn("2");
				}*/				
			}
			private void processLoad(Quad quad) {
				//Set set = _ptr.pointsTo(new QuadProgramLocation(_method, obj));
				print(quad);				
			}
			
			private void processStore(Quad quad) {
				//Set set = _ptr.pointsTo(new QuadProgramLocation(_method, quad));
				// We need to create SSABindings for evere location in the set
				//for(Iterator iter = set.iterator(); iter.hasNext();){
				//	SSALocation loc = (SSALocation)iter.next();
				//	
				//	/*int count = */
				//	addBinding(quad, loc, null, null);
				//} 
				//
				//print(quad);
			}
			/** A put instance field instruction. */
			public void visitPutfield(Quad obj) {
				processStore(obj);
			}
			/** A put static field instruction. */
			public void visitPutstatic(Quad obj) {
				processStore(obj);
			}
			/** A register move instruction. */
			public void visitMove(Quad obj) {
				//print(obj);
				//obj.getDefinedRegisters().registerOperandIterator().nextRegisterOperand()
			}
			/** An array load instruction. */
			public void visitALoad(Quad obj) {
				print(obj);
			}
			/** An array store instruction. */
			public void visitAStore(Quad obj) {
				print(obj);
			}
			/** An object allocation instruction. */
			public void visitNew(Quad obj) {
				print(obj);
			}
			/** An array allocation instruction. */
			public void visitNewArray(Quad obj) {
				print(obj);
			}
			/** A return from method instruction. */
			public void visitReturn(Quad obj) {
				print(obj);
			}
	
			/** Any quad */
			public void visitQuad(Quad obj) {print(obj);}
			
			protected void print(Quad obj){
				ProgramLocation loc = new QuadProgramLocation(_method, obj);
				String loc_str = null;
				
				try {
					loc_str = loc.getSourceFile() + ":" + loc.getLineNumber();
				}catch(Exception e){
					loc_str = "<unknown>";
				}
				
				System.out.println("Visited quad # " + obj.toString() + "\t\t\t at " + loc_str);
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
		class Stage3Visitor extends SSABindingVisitor {
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

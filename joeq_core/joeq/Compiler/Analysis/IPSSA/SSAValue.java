package Compil3r.Analysis.IPSSA;

import java.util.Iterator;
import java.util.Vector;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.ContextSet;
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operator.Invoke;

import Util.Assert;

public abstract class  SSAValue {
	protected SSADefinition _destination;
	
	public SSADefinition getDestination(){
		return _destination;
	}
	
	public Quad getQuad(){
		return _destination.getQuad();
	}
	
	void setDestination(SSADefinition def){
		_destination = def;
	}
	
	public abstract Iterator getUsedDefinitionIterator();			
		
	/**
	 * This value is just a reference to a definition.
	 * TODO: do we still have copies in the reduced representation?..
	 * */
	public static class Copy extends SSAValue {
		SSADefinition _definition;
		
		private Copy(SSADefinition def){
			this._definition = def;
		}
		public static class FACTORY {
			Copy create_copy(SSADefinition def){
				return new Copy(def);
			}
		}
		
		public Iterator/*<SSADefinition>*/ getUsedDefinitionIterator(){
			return new IteratorHelper.SingleIterator(_definition);
		}
		
		public SSADefinition getDefinition(){return _definition;}		
	}	

	public static abstract class Terminal extends SSAValue {}
	
	public static abstract class Constant extends Terminal {
		public Iterator/*<SSADefinition>*/ getUsedDefinitionIterator(){
			return IteratorHelper.EmptyIterator.FACTORY.get();
		}
	}
	
	public static class UnknownConstant extends Constant {
		/** Use UnknownContant.FACTORY */
		private UnknownConstant(){}
		
		public static class FACTORY {
			static UnknownConstant _sample = null;					
			public static UnknownConstant create_unknown_contant(){
				if(_sample == null){
					_sample = new UnknownConstant();
				}
				return _sample;				
			}
		}
		
		public String toString(){return "<Unknown>";}
	}
	
	public static class NullConstant extends Constant {
		/** Use NullContant.FACTORY */
		private NullConstant(){}

		public static class FACTORY {
			static NullConstant _sample = null;					
			public static NullConstant create_null_contant(){
				if(_sample == null){
					_sample = new NullConstant();
				}
				return _sample;				
			}
		}
		public String toString(){return "<Null>";}
	}
	
	public static class Normal extends Terminal {
		// TODO: this may contain arbitrary expressions
		public Iterator/*<SSADefinition>*/ getUsedDefinitionIterator(){
			return null;	// TODO
		}
	}

	public static abstract class Phi extends  SSAValue {
		protected Vector/* <SSADefinition> */ _definitions;
		
		public int getDefinitionCount(){
			return _definitions.size();
		}
		public SSADefinition getDefinition(int pos){
			return (SSADefinition)_definitions.get(pos);
		}
		public Iterator/*<SSADefinition>*/ getDefinitionIterator(){
			return _definitions.iterator();
		}
		
		public Iterator/*<SSADefinition>*/ getUsedDefinitionIterator(){
			return _definitions.iterator();
		}
		
		abstract public String getLetter();
		
		public String toString(){
			String result = getLetter() + "(";
			for(int i = 0; i < _definitions.size(); i++){
				SSADefinition def = getDefinition(i);
				
				result += def.toString() + ", ";
			}
			if(_definitions.size()>0){
				result = result.substring(result.length() - 2);
			}
			
			return result + ")";
		}
	}
	
	/**
	 * 	The representation of predicates is yet to be determined. It's currently pretty lame.
	 * */
	public static class Predicate {
		private String _predicate;

		public Predicate(String predicate){
			this._predicate = predicate;
		}
		public String toString(){
			return _predicate;
		}
		public static Predicate True() {
			return null;		// TODO
		}
	}
	
	public static abstract class Predicated extends Phi {
		protected Vector/* <SSAPredicate> */ _predicates;			
		
		public Predicate getPredicate(int pos){
			return (Predicate)_predicates.get(pos);
		}
		
		public void add(SSADefinition def, String predicate){
			_definitions.addElement(def);
			_predicates.addElement(predicate);
		}
		
		public String toString(){
			String result = getLetter() + "(";
			for(int i = 0; i < _definitions.size(); i++){
				SSADefinition def = getDefinition(i);
				Predicate pred = getPredicate(i);
		
				result += "<" + def.toString() + ", " + pred.toString() + ">, ";
			}
			if(_definitions.size()>0){
				result = result.substring(result.length() - 2);
			}
	
			return result + ")";
		}		
	}
	
	public static class OmegaPhi extends Phi {
		public String getLetter(){return "omega";}
	}
	
	public static class SigmaPhi extends Phi {
		private ContextSet _context;
		
		public SigmaPhi(ContextSet context, SSADefinition newDef, SSADefinition oldDef){
			setContext(context);
			_definitions.add(newDef);
			_definitions.add(oldDef);
		}
		public String getLetter(){return "sigma";}

		protected void setContext(ContextSet _context) {
			this._context = _context;
		}
		protected ContextSet getContext() {
			return _context;
		}
	}

	public static class Gamma extends Predicated {
		public String getLetter(){return "gamma";}	
	}
		
	public static abstract class IPPhi extends Phi {

	}
	
	public static class FormalIn extends IPPhi {
		protected Vector/*<Invoke>*/ _callers;
		
		Invoke getCaller(int pos){
			return (Invoke)_callers.get(pos); 
		}
		void add(SSADefinition def, Invoke caller){
			_definitions.addElement(def);
			_callers.addElement(caller);
		}
		public String getLetter(){return "iota";}
		public String toString(){
			String result = getLetter() + "(";
			for(int i = 0; i < _definitions.size(); i++){
				SSADefinition def = getDefinition(i);
				Invoke caller = getCaller(i);

				result += "<" + def.toString() + ", " + caller + ">, ";
			}
			if(_definitions.size()>0){
				result = result.substring(result.length() - 2);
			}

			return result + ")";
		}
	}
	
	public static class ActualOut extends IPPhi {
		protected Vector/*<jq_Method>*/ _callees;
		
		jq_Method getCallee(int pos){
			return (jq_Method)_callees.get(pos); 
		}
		void add(SSADefinition def, jq_Method method){
			_definitions.addElement(def);
			_callees.addElement(method);
		}
		public String getLetter(){return "rho";}
		
		public String toString(){
			String result = getLetter() + "(";
			for(int i = 0; i < _definitions.size(); i++){
				SSADefinition def = getDefinition(i);
				jq_Method method = getCallee(i);

				result += "<" + def.toString() + ", " + method + ">, ";
			}
			if(_definitions.size()>0){
				result = result.substring(result.length() - 2);
			}

			return result + ")";
		}
	}
	
	public String toString(){
		Assert._assert(false, "Don't call this");
		return "";
	}
}


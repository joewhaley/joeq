package joeq.Compil3r.Analysis.IPSSA;
import joeq.Clazz.jq_Method;
import joeq.Compil3r.Quad.Quad;

/**
 * An SSABinding is an assignment of an SSAValue to to an SSADefinition.
 * @see SSADefinition
 * @see Compil3r.Analysis.IPSSA.SSAValue
 * @see Compil3r.Analysis.IPSSA.SSADefinition
 * @version $Id$
 * */
public class SSABinding {
	protected Quad             _quad;
	protected SSADefinition    _destination;
	protected SSAValue 		   _value;
	
	public SSABinding(Quad quad, SSADefinition def, SSAValue value) {
		this._quad      	= quad;
		this._destination 	= def;
		this._value 		= value;
		
		value.setDestination(def);
	}
	
	public SSABinding(Quad quad, SSALocation loc, SSAValue value, jq_Method method) {
		this._quad = quad;		
		this._value = value;
		
		SSADefinition def = SSAProcInfo.Helper.create_ssa_definition(loc, quad, method);
		this._destination = def;
		
		if(value != null){
			value.setDestination(def);
		}
	}
	
	public void setValue(SSAValue value){
		this._value = value;
		_value.setDestination(_destination);
	}
	
	/** Tests whether the binding has been completed by filling out it RHS */
	public boolean isComplete(){
		return _value != null;
	}
	
	public boolean isValid(){
		return 
			(_value == null || _destination == _value.getDestination()) &&
			(_quad == _destination.getQuad());
	}
	
	public SSADefinition getDestination() {return _destination;}
	public SSAValue getValue() {return _value;}
    public Quad getQuad() {return _destination.getQuad();}

	public void accept(SSABindingVisitor vis) {
		vis.visit(this);		
	}
	
	public String toString(){
		return _destination.toString() + " = " + (_value == null ? "<incomplete>" : _value.toString());
	}
}
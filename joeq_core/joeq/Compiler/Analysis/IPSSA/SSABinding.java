package Compil3r.Analysis.IPSSA;
import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.Quad;

public class SSABinding {
	protected Quad 			_quad;
	protected SSADefinition _destination;
	protected SSAValue 		_value;
	
	public SSABinding(Quad quad, SSADefinition def, SSAValue value) {
		this._quad 			= quad;
		this._destination 	= def;
		this._value 		= value;
		
		value.setDestination(def);
	}
	
	public SSABinding(Quad quad, SSALocation loc, SSAValue value) {
		this._quad = quad;		
		this._value = value;
		
		SSADefinition def = SSAProcInfo.Helper.create_ssa_definition(loc, quad);
		this._destination = def;
		
		value.setDestination(def);
	}
	
	/** Tests whether the binding has been completed by filling out it RHS */
	public boolean isComplete(){
		return _value != null;
	}
	
	public boolean isValid(){
		return 
			(_destination == _value.getDestination()) &&
			(_quad == _destination.getQuad());
	}
	
	public SSADefinition getDestination() {return _destination;}
	public SSAValue getValue() {return _value;}
	public Quad getQuad(){return _quad;}

	public void accept(SSABindingVisitor vis) {
		vis.visit(this);		
	}
	
	public String toString(){
		return _destination.toString() + " = " + (_value == null ? "<incomplete>" : _value.toString());
	}
}
package Compil3r.Analysis.IPA;

//import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
//import Compil3r.Quad.Quad;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;

public class SSALocation {	
	protected Node _location;
	
	public SSALocation(Node node){
		this._location = node;
	}	
	/*
	protected jq_Field 	_location;
	public SSALocation(jq_Field node){
		this._location = node;
	}*/
	public String toString(){
		return _location.toString();
	}
}
package Compil3r.Analysis.IPA;

//import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
//import Compil3r.Quad.Quad;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;

/**
 * @author Vladimir Livshits
 */
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
	
	/**
	 * 	We need to have "abstract" temporary locations for IPSSA construction purposes 
	 * that do not necessarily correspond to anything tangible. 
	 * */
	public static class Temporary extends SSALocation {
		private Temporary(){
			// there's no underlying node
			super(null);
		}
	
		// There's only one Temporary location -- use the FACTORY to retrieve it	
		public static class FACTORY {
			private static Temporary _sample = new Temporary();
			public static Temporary get() {
				return _sample;
			}
		} 
	}
}
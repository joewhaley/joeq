package Compil3r.Analysis.IPSSA;
import java.util.HashMap;

import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.Quad;

public class SSADefinition {
	protected SSALocation	_location;
	int 					_version;
	Quad				    _quad;
	
	static class Helper {
		protected static HashMap/*<SSALocation, Integer>*/ _versionMap;
		
		static SSADefinition create_ssa_definition(SSALocation location, Quad quad){
			int version = 0;
			if(_versionMap.containsKey(location)){
				Integer i = (Integer)_versionMap.get(location);
				version = i.intValue();
				_versionMap.put(location, new Integer(version+1));
			}else{
				_versionMap.put(location, new Integer(0));
			}
			
			return new SSADefinition(location, version, quad);
		}
	};
	
	protected SSADefinition(SSALocation location, int version, Quad quad){
		this._location = location;
		this._version  = version;	
		this._quad 	   = quad;
	}
	
	public SSALocation getLocation() {return _location;}
	public int getVersion(){return _version;}
	public Quad getQuad(){return _quad;}
	
	public String toString(){
		return _location.toString() + "_" + _version;
	}	
}

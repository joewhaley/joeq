package Compil3r.Analysis.IPSSA;
import java.util.HashMap;

import Compil3r.Analysis.IPA.SSALocation;
import Compil3r.Quad.Quad;

public class SSADefinition {
	protected SSALocation	_location;
	int 					_version;
	Quad				    _quad;
	long 					_id;		// this is an absolutely unique definition ID
	
	static class Helper {
		protected static HashMap/*<SSALocation, Integer>*/ _versionMap;
		protected static long globalID; 
		
		static SSADefinition create_ssa_definition(SSALocation location, Quad quad){
			int version = 0;
			if(_versionMap.containsKey(location)){
				Integer i = (Integer)_versionMap.get(location);
				version = i.intValue();
				_versionMap.put(location, new Integer(version+1));
			}else{
				_versionMap.put(location, new Integer(0));
			}
			
			return new SSADefinition(location, version, quad, globalID++);
		}
	};
	
	protected SSADefinition(SSALocation location, int version, Quad quad, long id){
		this._location = location;
		this._version  = version;	
		this._quad 	   = quad;
		this._id       = id;
	}
	
	public SSALocation getLocation() {return _location;}
	public int getVersion(){return _version;}
	public Quad getQuad(){return _quad;}
	
	public String toString(){
		return _location.toString() + "_" + _version;
	}

	public long getID() {
		return _id;
	}
}

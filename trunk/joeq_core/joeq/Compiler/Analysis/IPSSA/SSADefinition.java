package Compil3r.Analysis.IPSSA;

import java.util.HashMap;
import java.util.LinkedHashSet;
import Compil3r.Analysis.IPSSA.Utils.DefinitionSet;
import Compil3r.Quad.Quad;

public class SSADefinition {
	protected SSALocation	_location;
	int 					_version;
	Quad				    _quad;
	LinkedHashSet			_uses;
	long 					_id;		// this is an absolutely unique definition ID
	
	public static class Helper {
		protected static HashMap/*<SSALocation, Integer>*/ 			_versionMap = new HashMap();
		protected static long 										_globalID;
		protected static DefinitionSet 								_definitionCache = new DefinitionSet(); 
		
		static SSADefinition create_ssa_definition(SSALocation location, Quad quad){
			int version = 0;
			if(_versionMap.containsKey(location)){
				Integer i = (Integer)_versionMap.get(location);
				version = i.intValue();
				_versionMap.put(location, new Integer(version+1));
			}else{
				_versionMap.put(location, new Integer(1));
			}
			
			SSADefinition def = new SSADefinition(location, version, quad, _globalID++);
			_definitionCache.add(def);
			
			return def;
		}
		
		public static SSAIterator.DefinitionIterator getAllDefinitionIterator(){
			return _definitionCache.getDefinitionIterator();
		}
	};
	
	protected SSADefinition(SSALocation location, int version, Quad quad, long id){
		this._location = location;
		this._version  = version;	
		this._quad 	   = quad;
		this._id       = id;
		
		_uses = new LinkedHashSet();
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

	public SSAIterator.ValueIterator getUseIterator() {
		return new SSAIterator.ValueIterator(_uses.iterator());
	}

	public void appendUse(SSAValue value) {
		_uses.add(value);		
	}
}

package Compil3r.Analysis.IPSSA;

import java.util.HashMap;
import java.util.LinkedHashSet;

import Clazz.jq_Method;
import Compil3r.Analysis.IPSSA.Utils.DefinitionSet;
import Compil3r.Quad.Quad;

/**
 * This is a definition in the SSA sense, meaning it's unique. The meaning of the 
 * definition is affected by the underlying location.
 * 
 * @see Compil3r.Analysis.IPSSA.SSALocation  
 * */
public class SSADefinition {
	protected SSALocation	_location;
	int 					_version;
	LinkedHashSet			_uses;
	long 					_id;		// this is an absolutely unique definition ID
    jq_Method               _method;
    Quad                    _quad;
	
	public static class Helper {
		protected static HashMap/*<SSALocation, Integer>*/ 			_versionMap = new HashMap();
		protected static long 										_globalID;
        /** The set of all definitions in the program, can be quite huge */
		protected static DefinitionSet 								_definitionCache = new DefinitionSet(); 
		
		static SSADefinition create_ssa_definition(SSALocation location, Quad quad, jq_Method method){
			int version = 0;
			if(_versionMap.containsKey(location)){
				Integer i = (Integer)_versionMap.get(location);
				version = i.intValue();
				_versionMap.put(location, new Integer(version+1));
			}else{
				_versionMap.put(location, new Integer(1));
			}
			
			SSADefinition def = new SSADefinition(location, version, quad, method, _globalID++);
			_definitionCache.add(def);
			//System.err.println(_definitionCache.size() + " definitions now");
			return def;
		}
		
		public static SSAIterator.DefinitionIterator getAllDefinitionIterator(){
			return _definitionCache.getDefinitionIterator();
		}

        /**
         *  This is slow reverse lookup. Don't use this very often.
         * */
        public static SSADefinition lookupDefinition(String name) {
            //System.err.println("Searching through " + _definitionCache.size() + " definitions");
            for(SSAIterator.DefinitionIterator iter = getAllDefinitionIterator(); iter.hasNext(); ) {
                SSADefinition def = iter.nextDefinition();
                
                if(def.toString().equals(name)) {
                    // don't even check for duplicates                   
                    return def;
                }else {
                    //System.err.println("Skipping " + def);
                }
            }
            return null;
        }
	};
	
	private SSADefinition(SSALocation location, int version, Quad quad, jq_Method method, long id){
		this._location    = location;
		this._version     = version;	
		this._quad        = quad;
        this._method      = method;
		this._id          = id;        
		
		_uses = new LinkedHashSet();
	}
	
	public SSALocation getLocation() {return _location;}
	public int getVersion(){return _version;}
    public jq_Method getMethod() {return _method;}
    public Quad getQuad() { return _quad;}
	
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


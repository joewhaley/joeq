package Compil3r.Analysis.IPSSA;

import java.util.HashMap;

import Util.Assert;
import Clazz.jq_LocalVarTableEntry;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Compil3r.Analysis.IPA.PA;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.Quad;
import Compil3r.Quad.RegisterFactory;

/**
 * @author Vladimir Livshits
 */
public interface SSALocation {

    /**
     * 	We need to have "abstract" temporary locations for IPSSA construction purposes 
     * that do not necessarily correspond to anything tangible. 
     * */
    public static class Temporary implements SSALocation {
        private Temporary() {
            // there's no underlying node
        }

        // There's only one Temporary location -- use the FACTORY to retrieve it	
        public static class FACTORY {
            private static Temporary INSTANCE = new Temporary();
            
            public static Temporary get() {
                return INSTANCE;
            }
        }

		public String toString(PA pa) {
			return null;
		}
		
		public String toString() {
			return "temp";
		}
    }
    
	public static class Unique implements SSALocation {
		private static long _count = 0;
		private long _id;
		
		private Unique(long id) {
			this._id = id;
		}

		// There's only one Temporary location -- use the FACTORY to retrieve it	
		public static class FACTORY {
			public static Unique get() {
				return new Unique(_count++);
			}
		}

		public String toString(PA pa) {
			return null;
		}
		
		public String toString() {
			return "uniq" + _id;
		}
	}

	String toString(PA pa);
        
    /**
     * These locations represent local variables.
     * */
    class LocalLocation implements SSALocation {    
        private RegisterFactory.Register _reg;
        private String _name = null;

        public static class FACTORY {
            static HashMap _locationMap = new HashMap();
            public static LocalLocation createLocalLocation(RegisterFactory.Register reg){
                LocalLocation loc = (LocalLocation) _locationMap.get(reg); 
                if(loc == null){
                    loc = new LocalLocation(reg);
                    _locationMap.put(reg, loc);
                }
                return loc;
            }
        }
        public RegisterFactory.Register getRegister(){
            return this._reg;
        }
        private LocalLocation(RegisterFactory.Register reg){
            Assert._assert(reg != null);
            this._reg = reg;
        }    
        public String toString(PA pa) {
            return toString();
        }
    
        // Looking at jq_Method.getLocalVarTableEntry( line number, register number ) may provide better output
        public String toString() {
            return _reg.toString();
        }
        public String getName(jq_Method method, Quad quad) {
            if(_reg.isTemp()) {
                return null;
            }
            if(_name != null) {
                return _name;   // a bit of caching
            }
            Integer i = ((Integer)CodeCache.getBCMap(method).get(quad));
            if(i == null) {
                return null;
            }
            int bci1 = i.intValue();            
            jq_LocalVarTableEntry entry = method.getLocalVarTableEntry(bci1, _reg.getNumber());
            if(entry == null) {
                return null;
            }
            jq_NameAndDesc nd = entry.getNameAndDesc();
            if(nd == null) {
                return null;
            }
            
            _name = nd.getName().toString();
            return _name;
        }
    }
}


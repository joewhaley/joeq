package Compil3r.Analysis.IPSSA.Utils;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import Compil3r.Analysis.IPSSA.SSADefinition;

public class ReachabilityTrace {
    LinkedList _definitions;
    
    ReachabilityTrace(){
        _definitions = new LinkedList();
    }
    
    public static class Algorithms {    
        public static Collection collectReachabilityTraces(SSADefinition def1, SSADefinition def2) {
            Collection result = new LinkedList();
            // do DF rechability and fill in result
            ReachabilityTrace trace = new ReachabilityTrace();
            
            // TODO: add the algorithm...
            trace.appendDefinition(def1);
            trace.appendDefinition(def2);
            
            result.add(trace);
                    
            return result;
        }
    }
    
    void appendDefinition(SSADefinition def) {
        _definitions.addLast(def);
    }
    
    int size() {return _definitions.size();}
    
    public String toString() {
        StringBuffer result = new StringBuffer("[ ");
        for(Iterator iter = _definitions.iterator(); iter.hasNext(); ) {
            SSADefinition def = (SSADefinition)iter.next();
            result.append(def.toString());
            result.append(" ");
        }
        result.append("]");
     
        return result.toString();
    }
}


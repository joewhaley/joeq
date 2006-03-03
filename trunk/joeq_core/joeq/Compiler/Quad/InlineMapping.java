/*
 * Created on Feb 21, 2006
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package joeq.Compiler.Quad;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

public class InlineMapping {
    private static Map _map = new HashMap();
    public static HashMap fakeMap = new HashMap();

    public static Quad getOriginalQuad(Quad newQuad) {
        Quad oldQuad = null;
        do {
            oldQuad = newQuad;
            newQuad = (Quad) _map.get(newQuad);            
        } while(newQuad != null && newQuad != oldQuad);
        
        return oldQuad;
    }

    /**
     * Add to the list of quads to watch
     * */
    public static void add(Quad q) {
        _map.put(q, q);
    }
    
    public static void update() {
        Set toUpdate = new HashSet();
        for(Iterator iter = _map.entrySet().iterator(); iter.hasNext();) {
            Map.Entry e = (Entry) iter.next();
            Quad q = (Quad) e.getKey();
            Quad old_q = (Quad) e.getValue();
            
            Quad new_q = (Quad) ControlFlowGraph.correspondenceMap.get(q);
            if(new_q != null) {
                System.out.println("Updating " + q + "\n\tfrom " + old_q + "\n\tto " + new_q);
                toUpdate.add(q);
            }
        }
        
        for(Iterator iter = toUpdate.iterator(); iter.hasNext();) {
            Quad q = (Quad) iter.next();
            
            _map.put(q, ControlFlowGraph.correspondenceMap.get(q));
        }
    }

    public static Quad map(Quad callSite) {
        return (Quad) _map.get(callSite);
    }

    public static void rememberFake(Quad callQuad, Quad newQuad) {
        fakeMap.put(callQuad, newQuad);
        
    }
    
    public static void invalidate() {
        _map.clear();
        fakeMap.clear();
    }
}

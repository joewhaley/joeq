/*
 * Created on Feb 21, 2006
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package joeq.Compiler.Quad;

import java.util.HashMap;
import java.util.Map;

public class InlineMapping {
    private static Map _map = new HashMap();

    public static void add(Quad q, Quad newInvoke) {
        _map.put(q, newInvoke);        
    }
    
    public static Quad getOriginalQuad(Quad newQuad) {
        Quad oldQuad = null;
        do {
            newQuad = oldQuad;
            oldQuad = (Quad) _map.get(newQuad);            
        } while(oldQuad != null);
        
        return newQuad;
    }
}

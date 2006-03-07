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
import joeq.Class.jq_FakeInstanceMethod;
import joeq.Class.jq_FakeStaticMethod;
import joeq.Class.jq_Method;
import joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import joeq.UTF.Utf8;
import jwutil.collections.Pair;
import jwutil.util.Assert;

public class InlineMapping {
    private static Map _map = new HashMap();
    public static HashMap fakeMap = new HashMap();

//    public static Quad getOriginalQuad(Quad newQuad) {
//        Quad oldQuad = null;
//        do {
//            oldQuad = newQuad;
//            newQuad = (Quad) _map.get(newQuad);            
//        } while(newQuad != null && newQuad != oldQuad);
//        
//        return oldQuad;
//    }

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

    public static void rememberFake(Pair newPair, Pair callPair) {
        fakeMap.put(newPair, callPair);        
    }
    
    public static String getEmacsName(QuadProgramLocation pl) {
        Pair oldPair  = new Pair(pl.getMethod(), pl.getQuad()); 
        Pair callPair = (Pair) fakeMap.get(oldPair);
        if(callPair != null) {
            jq_Method method = unfake((jq_Method) callPair.left);
            //Quad quad = (Quad) callPair.right;
            Utf8 source = method.getDeclaringClass().getSourceFile();
            if (source != null) {
//                Map map = CodeCache.getBCMap(method);
//                if(map != null) {
//                    Integer i = (Integer) map.get(pl.getQuad());
//                    if(i != null) {
//                        int bc = i.intValue();
//                        return source + ":" + method.getLineNumber(bc);
//                    }
//                }
                return source + ":" + method.getName() + " (fake)";
            }
        }
        
        return pl.getEmacsName();
    }
    
    public static void invalidate() {
        _map.clear();
        fakeMap.clear();
    }
    
    public static Map fakeMethods = new HashMap();

    public static void addToFakeMethods(jq_Method m, jq_Method newMethod) {
        Assert._assert(m != null && newMethod != null);
        // fake -> unfake
        fakeMethods.put(newMethod, m);
    }
    
    /**
     * Converts a fake methods to the original one it came from.
     * */
    public static jq_Method unfake(jq_Method m) {
        if(m instanceof jq_FakeInstanceMethod || m instanceof jq_FakeStaticMethod) {
            return (jq_Method) fakeMethods.get(m);
        } else {
            return m;
        }
    }
}

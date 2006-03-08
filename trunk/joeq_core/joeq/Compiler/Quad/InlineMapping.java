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
import sun.security.krb5.internal.be;
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
    public static HashMap oldLocationMap = new HashMap();
    static HashMap/*<Quad, Quad>*/ newlyInserted = new HashMap();    

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

    public static void rememberFake(Pair callPair, Pair allocPair) {
        jq_Method allocMethod = (jq_Method) allocPair.left;
        Quad allocQuad = (Quad) allocPair.right;
        Integer i = (Integer) CodeCache.getBCMap(allocMethod).get(allocQuad);
        String source = allocMethod.getDeclaringClass().getSourceFile().toString(); 
        String oldLocation = source + ":" + (i != null ? allocMethod.getLineNumber(i.intValue()) : -1);
        // old -> new
        fakeMap.put(callPair, allocPair);
    }
    
    public static void saveOldLocation(Quad quad, String location) {
        oldLocationMap.put(quad, location);
    }
    
    public static String getOldLocation(Quad quad) {
        return (String) oldLocationMap.get(quad);
    }
    
    public static void invalidate() {
        _map.clear();
        fakeMap.clear();
        oldLocationMap.clear();
        newlyInserted.clear();
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

    public static void add(Quad oldQuad, Quad newQuad, jq_Method newMethod) {        
        newlyInserted.put(oldQuad, newQuad);
        jq_Method beingInlined = MethodInline.beingInlined;
        Assert._assert(beingInlined != null);
        String location = getEmacsName(beingInlined, oldQuad);
        
        oldLocationMap.put(oldQuad, location);
        oldLocationMap.put(newQuad, location);
    }
    
    public static String getEmacsName(jq_Method method, Quad q) {
        String source = method.getDeclaringClass().getSourceFile().toString();
        Integer i = (Integer) CodeCache.getBCMap(method).get(q);
        if(i != null) {
            return source + ":" + method.getLineNumber(i.intValue());
        } else {
            return source + ":" + "-1";
        }
    }

}

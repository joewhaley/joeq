/*
 * Created on 12.10.2004
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package joeq.Compiler.Analysis.FlowInsensitive;

import java.util.HashMap;
import java.util.Iterator;
import joeq.Class.jq_Class;
import joeq.Class.jq_Initializer;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import jwutil.util.Assert;

public class BogusSummaryProvider {
    HashMap classMap 		  = new HashMap();
    HashMap methodMap 		  = new HashMap();
    
    private static final boolean TRACE = !System.getProperty("pa.tracebogus").equals("no");
    private static jq_Class realString;
    private static jq_Class realStringBuffer;
    private static jq_Class realHashMap;
    private static jq_Class fakeString;
    private static jq_Class fakeStringBuffer;
    private static jq_Class fakeHashMap;    
    
    public BogusSummaryProvider() {
        realString       = getClassByName("java.lang.String");
        realStringBuffer = getClassByName("java.lang.StringBuffer");
        realHashMap      = getClassByName("java.util.HashMap");
        Assert._assert(realString != null && realStringBuffer != null && realHashMap != null);
        
        fakeString       = getClassByName("MyMockLib.MyString");
        fakeStringBuffer = getClassByName("MyMockLib.MyStringBuffer");        
        fakeHashMap      = getClassByName("MyMockLib.MyHashMap");               
        Assert._assert(fakeString != null && fakeStringBuffer != null && fakeHashMap != null);
        
        classMap.put(realString, fakeString);
        classMap.put(realStringBuffer, fakeStringBuffer);
        classMap.put(realHashMap, fakeHashMap);
    }

    /**
     * Caching method to return a replacement for @param m.
     * 
     * @return replacement for m.
     * */
    public jq_Method getReplacementMethod(jq_Method m, Integer offset) {
        jq_Method replacement = (jq_Method) methodMap.get(m);
        
        if(replacement == null) {
	        jq_Class c = (jq_Class) classMap.get(m.getDeclaringClass());
	        
	        if(c != null) {
	            replacement = findReplacementMethod(c, m, offset);
	            
	            if(replacement == null) {
	                if(TRACE) System.err.println("No replacement for " + m + " found in " + c);
	                return null;
	            }
	            methodMap.put(m, replacement);
	            if(TRACE) System.out.println("Replaced " + m + " with " + replacement);
	            return replacement;
	        } else {
	            return null;
	        }
        } else {
            return replacement;
        }
    }
    
    private static jq_Method findReplacementMethod(jq_Class clazz, jq_Method originalMethod, Integer offset) {
        for(Iterator iter = clazz.getMembers().iterator(); iter.hasNext();){
            Object o = iter.next();
            if(!(o instanceof jq_Method)) continue;
            jq_Method m = (jq_Method) o;
            
            if(!m.getName().toString().equals(originalMethod.getName().toString())){
                continue;
            }
            
            if(m.getParamTypes().length != originalMethod.getParamTypes().length){
                continue;            
            }
            
            boolean allMatch = true;
            int base = 0;
            if(clazz == fakeHashMap){
                base = 1;                
            }
            if(originalMethod instanceof jq_Initializer){                
                base = 1;
            }
            for(int i = base; i < originalMethod.getParamTypes().length; i++){
                jq_Type type = m.getParamTypes()[i];
                jq_Type originalType = originalMethod.getParamTypes()[i];
                
                if(type != originalType){
                    allMatch = false;
                    break;
                }
            }
            if(!allMatch) {
                continue;
            }
         
            // done with the tests: m is good
            return m;
        }
        
        return null;
    }
    
    private static jq_Class getClassByName(String className) {
        jq_Class theClass = (jq_Class)jq_Type.parseType(className);
        Assert._assert(theClass != null, className + " is not available.");
        theClass.prepare();
        
        return theClass;
    }

    public boolean hasStaticReplacement(jq_Method replacement) {
        jq_Class clazz = replacement.getDeclaringClass();
        
        return clazz == fakeString || clazz == fakeStringBuffer;
    }
}
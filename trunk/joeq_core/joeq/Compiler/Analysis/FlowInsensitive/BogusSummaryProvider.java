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
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import jwutil.util.Assert;

public class BogusSummaryProvider {
    HashMap classMap 		  = new HashMap();
    HashMap methodMap 		  = new HashMap();
    
    private static final boolean TRACE = true;    
    
    public BogusSummaryProvider() {
        jq_Class realString       = getClassByName("java.lang.String");
        jq_Class realStringBuffer = getClassByName("java.lang.StringBuffer");
        
        jq_Class fakeString       = getClassByName("MyMockLib.MyString");
        jq_Class fakeStringBuffer = getClassByName("MyMockLib.MyStringBuffer");
        
        
        classMap.put(realString, fakeString);
        classMap.put(realStringBuffer, fakeStringBuffer);
    }

    /**
     * Caching method to return a replacement for @param m.
     * 
     * @return replacement for m.
     * */
    public jq_Method getReplacementMethod(jq_Method m) {
        jq_Method replacement = (jq_Method) methodMap.get(m);
        
        if(replacement == null) {
	        jq_Class c = (jq_Class) classMap.get(m.getDeclaringClass());
	        
	        if(c != null) {
	            replacement = findReplacementMethod(c, m);
	            
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
    
    private static jq_Method findReplacementMethod(jq_Class clazz, jq_Method originalMethod) {
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
            for(int i = 0; i < originalMethod.getParamTypes().length; i++){
                if(m.getParamTypes()[i] != originalMethod.getParamTypes()[i]){
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
}

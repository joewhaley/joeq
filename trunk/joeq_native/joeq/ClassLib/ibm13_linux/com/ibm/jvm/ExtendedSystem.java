/*
 * ExtendedSystem.java
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.com.ibm.jvm;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Type;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Run_Time.Unsafe;
import Run_Time.Reflection;

public abstract class ExtendedSystem {
    
    private static boolean isJVMUnresettable(jq_Class clazz) { return false; }

    private static java.lang.Object resizeArray(jq_Class clazz, int newSize, java.lang.Object array, int startIndex, int size) {
	jq_Array a = (jq_Array)Unsafe.getTypeOf(array);
	java.lang.Object o = a.newInstance(newSize);
	java.lang.System.arraycopy(array, 0, o, startIndex, size);
	return o;
    }
    private static java.lang.Object newArray(jq_Class clazz, java.lang.Class elementType, int size, java.lang.Object enclosingObject) {
	jq_Type t = Reflection.getJQType(elementType);
	jq_Array a = t.getArrayTypeForElementType();
	return a.newInstance(size);
    }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Lcom/ibm/jvm/ExtendedSystem;");

}

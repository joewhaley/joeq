/*
 * CodeAddress.java
 *
 * Created on September 13, 2002, 12:06 AM
 *
 */
 
package Memory;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_StaticField;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class CodeAddress extends Address {

    public static CodeAddressFactory FACTORY;
    
    public abstract static class CodeAddressFactory {
        public abstract int size();
    }
    
    public static final int size() {
        return FACTORY.size();
    }

    public native Address peek();
    public native byte    peek1();
    public native short   peek2();
    public native int     peek4();
    public native long    peek8();
    
    public native void poke(Address v);
    public native void poke1(byte v);
    public native void poke2(short v);
    public native void poke4(int v);
    public native void poke8(long v);
    
    public native Address offset(int offset);
    public native Address align(int shift);
    public native int difference(Address v);
    public native boolean isNull();
    
    public native int to32BitValue();
    public native String stringRep();
    
    public static final jq_Class _class;
    public static final jq_StaticField _FACTORY;
    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LMemory/CodeAddress;");
        _FACTORY = _class.getOrCreateStaticField("FACTORY", "LMemory/CodeAddress$CodeAddressFactory;");
    }
}

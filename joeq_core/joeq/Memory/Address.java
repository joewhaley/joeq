/*
 * Address.java
 *
 * Created on September 13, 2002, 12:06 AM
 *
 */
 
package Memory;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;

/**
 * @author  John Whaley
 * @version $Id$
 */
public abstract class Address {

    public abstract Address peek();
    public abstract byte  peek1();
    public abstract short peek2();
    public abstract int   peek4();
    public abstract long  peek8();
    
    public abstract void poke(Address v);
    public abstract void poke1(byte v);
    public abstract void poke2(short v);
    public abstract void poke4(int v);
    public abstract void poke8(long v);
    
    public abstract Address offset(int offset);
    public abstract Address align(int shift);
    public abstract int difference(Address v);
    public abstract boolean isNull();
    
    public abstract int to32BitValue();
    public abstract String stringRep();
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LMemory/Address;");
    }
}

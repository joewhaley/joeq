/*
 * CodeAddress.java
 *
 * Created on September 13, 2002, 12:06 AM
 *
 */
 
package Bootstrap;

import Clazz.jq_Class;
import Memory.Address;
import Memory.CodeAddress;
import Util.Strings;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class BootstrapCodeAddress extends CodeAddress implements BootstrapAddress {

    public static BootstrapCodeAddressFactory FACTORY = new BootstrapCodeAddressFactory(BootstrapCodeAllocator.DEFAULT);
    
    public static class BootstrapCodeAddressFactory extends CodeAddressFactory {
        final BootstrapCodeAllocator bca;
        public BootstrapCodeAddressFactory(BootstrapCodeAllocator bca) {
            this.bca = bca;
        }
        public int size() { return 4; }
        public CodeAddress getNull() { return NULL; }
        public static final BootstrapCodeAddress NULL = new BootstrapCodeAddress(0);
    }
    
    public final int value;
    
    public BootstrapCodeAddress(int value) { this.value = value; }
    
    public Address peek() { return FACTORY.bca.peek(this); }
    public byte    peek1() { return FACTORY.bca.peek1(this); }
    public short   peek2() { return FACTORY.bca.peek2(this); }
    public int     peek4() { return FACTORY.bca.peek4(this); }
    public long    peek8() { return FACTORY.bca.peek8(this); }
    
    public void poke(Address v) { FACTORY.bca.poke(this, v); }
    public void poke1(byte v) { FACTORY.bca.poke1(this, v); }
    public void poke2(short v) { FACTORY.bca.poke2(this, v); }
    public void poke4(int v) { FACTORY.bca.poke4(this, v); }
    public void poke8(long v) { FACTORY.bca.poke8(this, v); }
    
    public Address offset(int offset) { return new BootstrapCodeAddress(value+offset); }
    public Address align(int shift) {
        int mask = (1 << shift) - 1;
        return new BootstrapCodeAddress((value+mask)&~mask);
    }
    public int difference(Address v) { return this.value - v.to32BitValue(); }
    public boolean isNull() { return value == 0; }
    
    public int to32BitValue() { return value; }
    public String stringRep() { return Strings.hex8(value); }
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LBootstrap/BootstrapCodeAddress;");
    }
}

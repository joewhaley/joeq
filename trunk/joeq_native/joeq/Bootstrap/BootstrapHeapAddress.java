/*
 * BootstrapHeapAddress.java
 *
 * Created on September 13, 2002, 12:06 AM
 *
 */
 
package Bootstrap;

import Clazz.jq_Class;
import Main.jq;
import Memory.Address;
import Memory.HeapAddress;
import Memory.HeapAddress.HeapAddressFactory;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class BootstrapHeapAddress extends HeapAddress implements BootstrapAddress {

    public static BootstrapHeapAddressFactory FACTORY = new BootstrapHeapAddressFactory(BootImage.DEFAULT);
    
    public static class BootstrapHeapAddressFactory extends HeapAddressFactory {
        BootImage bi;
        public BootstrapHeapAddressFactory(BootImage bi) {
            jq.Assert(bi != null);
            this.bi = bi;
        }
        public int size() { return 4; }
        public HeapAddress getNull() { return NULL; }
        public HeapAddress addressOf(Object o) {
            //if (o == null) return NULL;
            return bi.getOrAllocateObject(o);
        }
        public HeapAddress address32(int v) {
            return new BootstrapHeapAddress(v);
        }
        public static final BootstrapHeapAddress NULL = new BootstrapHeapAddress(0);
    }
    
    public final int value;
    
    BootstrapHeapAddress(int value) { this.value = value; }
    
    public Address peek() { jq.UNREACHABLE(); return null; }
    public byte    peek1() { jq.UNREACHABLE(); return 0; }
    public short   peek2() { jq.UNREACHABLE(); return 0; }
    public int     peek4() { jq.UNREACHABLE(); return 0; }
    public long    peek8() { jq.UNREACHABLE(); return 0; }
    
    public void poke(Address v) { jq.UNREACHABLE(); }
    public void poke1(byte v) { jq.UNREACHABLE(); }
    public void poke2(short v) { jq.UNREACHABLE(); }
    public void poke4(int v) { jq.UNREACHABLE(); }
    public void poke8(long v) { jq.UNREACHABLE(); }
    
    public Address offset(int offset) { return new BootstrapHeapAddress(value+offset); }
    public Address align(int shift) {
        int mask = (1 << shift) - 1;
        return new BootstrapHeapAddress((value+mask)&~mask);
    }
    public int difference(Address v) { return this.value - v.to32BitValue(); }
    public boolean isNull() { return value == 0; }
    
    public int to32BitValue() { return value; }
    public String stringRep() { return jq.hex8(value); }
    
    public static final jq_Class _class;
    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("LBootstrap/BootstrapHeapAddress;");
    }
}

/*
 * CodeAllocator.java
 *
 * Created on February 9, 2001, 8:40 AM
 *
 * @author  John Whaley
 * @version 
 */

package Allocator;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Clazz.jq_BytecodeMap;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;

import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

public abstract class CodeAllocator {
    
    public static /*final*/ boolean TRACE = false;
    
    public static /*final*/ CodeAllocator DEFAULT;
    
    public abstract void init();
    public abstract x86CodeBuffer getCodeBuffer(int estimated_size);
    public abstract void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap);
    public abstract void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target);

    public abstract static class x86CodeBuffer {

        public abstract int getCurrentOffset();
        public abstract int/*CodeAddress*/ getCurrentAddress();

        public abstract void add1(byte i);
        public abstract void add2_endian(int i);
        public abstract void add2(int i);
        public abstract void add3(int i);
        public abstract void add4_endian(int i);

        public abstract byte get1(int k);
        public abstract int get4_endian(int k);

        public abstract void put1(int k, byte instr);
        public abstract void put4_endian(int k, int instr);
        
        public abstract jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex,
                                                          jq_BytecodeMap bcm, ExceptionDeliverer exd,
                                                          List code_relocs, List data_relocs);
    }
    
    private static final SortedMap compiled_methods;
    private static int/*CodeAddress*/ lowAddress = Integer.MAX_VALUE, highAddress = 0;
    static {
        compiled_methods = new TreeMap();
        jq_CompiledCode cc = new jq_CompiledCode(null, 0, 0, null, null, null, null, null);
        compiled_methods.put(cc, cc);
    }
    
    public static void registerCode(jq_CompiledCode cc) {
        if (TRACE) System.out.println("Registering code: "+cc);
        lowAddress = Math.min(cc.getEntrypoint(), lowAddress);
        highAddress = Math.max(cc.getEntrypoint()+cc.getLength(), highAddress);
        compiled_methods.put(cc, cc);
    }
    public static jq_CompiledCode getCodeContaining(int/*CodeAddress*/ ip) {
        return (jq_CompiledCode)compiled_methods.get(new InstructionPointer(ip));
    }
    public static int/*CodeAddress*/ getLowAddress() { return lowAddress; }
    public static int/*CodeAddress*/ getHighAddress() { return highAddress; }
    
    // NOTE: doesn't implement hashCode() very well!
    public static class InstructionPointer implements Comparable {
        private final int/*CodeAddress*/ ip;
        public InstructionPointer(int/*CodeAddress*/ ip) { this.ip = ip; }
        public int/*CodeAddress*/ getIP() { return ip; }
        public int compareTo(jq_CompiledCode that) {
            return -that.compareTo(this);
        }
        public int compareTo(InstructionPointer that) {
            if (this.ip < that.ip) return -1;
            if (this.ip > that.ip) return 1;
            return 0;
        }
        public int compareTo(Object that) {
            if (that instanceof jq_CompiledCode)
                return compareTo((jq_CompiledCode)that);
            else
                return compareTo((InstructionPointer)that);
        }
        public boolean equals(InstructionPointer that) {
            return this.ip == that.ip;
        }
        public boolean equals(jq_CompiledCode that) {
            return that.equals(this);
        }
        public boolean equals(Object that) {
            if (that instanceof jq_CompiledCode)
                return equals((jq_CompiledCode)that);
            else
                return equals((InstructionPointer)that);
        }
        public int hashCode() { return 0; }
        
        public static final jq_InstanceField _ip;
        static {
            jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/CodeAllocator$InstructionPointer;");
            _ip = k.getOrCreateInstanceField("ip", "I");
        }
    }
    
    public static final Set codeAddressFields;
    public static final jq_StaticField _compiled_methods;
    public static final jq_StaticField _DEFAULT;
    public static final jq_StaticField _lowAddress;
    public static final jq_StaticField _highAddress;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/CodeAllocator;");
        _compiled_methods = k.getOrCreateStaticField("compiled_methods", "Ljava/util/SortedMap;");
        _DEFAULT = k.getOrCreateStaticField("DEFAULT", "LAllocator/CodeAllocator;");
        _lowAddress = k.getOrCreateStaticField("lowAddress", "I");
        _highAddress = k.getOrCreateStaticField("highAddress", "I");
        codeAddressFields = new HashSet();
        codeAddressFields.add(Assembler.x86.Code2HeapReference._from_codeloc);
        codeAddressFields.add(Assembler.x86.Heap2CodeReference._to_codeloc);
        codeAddressFields.add(Assembler.x86.DirectBindCall._source);
        codeAddressFields.add(Clazz.jq_CompiledCode._entrypoint);
        codeAddressFields.add(InstructionPointer._ip);
        codeAddressFields.add(_lowAddress);
        codeAddressFields.add(_highAddress);
    }
}

/*
 * CodeAllocator.java
 *
 * Created on February 9, 2001, 8:40 AM
 *
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

import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * This class provides the abstract interface for code allocators.  A code allocator
 * handles the allocation and management of code buffers.
 *
 * It also provides static methods for keeping track of the compiled methods and
 * their address ranges.
 * 
 * It also includes an inner class that provides the interface for code buffers.
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class CodeAllocator {
    
    /** Trace flag. */
    public static /*final*/ boolean TRACE = false;
    
    /** Initialize this code allocator.  This method is always called before the
     *  code allocator is actually used.
     */
    public abstract void init();
    
    /** Allocate a code buffer of the given estimated size.
     * It is legal for code to exceed the estimated size, but the cost may be high
     * (i.e. it may require recopying of the buffer.)
     *
     * @param estimated_size  estimated size, in bytes, of desired code buffer
     * @return  the new code buffer
     */
    public abstract x86CodeBuffer getCodeBuffer(int estimated_size);
    
    /** Patch the given code address to refer to the given heap address, in absolute terms.
     *  This is used to patch heap address references in the code.
     *
     * @param code  code address to patch
     * @param heap  heap address to patch to
     */
    public abstract void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap);
    
    /** Patch the given code address to refer to the given code address, in relative terms.
     *  This is used to patch branch targets in the code.
     *
     * @param code  code address to patch
     * @param target  code address to patch to
     */
    public abstract void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target);

    /** This class provides the interface for x86 code buffers.
     *  These code buffers are used to store generated x86 code.
     *  After the code is generated, use the allocateCodeBlock method to obtain a jq_CompiledCode object.
     */
    public abstract static class x86CodeBuffer {

        /** Returns the current offset in this code buffer.
         * @return  current offset
         */
        public abstract int getCurrentOffset();
        
        /** Returns the current address in this code buffer.
         * @return  current address
         */
        public abstract int/*CodeAddress*/ getCurrentAddress();

        /** Adds one byte to the end of this code buffer.  Offset/address increase by 1.
         * @param i  the byte to add
         */
        public abstract void add1(byte i);
        
        /** Adds two bytes (little-endian) to the end of this code buffer.  Offset/address increase by 2.
         * @param i  the little-endian value to add
         */
        public abstract void add2_endian(int i);
        
        /** Adds two bytes (big-endian) to the end of this code buffer.  Offset/address increase by 2.
         * @param i  the big-endian value to add
         */
        public abstract void add2(int i);
        
        /** Adds three bytes (big-endian) to the end of this code buffer.  Offset/address increase by 3.
         * @param i  the big-endian value to add
         */
        public abstract void add3(int i);
        
        /** Adds four bytes (little-endian) to the end of this code buffer.  Offset/address increase by 4.
         * @param i  the little-endian value to add
         */
        public abstract void add4_endian(int i);
        
        /** Gets the byte at the given offset in this code buffer.
         * @param k  offset of byte to return
         * @return  byte at given offset
         */
        public abstract byte get1(int k);
        
        /** Gets the (little-endian) 4 bytes at the given offset in this code buffer.
         * @param k  offset of little-endian 4 bytes to return
         * @return  little-endian 4 bytes at given offset
         */
        public abstract int get4_endian(int k);

        /** Sets the byte at the given offset to the given value.
         * @param k  offset of byte to set
         * @param instr  value to set it to
         */
        public abstract void put1(int k, byte instr);
        
        /** Sets the 4 bytes at the given offset to the given (little-endian) value.
         * @param k  offset of 4 bytes to set
         * @param instr  little-endian value to set it to
         */
        public abstract void put4_endian(int k, int instr);
        
        /** Uses the code in this buffer, along with the arguments, to create a jq_CompiledCode object.
         *  Call this method after you are done generating code, and actually want to use it.
         * 
         * @param m  Java method of this code block, or null if none
         * @param ex  exception handler table, or null if none
         * @param bcm  bytecode map, or null if none
         * @param exd  exception deliverer to use for this code, or null if none
         * @param code_relocs  list of code relocations for this code buffer, or null if none
         * @param data_relocs  list of data relocations for this code buffer, or null if none
         * @return  a new jq_CompiledCode object for the code
         */
        public abstract jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex,
                                                          jq_BytecodeMap bcm, ExceptionDeliverer exd,
                                                          List code_relocs, List data_relocs);
    }
    
    /** Map of compiled methods, sorted by address. */
    private static final SortedMap compiled_methods;
    
    /** Address range of compiled code.  Code outside of this range cannot be generated by us. */
    private static int/*CodeAddress*/ lowAddress = Integer.MAX_VALUE, highAddress = 0;
    static {
        compiled_methods = new TreeMap();
        jq_CompiledCode cc = new jq_CompiledCode(null, 0, 0, null, null, null, null, null);
        compiled_methods.put(cc, cc);
    }
    
    /** Register the given compiled code, so lookups by address will return this code.
     *
     * @param cc  compiled code to register
     */
    public static void registerCode(jq_CompiledCode cc) {
        if (TRACE) System.out.println("Registering code: "+cc);
        lowAddress = Math.min(cc.getEntrypoint(), lowAddress);
        highAddress = Math.max(cc.getEntrypoint()+cc.getLength(), highAddress);
        compiled_methods.put(cc, cc);
    }
    
    /** Return the compiled code which contains the given code address.
     *  Returns null if there is no registered code that contains the given address.
     *
     * @param ip  code address to check
     * @return  compiled code containing given address, or null
     */
    public static jq_CompiledCode getCodeContaining(int/*CodeAddress*/ ip) {
        return (jq_CompiledCode)compiled_methods.get(new InstructionPointer(ip));
    }
    
    /** Returns the lowest address of any registered code.
     * @return  lowest address of any registered code.
     */
    public static int/*CodeAddress*/ getLowAddress() { return lowAddress; }
    /** Returns the highest address of any registered code.
     * @return  highest address of any registered code.
     */
    public static int/*CodeAddress*/ getHighAddress() { return highAddress; }

    /** Returns an iterator of the registered jq_CompiledCode objects, in address order.
     * @return  iterator of jq_CompiledCode objects
     */
    public static Iterator/*<jq_CompiledCode>*/ getCompiledMethods() {
        Iterator i = compiled_methods.keySet().iterator();
        i.next(); // skip bogus compiled code
        return i;
    }
    
    /** Returns the number of registered jq_CompiledCode objects.
     * @return  number of registered jq_CompiledCode objects
     */
    public static int getNumberOfCompiledMethods() {
        return compiled_methods.keySet().size()-1;  // skip bogus compiled code
    }
    
    /** An object of this class represents a code address.
     *  It can be compared with a jq_CompiledCode object with compareTo and equals.
     *  They are equal if the InstructionPointer points within the range of the compiled code;
     *  the InstructionPointer is less if it is before the start address of the compiled code;
     *  the InstructionPointer is less if it is after the end address of the compiled code.
     */
    public static class InstructionPointer implements Comparable {
        
        /** The (actual) address. */
        private final int/*CodeAddress*/ ip;
        
        /** Create a new instruction pointer.
         * @param ip  instruction pointer value
         */
        public InstructionPointer(int/*CodeAddress*/ ip) { this.ip = ip; }
        
        /** Extract the address of this instruction pointer.
         * @return  address of this instruction pointer
         */
        public int/*CodeAddress*/ getIP() { return ip; }
        
        /** Compare this instruction pointer to a compiled code object.
         * @param that  compiled code to compare against
         * @return  -1 if this ip comes before the given code, 0 if it is inside the given code, 1 if it is after the given code
         */
        public int compareTo(jq_CompiledCode that) {
            return -that.compareTo(this);
        }
        
        /** Compare this instruction pointer to another instruction pointer.
         * @param that  instruction pointer to compare against
         * @return  -1 if this ip is before the given ip, 0 if it is equal to the given ip, 1 if it is after the given ip
         */
        public int compareTo(InstructionPointer that) {
            if (this.ip < that.ip) return -1;
            if (this.ip > that.ip) return 1;
            return 0;
        }
        
        /** Compares this instruction pointer to the given object (InstructionPointer or jq_CompiledCode)
         * @param that  object to compare to
         * @return  -1 if this is less than, 0 if this is equal, 1 if this is greater than
         */
        public int compareTo(java.lang.Object that) {
            if (that instanceof jq_CompiledCode)
                return compareTo((jq_CompiledCode)that);
            else
                return compareTo((InstructionPointer)that);
        }
        
        /** Returns true if this instruction pointer refers to a location within the given compiled code,
         *  false otherwise.
         * @param that  compiled code to compare to
         * @return  true if the instruction pointer is within, false otherwise
         */
        public boolean equals(jq_CompiledCode that) {
            return that.equals(this);
        }
        
        /** Returns true if this instruction pointer refers to the same location as the given instruction pointer,
         *  false otherwise.
         * @param that  instruction pointer to compare to
         * @return  true if the instruction pointers are equal, false otherwise
         */
        public boolean equals(InstructionPointer that) {
            return this.ip == that.ip;
        }
        
        /** Compares this instruction pointer with the given object (InstructionPointer or jq_CompiledCode).
         * @param that  object to compare with
         * @return  true if these objects are equal, false otherwise
         */
        public boolean equals(Object that) {
            if (that instanceof jq_CompiledCode)
                return equals((jq_CompiledCode)that);
            else
                return equals((InstructionPointer)that);
        }
        
        /**  Returns the hash code of this instruction pointer.  We use the pointer value itself.
         *   NOTE that this cannot be used when comparing with jq_CompiledCode objects.
         * @return  hash code
         */
        public int hashCode() { return ip; }
        
        public static final jq_InstanceField _ip;
        static {
            jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/CodeAllocator$InstructionPointer;");
            _ip = k.getOrCreateInstanceField("ip", "I");
        }
    }
    
    public static final Set codeAddressFields;
    public static final jq_StaticField _compiled_methods;
    public static final jq_StaticField _lowAddress;
    public static final jq_StaticField _highAddress;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/CodeAllocator;");
        _compiled_methods = k.getOrCreateStaticField("compiled_methods", "Ljava/util/SortedMap;");
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

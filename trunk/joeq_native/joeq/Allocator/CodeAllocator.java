/*
 * CodeAllocator.java
 *
 * Created on January 11, 2001, 10:55 AM
 *
 * @author  jwhaley
 * @version 
 */

package Allocator;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_Class;
import Clazz.jq_StaticField;
import Clazz.jq_TryCatch;
import Clazz.jq_BytecodeMap;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;
import Run_Time.Unsafe;
import jq;

import java.util.*;

public abstract class CodeAllocator {

    public static /*final*/ boolean TRACE = false;
    
    private static final int LOWER_THRESHOLD = 0x00500000;
    
    private static SortedMap compiled_methods = new TreeMap();
    static {
        jq_CompiledCode cc = new jq_CompiledCode(null, 0, 0, null, null, null);
        compiled_methods.put(cc, cc);
    }
    
    public static jq_CompiledCode allocateCodeBlock(jq_Method m, x86CodeBuffer cb,
                                                    jq_TryCatch[] ex, jq_BytecodeMap bcm,
                                                    ExceptionDeliverer exd) {
        int total = cb.getCurrentOffset();
        byte[] instr = new byte[total];
        int instr_ptr = 0;
        for (int i=0; i<cb.bundle_idx; ++i) {
            byte[] bundle = (byte[]) cb.bundles.elementAt(i);
            System.arraycopy(bundle, 0, instr, instr_ptr, cb.bundle_size);
            instr_ptr += cb.bundle_size;
        }
        System.arraycopy(cb.current_bundle, 0, instr, instr_ptr, cb.idx+1);
        jq_CompiledCode cc = new jq_CompiledCode(m, Unsafe.addressOf(instr), total, ex, bcm, exd);
        jq.assert(Unsafe.addressOf(instr) > LOWER_THRESHOLD, jq.hex8(Unsafe.addressOf(instr)));
        compiled_methods.put(cc, cc);
        if (TRACE) System.out.println(cc.toString());
        return cc;
    }
    
    public static int getStartAddress() { return LOWER_THRESHOLD; }
    
    public static jq_CompiledCode getCodeContaining(int ip) {
        return (jq_CompiledCode)compiled_methods.get(new InstructionPointer(ip));
    }
    
    // NOTE: doesn't implement hashCode() very well!
    public static class InstructionPointer implements Comparable {
        private final int/*Address*/ ip;
        public InstructionPointer(int ip) { this.ip = ip; }
        public int getIP() { return ip; }
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
    }
    
    public static class x86CodeBuffer {

        public int getCurrentOffset() {
            return bundle_size*bundle_idx+idx+1;
        }

        public void checkSize() {
            if (idx == bundle_size-1) {
                bundles.addElement(current_bundle = new byte[bundle_size]);
                ++bundle_idx; idx=-1;
            }
        }

        public void add1(byte i) {
            checkSize(); current_bundle[++idx] = i;
        }
        public void add2_endian(int i) {
            checkSize(); current_bundle[++idx] = (byte)(i);
            checkSize(); current_bundle[++idx] = (byte)(i >> 8);
        }
        public void add2(int i) {
            checkSize(); current_bundle[++idx] = (byte)(i >> 8);
            checkSize(); current_bundle[++idx] = (byte)(i);
        }
        public void add3(int i) {
            checkSize(); current_bundle[++idx] = (byte)(i >> 16);
            checkSize(); current_bundle[++idx] = (byte)(i >> 8);
            checkSize(); current_bundle[++idx] = (byte)(i);
        }
        public void add4_endian(int i) {
            checkSize(); current_bundle[++idx] = (byte)(i);
            checkSize(); current_bundle[++idx] = (byte)(i >> 8);
            checkSize(); current_bundle[++idx] = (byte)(i >> 16);
            checkSize(); current_bundle[++idx] = (byte)(i >> 24);
        }

        public byte get(int k) {
            int i = k >> bundle_shift;
            int j = k & bundle_mask;
            byte[] b = (byte[])bundles.elementAt(i);
            return b[j];
        }

        public int get4_endian (int k) {
            int b = (int)get(k) & 0x000000FF;
            b |= ((int)get(k+1) << 8) & 0x0000FF00;
            b |= ((int)get(k+2) << 16) & 0x00FF0000;
            b |= ((int)get(k+3) << 24) & 0xFF000000;
            return b;
        }

        public void put1(int k, byte instr) {
            int i = k >> bundle_shift;
            int j = k & bundle_mask;
            byte[] b = (byte[])bundles.elementAt(i);
            b[j] = instr;
        }

        public void put4_endian(int k, int instr) {
            put1(k, (byte)(instr));
            put1(k+1, (byte)(instr>>8));
            put1(k+2, (byte)(instr>>16));
            put1(k+3, (byte)(instr>>24));
        }

        private static final int bundle_shift = 12;
        private static final int bundle_size = 1 << bundle_shift;
        private static final int bundle_mask = bundle_size-1;
        private Vector bundles;
        private byte[] current_bundle;
        private int bundle_idx;
        private int idx;

        public x86CodeBuffer() {
            bundles = new Vector();
            bundles.addElement(current_bundle = new byte[bundle_size]);
            bundle_idx = 0;
            idx = -1;
        }
    }
    
    public static final jq_StaticField _compiled_methods;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/CodeAllocator;");
        _compiled_methods = k.getOrCreateStaticField("compiled_methods", "Ljava/util/SortedSet;");
    }
}

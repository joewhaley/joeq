/*
 * BootstrapCodeAllocator.java
 *
 * Created on February 8, 2001, 11:23 AM
 *
 * @author  John Whaley
 * @version 
 */

package Bootstrap;

import Allocator.CodeAllocator;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Clazz.jq_BytecodeMap;
import Clazz.jq_CompiledCode;
import Run_Time.ExceptionDeliverer;
import Run_Time.Unsafe;
import jq;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.io.OutputStream;
import java.io.IOException;

public class BootstrapCodeAllocator extends CodeAllocator {

    /** Creates new BootstrapCodeAllocator */
    public BootstrapCodeAllocator() {}

    private static final int bundle_shift = 12;
    private static final int bundle_size = 1 << bundle_shift;
    private static final int bundle_mask = bundle_size-1;
    private Vector bundles;
    private byte[] current_bundle;
    private int bundle_idx;
    private int idx;
    
    private List all_code_relocs, all_data_relocs;
    
    public void init() {
        bundles = new Vector();
        bundles.addElement(current_bundle = new byte[bundle_size]);
        bundle_idx = 0;
        idx = -1;
        all_code_relocs = new LinkedList();
        all_data_relocs = new LinkedList();
    }
    
    public x86CodeBuffer getCodeBuffer(int estimated_size) {
        return new Bootstrapx86CodeBuffer();
    }
    
    public int size() { return bundle_size*bundle_idx+idx+1; }

    public List getAllCodeRelocs() { return all_code_relocs; }
    public List getAllDataRelocs() { return all_data_relocs; }

    public void dump(OutputStream out)
    throws IOException {
        for (int i=0; i<bundle_idx; ++i) {
            byte[] bundle = (byte[]) bundles.elementAt(i);
            out.write(bundle);
        }
        out.write(current_bundle, 0, idx+1);
    }
    
    public void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap) {
        poke4(code, heap);
    }
    
    public void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target) {
        poke4(code, target-code-4);
    }
    
    public void poke1(int/*CodeAddress*/ k, byte v) {
        int i = k >> bundle_shift;
        int j = k & bundle_mask;
        byte[] b = (byte[])bundles.elementAt(i);
        b[j] = v;
    }
    
    public void poke4(int/*CodeAddress*/ k, int v) {
        poke1(k, (byte)(v));
        poke1(k+1, (byte)(v>>8));
        poke1(k+2, (byte)(v>>16));
        poke1(k+3, (byte)(v>>24));
    }
    
    public byte peek1(int/*CodeAddress*/ k) {
        int i = k >> bundle_shift;
        int j = k & bundle_mask;
        byte[] b = (byte[])bundles.elementAt(i);
        return b[j];
    }
    
    public int peek4(int/*CodeAddress*/ k) {
        return jq.fourBytesToInt(peek1(k), peek1(k+1), peek1(k+2), peek1(k+3));
    }
    
    public class Bootstrapx86CodeBuffer extends CodeAllocator.x86CodeBuffer {

        private int startIndex;
        
        Bootstrapx86CodeBuffer() {
            startIndex = size();
        }
        
        public int getStartIndex() { return startIndex; }
        
        public int getCurrentOffset() { return size()-startIndex; }
        public int getCurrentAddress() { return size(); }

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

        public byte get1(int k) {
            k += startIndex;
            int i = k >> bundle_shift;
            int j = k & bundle_mask;
            byte[] b = (byte[])bundles.elementAt(i);
            return b[j];
        }

        public int get4_endian (int k) {
            int b = (int)get1(k) & 0x000000FF;
            b |= ((int)get1(k+1) << 8) & 0x0000FF00;
            b |= ((int)get1(k+2) << 16) & 0x00FF0000;
            b |= ((int)get1(k+3) << 24) & 0xFF000000;
            return b;
        }

        public void put1(int k, byte instr) {
            k += startIndex;
            poke1(k, instr);
        }

        public void put4_endian(int k, int instr) {
            put1(k, (byte)(instr));
            put1(k+1, (byte)(instr>>8));
            put1(k+2, (byte)(instr>>16));
            put1(k+3, (byte)(instr>>24));
        }

        public jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex, jq_BytecodeMap bcm,
                                                 ExceptionDeliverer exd, List code_relocs, List data_relocs) {
            int total = getCurrentOffset();
            int start = getStartIndex();
            if (code_relocs != null)
                all_code_relocs.addAll(code_relocs);
            if (data_relocs != null)
                all_data_relocs.addAll(data_relocs);
            jq_CompiledCode cc = new jq_CompiledCode(m, start, total, ex, bcm, exd, code_relocs, data_relocs);
            CodeAllocator.registerCode(cc);
            return cc;
        }
    
    }
    
}

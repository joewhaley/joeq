/*
 * BootstrapCodeAllocator.java
 *
 * Created on February 8, 2001, 11:23 AM
 *
 */

package Bootstrap;

import java.io.IOException;
import java.io.OutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import Allocator.CodeAllocator;
import Clazz.jq_BytecodeMap;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Main.jq;
import Run_Time.ExceptionDeliverer;

/*
 * @author  John Whaley
 * @version $Id$
 */
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
    
    /** Allocate a code buffer of the given estimated size, such that the given
     * offset will have the given alignment.
     * In this allocator, the memory is allocated in chunks, so exceeding the
     * estimated size doesn't cost extra.
     *
     * @param estimatedSize  estimated size, in bytes, of desired code buffer
     * @param offset  desired offset to align to
     * @param alignment  desired alignment, or 0 if don't care
     * @return  the new code buffer
     */
    public x86CodeBuffer getCodeBuffer(int estimatedSize,
                                       int offset,
                                       int alignment) {
        // align pointer first
        int entrypoint = size() + offset;
        if (alignment > 1) {
        	entrypoint += alignment-1;
	        entrypoint &= ~(alignment-1);
        }
        idx += (entrypoint-offset) - size();
        return new Bootstrapx86CodeBuffer(entrypoint-offset, estimatedSize);
    }
    
    public int size() { return bundle_size*bundle_idx+idx+1; }
    public void ensureCapacity(int size) {
        int i = size >> bundle_shift;
        while (i+1 >= bundles.size()) {
            bundles.addElement(new byte[bundle_size]);
        }
    }

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
        private int entryIndex;
        
        Bootstrapx86CodeBuffer(int startIndex, int estSize) {
            this.startIndex = startIndex;
            ensureCapacity(startIndex+estSize);
        }
        
        public int getStartIndex() { return startIndex; }
        public int getEntryIndex() { return entryIndex; }
        
        public int getCurrentOffset() { return size()-startIndex; }
        public int getCurrentAddress() { return size(); }
        
        public void setEntrypoint() { entryIndex = size(); }

        public void checkSize() {
            if (idx >= bundle_size-1) {
                if (bundle_idx < bundles.size()-1) {
                    if (TRACE) System.out.println("getting next bundle idx "+(bundle_idx+1));
                    current_bundle = (byte[])bundles.get(++bundle_idx);
                    idx -= bundle_size;
                } else {
                    if (TRACE) System.out.println("allocing new bundle idx "+(bundle_idx+1));
                    bundles.addElement(current_bundle = new byte[bundle_size]);
                    ++bundle_idx; idx -= bundle_size;
                }
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
            return peek1(k);
        }

        public int get4_endian (int k) {
            k += startIndex;
            return peek4(k);
        }

        public void put1(int k, byte instr) {
            k += startIndex;
            poke1(k, instr);
        }

        public void put4_endian(int k, int instr) {
            k += startIndex;
            poke4(k, instr);
        }

        public void skip(int nbytes) {
        	jq.Assert(nbytes < bundle_size);
        	idx += nbytes;
        	//checkSize();
        }

        public jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex, jq_BytecodeMap bcm,
                                                 ExceptionDeliverer exd, List code_relocs, List data_relocs) {
            int total = getCurrentOffset();
            int start = getStartIndex();
            int entry = getEntryIndex();
            if (code_relocs != null)
                all_code_relocs.addAll(code_relocs);
            if (data_relocs != null)
                all_data_relocs.addAll(data_relocs);
            jq_CompiledCode cc = new jq_CompiledCode(m, start, total, entry, ex, bcm, exd, code_relocs, data_relocs);
            CodeAllocator.registerCode(cc);
            return cc;
        }
    
    }
    
}

/*
 * RuntimeCodeAllocator.java
 *
 * Created on January 11, 2001, 10:55 AM
 *
 */

package Allocator;

import java.util.List;

import Clazz.jq_BytecodeMap;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_TryCatch;
import Main.jq;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class RuntimeCodeAllocator extends CodeAllocator {

	// Memory layout:
	//
	// start:   |   next ptr   |---->
	//          |   end ptr    |
	//          |.....data.....|
	// current: |.....free.....|
	// end:     | current ptr  |

    /** Size of blocks allocated from the OS.
     */
    public static final int BLOCK_SIZE = 131072;
    
    /** Pointers to the start, current, and end of the heap.
     */
    private int/*CodeAddress*/ heapStart, heapCurrent, heapEnd;
    
    /** Pointer to the first block.
     */
    private int/*CodeAddress*/ heapFirst;
    
    /** Max memory free in all allocated blocks.
     */
    private int maxFreePrevious;
    
    boolean isGenerating = false;
    
    public void init()
    throws OutOfMemoryError {
        if (0 == (heapStart = heapFirst = SystemInterface.syscalloc(BLOCK_SIZE)))
            HeapAllocator.outOfMemory();
        Unsafe.poke4(heapStart, 0);
        Unsafe.poke4(heapStart + 4, heapEnd = heapStart + BLOCK_SIZE - 4);
        Unsafe.poke4(heapEnd, heapCurrent = heapStart + 8);
        if (TRACE) SystemInterface.debugmsg("Initialized run-time code allocator, start="+jq.hex8(heapCurrent)+" end="+jq.hex8(heapEnd));
    }
    
    /** Allocate a code buffer of the given estimated size, such that the given
     * offset will have the given alignment.
     * It is legal for code to exceed the estimated size, but the cost may be
     * high (i.e. it may require recopying of the buffer.)
     *
     * @param estimatedSize  estimated size, in bytes, of desired code buffer
     * @param offset  desired offset to align to
     * @param alignment  desired alignment, or 0 if don't care
     * @return  the new code buffer
     */
    public x86CodeBuffer getCodeBuffer(int estimatedSize,
                                       int offset,
                                       int alignment) {
        // should not be called recursively.
        jq.Assert(!isGenerating); isGenerating = true;
        if (TRACE) SystemInterface.debugmsg("Code generation started: "+this);
        // align pointer
        int entrypoint = heapCurrent+offset;
        if (alignment > 1) {
        	entrypoint += alignment-1;
	        entrypoint &= ~(alignment-1);
        }
        if (entrypoint + estimatedSize - offset <= heapEnd) {
            if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimatedSize)+" fits within free space in current block "+jq.hex8(heapCurrent)+"-"+jq.hex8(heapEnd));
            return new Runtimex86CodeBuffer(entrypoint-offset, heapEnd);
        }
        if (estimatedSize < maxFreePrevious) {
            // use a prior block's unused space.
            if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimatedSize)+" fits within a prior block: maxfreeprev="+jq.hex(maxFreePrevious));
            // start searching at the first block
            int start_ptr = heapFirst;
            for (;;) {
                jq.Assert(start_ptr != 0);                   // points to start of current block
                int end_ptr     = Unsafe.peek(start_ptr+4);  // points to end of current block
                int current_ptr = Unsafe.peek(end_ptr);      // current pointer for current block
                if (TRACE) SystemInterface.debugmsg("Checking block "+jq.hex8(start_ptr)+"-"+jq.hex8(end_ptr)+", current ptr="+jq.hex8(current_ptr));
                if ((end_ptr-current_ptr) >= estimatedSize) {
                    if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimatedSize)+") fits within free space "+jq.hex8(current_ptr)+"-"+jq.hex8(end_ptr));
                    return new Runtimex86CodeBuffer(current_ptr, end_ptr);
                }
                start_ptr = Unsafe.peek(start_ptr); // go to the next block
                if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimatedSize)+") doesn't fit, trying next block "+jq.hex8(start_ptr));
            }
        }
        if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimatedSize)+" is too large for current block "+jq.hex8(heapCurrent)+"-"+jq.hex8(heapEnd));
        // allocate new block.
        allocateNewBlock(Math.max(estimatedSize, BLOCK_SIZE));
        return new Runtimex86CodeBuffer(heapCurrent, heapEnd);
    }
    
    private void allocateNewBlock(int blockSize)
    throws OutOfMemoryError {
        if (TRACE) SystemInterface.debugmsg("Allocating new code block (current="+jq.hex8(heapCurrent)+", end="+jq.hex8(heapEnd)+")");
        Unsafe.poke4(heapStart + 4, heapCurrent);
        int newBlock;
        if (0 == (newBlock = SystemInterface.syscalloc(blockSize)))
            HeapAllocator.outOfMemory();
        Unsafe.poke4(heapStart, newBlock);
        heapStart = newBlock;
        Unsafe.poke4(heapStart, 0);
        Unsafe.poke4(heapStart + 4, heapEnd = newBlock + blockSize - 4);
        Unsafe.poke4(heapEnd, heapCurrent = newBlock + 8);
        if (TRACE) SystemInterface.debugmsg("Allocated new code block, start="+jq.hex8(heapCurrent)+" end="+jq.hex8(heapEnd));
    }
    
    public void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap) {
        Unsafe.poke4(code, heap);
    }
    public void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target) {
        Unsafe.poke4(code, target-code-4);
    }
    
    public class Runtimex86CodeBuffer extends CodeAllocator.x86CodeBuffer {

        private int/*CodeAddress*/ startAddress;
        private int/*CodeAddress*/ entrypointAddress;
        private int/*CodeAddress*/ currentAddress;
        private int/*CodeAddress*/ endAddress;

        Runtimex86CodeBuffer(int startAddress, int endAddress) {
            this.startAddress = startAddress;
            this.endAddress = endAddress;
            this.currentAddress = startAddress-1;
        }
        
        public int getCurrentOffset() { return currentAddress - startAddress + 1; }
        public int getCurrentAddress() { return currentAddress + 1; }
        
        public int/*CodeAddress*/ getStart() { return startAddress; }
        public int/*CodeAddress*/ getCurrent() { return currentAddress+1; }
        public int/*CodeAddress*/ getEntry() { return entrypointAddress; }
        public int/*CodeAddress*/ getEnd() { return endAddress; }
        
        public void setEntrypoint() { this.entrypointAddress = getCurrent(); }
        
        public void checkSize(int size) {
            if (currentAddress+size < endAddress) return;
            // overflow!
            int newEstimatedSize = (endAddress - startAddress) << 1;
            allocateNewBlock(Math.max(BLOCK_SIZE, newEstimatedSize));
            jq.Assert(currentAddress-startAddress+size < heapEnd-heapCurrent);
            SystemInterface.mem_cpy(heapCurrent, startAddress, currentAddress-startAddress);
            if (entrypointAddress != 0)
                entrypointAddress = entrypointAddress - startAddress + heapCurrent;
            currentAddress = currentAddress - startAddress + heapCurrent;
            startAddress = heapCurrent;
            endAddress = heapEnd;
        }
        
        public void add1(byte i) {
            checkSize(1);
            Unsafe.poke1(++currentAddress, i);
        }
        public void add2_endian(int i) {
            checkSize(2);
            Unsafe.poke2(++currentAddress, (short)i);
            ++currentAddress;
        }
        public void add2(int i) {
            checkSize(2);
            Unsafe.poke2(++currentAddress, endian2(i));
            ++currentAddress;
        }
        public void add3(int i) {
            checkSize(3);
            Unsafe.poke1(++currentAddress, (byte)(i >> 16));
            Unsafe.poke2(++currentAddress, endian2(i));
            ++currentAddress;
        }
        public void add4_endian(int i) {
            checkSize(4);
            Unsafe.poke4(++currentAddress, i);
            currentAddress += 3;
        }

        public byte get1(int k) {
            k += startAddress;
            return (byte)Unsafe.peek(k);
        }
        public int get4_endian(int k) {
            k += startAddress;
            return Unsafe.peek(k);
        }

        public void put1(int k, byte instr) {
            k += startAddress;
            Unsafe.poke1(k, instr);
        }
        public void put4_endian(int k, int instr) {
            k += startAddress;
            Unsafe.poke4(k, instr);
        }

        public void skip(int nbytes) {
        	currentAddress += nbytes;
        }
        
        public jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex,
                                                 jq_BytecodeMap bcm, ExceptionDeliverer exd,
                                                 List code_relocs, List data_relocs) {
            jq.Assert(isGenerating);
            int start = getStart();
            int entrypoint = getEntry();
            int current = getCurrent();
            int end = getEnd();
            jq.Assert(current <= end);
            if (TRACE) SystemInterface.debugmsg("Allocating code block, start="+jq.hex8(start)+" current="+jq.hex8(current)+" end="+jq.hex8(end));
            if (end != heapEnd) {
                if (TRACE) SystemInterface.debugmsg("Prior block, recalculating maxfreeprevious (was "+jq.hex(maxFreePrevious)+")");
                // prior block
                Unsafe.poke4(end, current);
                // recalculate max free previous
                maxFreePrevious = 0;
                int start_ptr = heapFirst;
                while (start_ptr != 0) {
                	int end_ptr = Unsafe.peek(start_ptr+4);
                    int current_ptr = Unsafe.peek(end_ptr);
                    int temp = end_ptr = current_ptr;
                    if (TRACE) SystemInterface.debugmsg("Free space in block "+jq.hex8(start_ptr)+": "+jq.hex(temp)+")");
                    maxFreePrevious = Math.max(maxFreePrevious, temp);
                	start_ptr = Unsafe.peek(start_ptr);
                }
                if (TRACE) SystemInterface.debugmsg("New maxfreeprevious: "+jq.hex(maxFreePrevious));
            } else {
                // current block
                heapCurrent = current;
                Unsafe.poke4(heapEnd, heapCurrent);
            }
            isGenerating = false;
            if (TRACE) SystemInterface.debugmsg("Code generation completed: "+this);
            jq_CompiledCode cc = new jq_CompiledCode(m, start, current-start, entrypoint, ex, bcm, exd, code_relocs, data_relocs);
            CodeAllocator.registerCode(cc);
            return cc;
        }
    }

    public static short endian2(int k) {
        return (short)(((k>>8)&0xFF) | (k<<8));
    }
}

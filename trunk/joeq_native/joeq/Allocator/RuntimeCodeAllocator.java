/*
 * RuntimeCodeAllocator.java
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
import Run_Time.SystemInterface;
import jq;

import java.util.List;

public class RuntimeCodeAllocator extends CodeAllocator {

    public static final int ALIGNMENT = 32;
    
    /** Size of blocks allocated from the OS.  No single code buffer can be larger than this.
     */
    public static final int BLOCK_SIZE = 1048576;
    
    /** Pointers to the start, current, and end of the heap.
     */
    private int/*CodeAddress*/ heapFirst, heapCurrent, heapEnd;
    
    /** Max memory free in all allocated blocks.
     */
    private int maxFreePrevious;
    
    boolean isGenerating = false;
    
    public void init()
    throws OutOfMemoryError {
        if (0 == (heapCurrent = heapFirst = SystemInterface.syscalloc(BLOCK_SIZE)))
            HeapAllocator.outOfMemory();
        heapEnd = heapFirst + BLOCK_SIZE - 8;
        if (TRACE) SystemInterface.debugmsg("Initialized run-time code allocator, start="+jq.hex8(heapFirst)+" end="+jq.hex8(heapEnd));
    }
    
    public x86CodeBuffer getCodeBuffer(int estimated_size) {
        // should not be called recursively.
        jq.assert(!isGenerating); isGenerating = true;
        if (heapCurrent + estimated_size <= heapEnd) {
            if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimated_size)+" fits within free space in current block "+jq.hex8(heapCurrent)+"-"+jq.hex8(heapEnd));
            return new Runtimex86CodeBuffer(heapCurrent, heapEnd);
        }
        if (estimated_size < maxFreePrevious) {
            // use a prior block's unused space.
            if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimated_size)+" fits within a prior block: maxfreeprev="+jq.hex(maxFreePrevious));
            int ptr = heapFirst;
            for (;;) {
                jq.assert(ptr != 0);
                int ptr2 = ptr+BLOCK_SIZE-8;
                int ptr3 = Unsafe.peek(ptr2);
                if (TRACE) SystemInterface.debugmsg("Checking block "+jq.hex8(ptr)+"-"+jq.hex8(ptr2)+", current ptr="+jq.hex8(ptr3));
                if ((ptr3-ptr2) >= estimated_size) {
                    if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimated_size)+") fits within block "+jq.hex8(ptr3)+"-"+jq.hex8(ptr2));
                    return new Runtimex86CodeBuffer(ptr3, ptr2);
                }
                ptr = Unsafe.peek(ptr2+4);
                if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimated_size)+") doesn't fit, trying next block "+jq.hex8(ptr));
            }
        }
        if (TRACE) SystemInterface.debugmsg("Estimated size ("+jq.hex(estimated_size)+" is too large for current block "+jq.hex8(heapCurrent)+"-"+jq.hex8(heapEnd));
        // allocate new block.
        allocateNewBlock();
        return new Runtimex86CodeBuffer(heapCurrent, heapEnd);
    }
    
    private void allocateNewBlock()
    throws OutOfMemoryError {
        if (TRACE) SystemInterface.debugmsg("Allocating new code block (current="+jq.hex8(heapCurrent)+", end="+jq.hex8(heapEnd)+")");
        Unsafe.poke4(heapEnd, heapCurrent);
        if (0 == (heapCurrent = SystemInterface.syscalloc(BLOCK_SIZE)))
            HeapAllocator.outOfMemory();
        Unsafe.poke4(heapEnd+4, heapCurrent);
        heapEnd = heapCurrent + BLOCK_SIZE - 8;
        if (TRACE) SystemInterface.debugmsg("Allocated new code block, start="+jq.hex8(heapCurrent)+" end="+jq.hex8(heapEnd));
    }
    
    public void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap) {
        Unsafe.poke4(code, heap);
    }
    public void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target) {
        Unsafe.poke4(code, target-code-4);
    }
    
    public class Runtimex86CodeBuffer extends CodeAllocator.x86CodeBuffer {

        private int/*CodeAddress*/ startAddress, currentAddress, endAddress;

        Runtimex86CodeBuffer(int startAddress, int endAddress) {
            this.startAddress = startAddress;
            this.endAddress = endAddress;
            this.currentAddress = startAddress-1;
        }
        
        public int getCurrentOffset() { return currentAddress - startAddress + 1; }
        public int getCurrentAddress() { return currentAddress + 1; }
        
        public int/*CodeAddress*/ getStart() { return startAddress; }
        public int/*CodeAddress*/ getCurrent() { return currentAddress+1; }
        public int/*CodeAddress*/ getEnd() { return endAddress; }
        
        public void checkSize(int size) {
            if (currentAddress+size < endAddress) return;
            // overflow!
            allocateNewBlock();
            jq.assert(currentAddress-startAddress+size < heapEnd-heapCurrent);
            SystemInterface.mem_cpy(heapCurrent, startAddress, currentAddress-startAddress);
            currentAddress = currentAddress-startAddress+heapCurrent;
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

        public jq_CompiledCode allocateCodeBlock(jq_Method m, jq_TryCatch[] ex,
                                                 jq_BytecodeMap bcm, ExceptionDeliverer exd,
                                                 List code_relocs, List data_relocs) {
            jq.assert(isGenerating); isGenerating = false;
            int start = getStart();
            int current = getCurrent();
            int end = getEnd();
            jq.assert(current <= end);
            // align pointer for next allocation
            int align = current & (ALIGNMENT-1);
            if (align != 0) current += (ALIGNMENT-align);
            current = Math.min(current, end);
            if (TRACE) SystemInterface.debugmsg("Allocating code block, start="+jq.hex8(start)+" current="+jq.hex8(current)+" end="+jq.hex8(end));
            if (end != heapEnd) {
                if (TRACE) SystemInterface.debugmsg("Prior block, recalculating maxfreeprevious (was "+jq.hex(maxFreePrevious)+")");
                // prior block
                Unsafe.poke4(end, current);
                // recalculate max free previous
                int ptr = heapFirst + BLOCK_SIZE - 8;
                maxFreePrevious = ptr - Unsafe.peek(ptr);
                if (TRACE) SystemInterface.debugmsg("Free space in block "+jq.hex8(ptr-BLOCK_SIZE+8)+": "+jq.hex(maxFreePrevious)+")");
                for (;;) {
                    ptr = Unsafe.peek(ptr+4) + BLOCK_SIZE - 8;
                    if (ptr == heapEnd) break;
                    int temp = ptr - Unsafe.peek(ptr);
                    if (TRACE) SystemInterface.debugmsg("Free space in block "+jq.hex8(ptr-BLOCK_SIZE+8)+": "+jq.hex(temp)+")");
                    if (maxFreePrevious < temp) maxFreePrevious = temp;
                }
                if (TRACE) SystemInterface.debugmsg("New maxfreeprevious: "+jq.hex(maxFreePrevious));
            } else {
                // current block
                heapCurrent = current;
            }
            jq_CompiledCode cc = new jq_CompiledCode(m, start, current-start, ex, bcm, exd, code_relocs, data_relocs);
            CodeAllocator.registerCode(cc);
            return cc;
        }
    }

    public static short endian2(int k) {
        return (short)(((k>>8)&0xFF) | (k<<8));
    }
}

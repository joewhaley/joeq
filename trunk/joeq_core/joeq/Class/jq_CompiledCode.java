/*
 * jq_CompiledCode.java
 *
 * Created on January 5, 2001, 8:04 PM
 *
 */
package Clazz;

import java.util.Iterator;
import java.util.List;

import Allocator.CodeAllocator;
import Assembler.x86.DirectBindCall;
import Bootstrap.PrimordialClassLoader;
import Main.jq;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class jq_CompiledCode implements Comparable {

    public static /*final*/ boolean TRACE = false;
    public static /*final*/ boolean TRACE_REDIRECT = false;
    
    protected final int/*CodeAddress*/ entrypoint;
    protected final jq_Method method;
    protected final int/*CodeAddress*/ start;
    protected final int length;
    protected final jq_TryCatch[] handlers;
    protected final jq_BytecodeMap bcm;
    protected final ExceptionDeliverer ed;
    protected final List code_reloc, data_reloc;

    public jq_CompiledCode(jq_Method method,
                           int/*CodeAddress*/ start, int length,
                           int/*CodeAddress*/ entrypoint,
                           jq_TryCatch[] handlers, jq_BytecodeMap bcm,
                           ExceptionDeliverer ed, List code_reloc, List data_reloc) {
        this.method = method;
        this.entrypoint = entrypoint;
        this.start = start;
        this.length = length;
        this.handlers = handlers;
        this.bcm = bcm;
        this.ed = ed;
        this.code_reloc = code_reloc;
        this.data_reloc = data_reloc;
    }
    
    public jq_Method getMethod() { return method; }
    public int/*CodeAddress*/ getStart() { return start; }
    public int getLength() { return length; }
    public int/*CodeAddress*/ getEntrypoint() { return entrypoint; }

    public int/*CodeAddress*/ findCatchBlock(int/*CodeAddress*/ ip, jq_Class extype) {
        int offset = ip - start;
        if (TRACE) SystemInterface.debugmsg("checking for handlers for ip "+jq.hex8(ip)+" offset "+jq.hex(offset)+" in "+this);
        if (handlers == null) {
            if (TRACE) SystemInterface.debugmsg("no handlers in "+this);
            return 0;
        }
        for (int i=0; i<handlers.length; ++i) {
            jq_TryCatch tc = handlers[i];
            if (TRACE) SystemInterface.debugmsg("checking handler: "+tc);
            if (tc.catches(offset, extype))
                return tc.getHandlerEntry()+start;
            if (TRACE) SystemInterface.debugmsg("does not catch");
        }
        if (TRACE) SystemInterface.debugmsg("no appropriate handler found in "+this);
        return 0;
    }
    
    public void deliverException(int/*CodeAddress*/ entry, int/*StackAddress*/ fp, Throwable x) {
        jq.Assert(ed != null);
        ed.deliverToStackFrame(this, x, entry, fp);
    }
    
    public Object getThisPointer(int/*CodeAddress*/ ip, int/*StackAddress*/ fp) {
        jq.Assert(ed != null);
        return ed.getThisPointer(this, ip, fp);
    }
    
    public int getBytecodeIndex(int ip) {
        if (bcm == null) return -1;
        return bcm.getBytecodeIndex(ip-start);
    }
    
    /** Rewrite the entrypoint to branch to the given compiled code. */
    public void redirect(jq_CompiledCode that) {
        int/*CodeAddress*/ newEntrypoint = that.getEntrypoint();
        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting "+this+" to point to "+that);
        if (entrypoint >= start+5) {
	        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting via trampoline");
        	// both should start with "push EBP"
            jq.Assert((Unsafe.peek(entrypoint) & 0xFF) == (Unsafe.peek(newEntrypoint) & 0xFF));
            // put target address (just after push EBP)
            Unsafe.poke4(entrypoint - 4, newEntrypoint + 1 - entrypoint);
            // put jump instruction
            Unsafe.poke1(entrypoint - 5, (byte)0xE9); // JMP
            // put backward branch to jump instruction
            Unsafe.poke2(entrypoint + 1, (short)0xF8EB);
        } else {
	        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting by rewriting targets");
	        Iterator it = CodeAllocator.getCompiledMethods();
	        while (it.hasNext()) {
	        	jq_CompiledCode cc = (jq_CompiledCode)it.next();
	        	cc.patchDirectBindCalls(this.method, that);
	        }
        }
    }
    
    public String toString() { return method+" address: ("+jq.hex8(start)+"-"+jq.hex8(start+length)+")"; }

    public boolean contains(int address) {
        return address >= start && address < start+length;
    }
    
    public void patchDirectBindCalls() {
        jq.Assert(!jq.Bootstrapping);
        if (code_reloc != null) {
            Iterator i = code_reloc.iterator();
            while (i.hasNext()) {
                DirectBindCall r = (DirectBindCall)i.next();
                r.patch();
            }
        }
    }
    
    public void patchDirectBindCalls(jq_Method method, jq_CompiledCode cc) {
        jq.Assert(!jq.Bootstrapping);
        if (code_reloc != null) {
            Iterator i = code_reloc.iterator();
            while (i.hasNext()) {
                DirectBindCall r = (DirectBindCall)i.next();
                if (r.getTarget() == method) {
			        if (TRACE_REDIRECT) SystemInterface.debugmsg("patching direct bind call in "+this+" at "+jq.hex8(r.getSource())+" to refer to "+cc);
	                r.patchTo(cc);
                }
            }
        }
    }
    
    public int compareTo(CodeAllocator.InstructionPointer that) {
        int/*CodeAddress*/ ip = that.getIP();
        if (this.start >= ip) return 1;
        if (this.start+this.length < ip) return -1;
        return 0;
    }
    public int compareTo(jq_CompiledCode that) {
        if (this == that) return 0;
        if (this.start < that.start) return -1;
        if (this.start < that.start+that.length) {
            jq.UNREACHABLE(this+" overlaps "+that);
        }
        return 1;
    }
    public int compareTo(java.lang.Object o) {
        if (o instanceof jq_CompiledCode)
            return compareTo((jq_CompiledCode)o);
        else
            return compareTo((CodeAllocator.InstructionPointer)o);
    }
    public boolean equals(CodeAllocator.InstructionPointer that) {
        int/*CodeAddress*/ ip = that.getIP();
        if (ip < start) return false;
        if (ip > start+length) return false;
        return true;
    }
    public boolean equals(Object o) {
        if (o instanceof jq_CompiledCode)
            return this == o;
        else
            return equals((CodeAllocator.InstructionPointer)o);
    }
    /**
     * NOTE that this violates the contract of hashCode when comparing against InstructionPointer objects!
     */
    public int hashCode() { return super.hashCode(); }
    
    public static final jq_InstanceField _entrypoint;
    public static final jq_InstanceField _start;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_CompiledCode;");
        _entrypoint = k.getOrCreateInstanceField("entrypoint", "I");
        _start = k.getOrCreateInstanceField("start", "I");
    }
}

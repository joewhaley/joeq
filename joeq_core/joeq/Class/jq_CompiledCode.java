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
import Memory.CodeAddress;
import Memory.StackAddress;
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
    
    protected final CodeAddress entrypoint;
    protected final jq_Method method;
    protected final CodeAddress start;
    protected final int length;
    protected final jq_TryCatch[] handlers;
    protected final jq_BytecodeMap bcm;
    protected final ExceptionDeliverer ed;
    protected final List code_reloc, data_reloc;

    public jq_CompiledCode(jq_Method method,
                           CodeAddress start, int length,
                           CodeAddress entrypoint,
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
    public CodeAddress getStart() { return start; }
    public int getLength() { return length; }
    public CodeAddress getEntrypoint() { return entrypoint; }

    public CodeAddress findCatchBlock(CodeAddress ip, jq_Class extype) {
        int offset = ip.difference(start);
        if (handlers == null) {
            if (TRACE) SystemInterface.debugmsg("no handlers in "+this);
            return null;
        }
        for (int i=0; i<handlers.length; ++i) {
            jq_TryCatch tc = handlers[i];
            if (TRACE) SystemInterface.debugmsg("checking handler: "+tc);
            if (tc.catches(offset, extype))
                return (CodeAddress)start.offset(tc.getHandlerEntry());
            if (TRACE) SystemInterface.debugmsg("does not catch");
        }
        if (TRACE) SystemInterface.debugmsg("no appropriate handler found in "+this);
        return null;
    }
    
    public void deliverException(CodeAddress entry, StackAddress fp, Throwable x) {
        jq.Assert(ed != null);
        ed.deliverToStackFrame(this, x, entry, fp);
    }
    
    public Object getThisPointer(CodeAddress ip, StackAddress fp) {
        jq.Assert(ed != null);
        return ed.getThisPointer(this, ip, fp);
    }
    
    public int getBytecodeIndex(CodeAddress ip) {
        if (bcm == null) return -1;
        return bcm.getBytecodeIndex(ip.difference(start));
    }
    
    /** Rewrite the entrypoint to branch to the given compiled code. */
    public void redirect(jq_CompiledCode that) {
        CodeAddress newEntrypoint = that.getEntrypoint();
        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting "+this+" to point to "+that);
        if (entrypoint.difference(start.offset(5)) >= 0) {
	        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting via trampoline");
        	// both should start with "push EBP"
            jq.Assert(entrypoint.peek1() == newEntrypoint.peek1());
            // put target address (just after push EBP)
            entrypoint.offset(-4).poke4(newEntrypoint.difference(entrypoint)+1);
            // put jump instruction
            entrypoint.offset(-5).poke1((byte)0xE9); // JMP
            // put backward branch to jump instruction
            entrypoint.offset(1).poke2((short)0xF8EB); // JMP
        } else {
	        if (TRACE_REDIRECT) SystemInterface.debugmsg("redirecting by rewriting targets");
	        Iterator it = CodeAllocator.getCompiledMethods();
	        while (it.hasNext()) {
	        	jq_CompiledCode cc = (jq_CompiledCode)it.next();
	        	cc.patchDirectBindCalls(this.method, that);
	        }
        }
    }
    
    public String toString() { return method+" address: ("+start.stringRep()+"-"+start.offset(length).stringRep()+")"; }

    public boolean contains(CodeAddress address) {
        return address.difference(start) >= 0 && address.difference(start.offset(length)) < 0;
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
                    if (TRACE_REDIRECT) SystemInterface.debugmsg("patching direct bind call in "+this+" at "+r.getSource().stringRep()+" to refer to "+cc);
	            r.patchTo(cc);
                }
            }
        }
    }
    
    public int compareTo(CodeAllocator.InstructionPointer that) {
        CodeAddress ip = that.getIP();
        if (this.start.difference(ip) >= 0) return 1;
        if (this.start.offset(this.length).difference(ip) < 0) return -1;
        return 0;
    }
    public int compareTo(jq_CompiledCode that) {
        if (this == that) return 0;
        if (this.start.difference(that.start) < 0) return -1;
        if (this.start.difference(that.start.offset(that.length)) < 0) {
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
        CodeAddress ip = that.getIP();
        if (ip.difference(start) < 0) return false;
        if (ip.difference(start.offset(length)) > 0) return false;
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
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_CompiledCode;");
        _entrypoint = k.getOrCreateInstanceField("entrypoint", "LMemory/CodeAddress;");
    }
}

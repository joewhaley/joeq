/*
 * jq_CompiledCode.java
 *
 * Created on January 5, 2001, 8:04 PM
 *
 */
package Clazz;

import Assembler.x86.DirectBindCall;
import Allocator.CodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import UTF.Utf8;

import jq;

import java.util.List;
import java.util.Iterator;

/**
 * @author  John Whaley
 * @version 
 */
public class jq_CompiledCode implements Comparable {

    public static /*final*/ boolean TRACE = false;
    
    protected final int/*CodeAddress*/ entrypoint;
    protected final jq_Method method;
    protected final int length;
    protected final jq_TryCatch[] handlers;
    protected final jq_BytecodeMap bcm;
    protected final ExceptionDeliverer ed;
    protected final List code_reloc, data_reloc;

    public jq_CompiledCode(jq_Method method,
                           int/*CodeAddress*/ entrypoint, int length,
                           jq_TryCatch[] handlers, jq_BytecodeMap bcm,
                           ExceptionDeliverer ed, List code_reloc, List data_reloc) {
        this.method = method;
        this.entrypoint = entrypoint;
        this.length = length;
        this.handlers = handlers;
        this.bcm = bcm;
        this.ed = ed;
        this.code_reloc = code_reloc;
        this.data_reloc = data_reloc;
    }
    
    public jq_Method getMethod() { return method; }
    public int/*CodeAddress*/ getEntrypoint() { return entrypoint; }
    public int getLength() { return length; }

    public int/*CodeAddress*/ findCatchBlock(int/*CodeAddress*/ ip, jq_Class extype) {
        int offset = ip - entrypoint;
        if (TRACE) SystemInterface.debugmsg("checking for handlers for ip "+jq.hex8(ip)+" offset "+jq.hex(offset)+" in "+this);
        if (handlers == null) {
            if (TRACE) SystemInterface.debugmsg("no handlers in "+this);
            return 0;
        }
        for (int i=0; i<handlers.length; ++i) {
            jq_TryCatch tc = handlers[i];
            if (TRACE) SystemInterface.debugmsg("checking handler: "+tc);
            if (tc.catches(offset, extype))
                return tc.getHandlerEntry()+entrypoint;
            if (TRACE) SystemInterface.debugmsg("does not catch");
        }
        if (TRACE) SystemInterface.debugmsg("no appropriate handler found in "+this);
        return 0;
    }
    
    public void deliverException(int/*CodeAddress*/ entry, int/*StackAddress*/ fp, Throwable x) {
        jq.assert(ed != null);
        ed.deliverToStackFrame(this, x, entry, fp);
    }
    
    public Object getThisPointer(int/*CodeAddress*/ ip, int/*StackAddress*/ fp) {
        jq.assert(ed != null);
        return ed.getThisPointer(this, ip, fp);
    }
    
    public int getBytecodeIndex(int ip) {
        if (bcm == null) return -1;
        return bcm.getBytecodeIndex(ip-entrypoint);
    }
    
    public String toString() { return method+" address: ("+jq.hex8(entrypoint)+"-"+jq.hex8(entrypoint+length)+")"; }

    public boolean contains(int address) {
        return address >= entrypoint && address < entrypoint+length;
    }
    
    public void patchDirectBindCalls() {
        jq.assert(!jq.Bootstrapping);
        if (code_reloc != null) {
            Iterator i = code_reloc.iterator();
            while (i.hasNext()) {
                DirectBindCall r = (DirectBindCall)i.next();
                r.patch();
            }
        }
    }
    
    public int compareTo(CodeAllocator.InstructionPointer that) {
        int/*CodeAddress*/ ip = that.getIP();
        if (this.entrypoint >= ip) return 1;
        if (this.entrypoint+this.length < ip) return -1;
        return 0;
    }
    public int compareTo(jq_CompiledCode that) {
        if (this == that) return 0;
        if (this.entrypoint < that.entrypoint) return -1;
        if (this.entrypoint < that.entrypoint+that.length) {
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
        if (ip < entrypoint) return false;
        if (ip > entrypoint+length) return false;
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
        _entrypoint = k.getOrCreateInstanceField("entrypoint", "I");
    }
}

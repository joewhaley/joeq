/*
 * jq_CompiledCode.java
 *
 * Created on January 5, 2001, 8:04 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

import Allocator.CodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Run_Time.ExceptionDeliverer;
import Run_Time.SystemInterface;
import UTF.Utf8;

import jq;

// NOTE: doesn't implement hashCode() very well!
public class jq_CompiledCode implements Comparable {

    public static /*final*/ boolean TRACE = false;
    
    protected final int/*Address*/ entrypoint;
    protected final jq_Method method;
    protected final int length;
    protected final jq_TryCatch[] handlers;
    protected final jq_BytecodeMap bcm;
    protected final ExceptionDeliverer ed;

    public jq_CompiledCode(jq_Method method,
                           int/*Address*/ entrypoint, int length,
                           jq_TryCatch[] handlers, jq_BytecodeMap bcm,
                           ExceptionDeliverer ed) {
        this.method = method;
        this.entrypoint = entrypoint;
        this.length = length;
        this.handlers = handlers;
        this.bcm = bcm;
        this.ed = ed;
    }
    
    public jq_Method getMethod() { return method; }
    public int/*Address*/ getEntrypoint() { return entrypoint; }
    public int getLength() { return length; }

    public int/*Address*/ findCatchBlock(int/*Address*/ ip, jq_Class extype) {
        int offset = ip - entrypoint;
        if (TRACE) SystemInterface.debugmsg("checking for handlers for ip "+jq.hex8(ip)+" offset "+jq.hex(offset)+" in "+method);
        for (int i=0; i<handlers.length; ++i) {
            jq_TryCatch tc = handlers[i];
            if (TRACE) SystemInterface.debugmsg("checking handler: "+tc);
            if (tc.catches(offset, extype))
                return tc.getHandlerEntry()+entrypoint;
            if (TRACE) SystemInterface.debugmsg("does not catch");
        }
        if (TRACE) SystemInterface.debugmsg("no appropriate handler found in "+method);
        return 0;
    }
    
    public void deliverException(int/*Address*/ entry, int/*Address*/ fp, Throwable x) {
        ed.deliverToStackFrame(this, x, entry, fp);
    }
    
    public int getBytecodeIndex(int ip) { return bcm.getBytecodeIndex(ip-entrypoint); }
    
    public String toString() { return method+" address: ("+jq.hex8(entrypoint)+"-"+jq.hex8(entrypoint+length)+")"; }

    public boolean contains(int address) {
        return address >= entrypoint && address < entrypoint+length;
    }
    
    public int compareTo(CodeAllocator.InstructionPointer that) {
        int/*Address*/ ip = that.getIP();
        if (this.entrypoint > ip) return 1;
        if (this.entrypoint+this.length < ip) return -1;
        return 0;
    }
    public int compareTo(jq_CompiledCode that) {
        if (this == that) return 0;
        if (this.entrypoint < that.entrypoint) return -1;
        jq.assert(this.entrypoint >= that.entrypoint+that.length);
        return 1;
    }
    public int compareTo(Object o) {
        if (o instanceof jq_CompiledCode)
            return compareTo((jq_CompiledCode)o);
        else
            return compareTo((CodeAllocator.InstructionPointer)o);
    }
    public boolean equals(CodeAllocator.InstructionPointer that) {
        int/*Address*/ ip = that.getIP();
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
    public int hashCode() { return 0; }
    
    public static final jq_InstanceField _entrypoint;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_CompiledCode;");
        _entrypoint = k.getOrCreateInstanceField("entrypoint", "I");
    }
}

/*
 * StackWalker.java
 *
 * Created on January 11, 2001, 10:34 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Allocator.CodeAllocator;
import Clazz.jq_CompiledCode;
import Run_Time.Unsafe;
import Run_Time.SystemInterface;
import jq;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class StackWalker implements Iterator {

    public static /*final*/ boolean TRACE = false;
    
    int/*CodeAddress*/ ip;
    int/*StackAddress*/ fp;
    
    public int/*CodeAddress*/ getIP() { return ip; }
    public int/*StackAddress*/ getFP() { return fp; }
    public jq_CompiledCode getCode() { return CodeAllocator.getCodeContaining(ip); }

    public StackWalker(int/*CodeAddress*/ ip, int/*StackAddress*/ fp) {
        this.ip = ip;
        this.fp = fp;
        if (TRACE) SystemInterface.debugmsg("StackWalker init: fp="+jq.hex8(fp)+" ip="+jq.hex8(ip)+" "+getCode());
    }
    
    public void gotoNext() throws NoSuchElementException {
        if (fp == 0) throw new NoSuchElementException();
        ip = Unsafe.peek(fp+4);
        fp = Unsafe.peek(fp);
        if (TRACE) SystemInterface.debugmsg("StackWalker next: fp="+jq.hex8(fp)+" ip="+jq.hex8(ip)+" "+getCode());
    }
    
    public boolean hasNext() {
        if (fp == 0) return false;
        int/*CodeAddress*/ addr = Unsafe.peek(fp+4);
        if (TRACE) SystemInterface.debugmsg("StackWalker hasnext: next ip="+jq.hex8(addr)+" "+CodeAllocator.getCodeContaining(addr));
        return true;
    }
    
    public Object next() throws NoSuchElementException {
        gotoNext();
        return new CodeAllocator.InstructionPointer(ip);
    }
    
    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
}

/*
 * StackWalker.java
 *
 * Created on January 11, 2001, 10:34 PM
 *
 */

package Run_Time;

import java.util.Iterator;
import java.util.NoSuchElementException;

import Allocator.CodeAllocator;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Main.jq;
import Memory.CodeAddress;
import Memory.StackAddress;
import UTF.Utf8;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class StackWalker implements Iterator {

    public static /*final*/ boolean TRACE = false;
    
    CodeAddress ip;
    StackAddress fp;
    
    public CodeAddress getIP() { return ip; }
    public StackAddress getFP() { return fp; }
    public jq_CompiledCode getCode() { return CodeAllocator.getCodeContaining(ip); }

    public StackWalker(CodeAddress ip, StackAddress fp) {
        this.ip = ip;
        this.fp = fp;
        if (TRACE) SystemInterface.debugmsg("StackWalker init: fp="+fp.stringRep()+" ip="+ip.stringRep()+" "+getCode());
    }
    
    public void gotoNext() throws NoSuchElementException {
        if (fp.isNull()) throw new NoSuchElementException();
        ip = (CodeAddress) fp.offset(4).peek();
        fp = (StackAddress) fp.peek();
        if (TRACE) SystemInterface.debugmsg("StackWalker next: fp="+fp.stringRep()+" ip="+ip.stringRep()+" "+getCode());
    }
    
    public boolean hasNext() {
        if (fp.isNull()) return false;
        CodeAddress addr = (CodeAddress) fp.offset(4).peek();
        if (TRACE) SystemInterface.debugmsg("StackWalker hasnext: next ip="+addr.stringRep()+" "+CodeAllocator.getCodeContaining(addr));
        return true;
    }
    
    public Object next() throws NoSuchElementException {
        gotoNext();
        return new CodeAllocator.InstructionPointer(ip);
    }
    
    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
    
    public static void stackDump(CodeAddress init_ip, StackAddress init_fp) {
        StackWalker sw = new StackWalker(init_ip, init_fp);
        while (sw.hasNext()) {
            jq_CompiledCode cc = sw.getCode();
            CodeAddress ip = sw.getIP();
            String s;
            if (cc != null) {
                jq_Method m = cc.getMethod();
                int code_offset = ip.difference(cc.getStart());
                if (m != null) {
                    Utf8 sourcefile = m.getDeclaringClass().getSourceFile();
                    int bc_index = cc.getBytecodeIndex(ip);
                    int line_num = m.getLineNumber(bc_index);
                    s = "\tat "+m+" ("+sourcefile+":"+line_num+" bc:"+bc_index+" off:"+jq.hex(code_offset)+")";
                } else {
                    s = "\tat <unknown cc> (start:"+cc.getStart().stringRep()+" off:"+jq.hex(code_offset)+")";
                }
            } else {
                s = "\tat <unknown addr> (ip:"+ip.stringRep()+")";
            }
            SystemInterface.debugmsg(s);
            sw.gotoNext();
        }
    }
    
}

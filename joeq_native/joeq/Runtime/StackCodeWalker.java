/*
 * StackCodeWalker.java
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
import Memory.HeapAddress;
import UTF.Utf8;
import Util.Strings;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class StackCodeWalker implements Iterator {

    public static /*final*/ boolean TRACE = false;

    CodeAddress ip;
    StackAddress fp;

    public CodeAddress getIP() {
        return ip;
    }

    public StackAddress getFP() {
        return fp;
    }

    public jq_CompiledCode getCode() {
        return CodeAllocator.getCodeContaining(ip);
    }

    public jq_Method getMethod() {
        jq_CompiledCode cc = this.getCode();
        if (cc == null) return null;
        return cc.getMethod();
    }
    
    public int getCodeOffset() {
        jq_CompiledCode cc = this.getCode();
        if (cc == null) return 0;
        CodeAddress ip = this.getIP();
        int code_offset = ip.difference(cc.getStart());
        return code_offset;
    }
    
    public Utf8 getSourceFile() {
        jq_Method m = getMethod();
        if (m == null) return null;
        Utf8 sourcefile = m.getDeclaringClass().getSourceFile();
        return sourcefile;
    }
    public int getBCIndex() {
        jq_CompiledCode cc = this.getCode();
        if (cc == null) return -1;
        int bc_index = cc.getBytecodeIndex(ip);
        return bc_index;
    }
    
    public int getLineNum() {
        jq_Method m = getMethod();
        if (m == null) return -1;
        int bc_index = getBCIndex();
        int line_num = m.getLineNumber(bc_index);
        return line_num;
    }
    
    public StackCodeWalker(CodeAddress ip, StackAddress fp) {
        this.ip = ip;
        this.fp = fp;
        if (TRACE) SystemInterface.debugmsg("StackCodeWalker init: fp=" + fp.stringRep() + " ip=" + ip.stringRep() + " " + getCode());
    }

    public void gotoNext() throws NoSuchElementException {
        if (fp.isNull()) throw new NoSuchElementException();
        ip = (CodeAddress) fp.offset(4).peek();
        fp = (StackAddress) fp.peek();
        if (TRACE) SystemInterface.debugmsg("StackCodeWalker next: fp=" + fp.stringRep() + " ip=" + ip.stringRep() + " " + getCode());
    }

    public boolean hasNext() {
        if (fp.isNull()) return false;
        CodeAddress addr = (CodeAddress) fp.offset(4).peek();
        if (TRACE) SystemInterface.debugmsg("StackCodeWalker hasnext: next ip=" + addr.stringRep() + " " + CodeAllocator.getCodeContaining(addr));
        return true;
    }

    public Object next() throws NoSuchElementException {
        gotoNext();
        return new CodeAllocator.InstructionPointer(ip);
    }

    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }

    public String toString() {
        jq_CompiledCode cc = this.getCode();
        CodeAddress ip = this.getIP();
        String s;
        if (cc != null) {
            jq_Method m = cc.getMethod();
            int code_offset = ip.difference(cc.getStart());
            if (m != null) {
                Utf8 sourcefile = m.getDeclaringClass().getSourceFile();
                int bc_index = cc.getBytecodeIndex(ip);
                int line_num = m.getLineNumber(bc_index);
                s = "\tat " + m + " (" + sourcefile + ":" + line_num + " bc:" + bc_index + " off:" + Strings.hex(code_offset) + ")";
            } else {
                s = "\tat <unknown cc> (start:" + cc.getStart().stringRep() + " off:" + Strings.hex(code_offset) + ")";
            }
        } else {
            s = "\tat <unknown addr> (ip:" + ip.stringRep() + ")";
        }
        return s;
    }
    
    public static void stackDump(CodeAddress init_ip, StackAddress init_fp) {
        StackCodeWalker sw = new StackCodeWalker(init_ip, init_fp);
        while (sw.hasNext()) {
            String s = sw.toString();
            SystemInterface.debugmsg(s);
            sw.gotoNext();
        }
    }

}

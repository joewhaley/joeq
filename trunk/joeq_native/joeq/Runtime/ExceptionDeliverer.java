/*
 * ExceptionDeliverer.java
 *
 * Created on January 11, 2001, 10:34 PM
 *
 */

package Run_Time;

import Allocator.CodeAllocator;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Main.jq;
import Memory.CodeAddress;
import Memory.StackAddress;
import UTF.Utf8;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static void athrow(Throwable k) {
        CodeAddress ip = (CodeAddress) StackAddress.getBasePointer().offset(StackAddress.size()).peek();
        StackAddress fp = (StackAddress) StackAddress.getBasePointer().peek();
        ExceptionDeliverer.deliverToCurrentThread(k, ip, fp);
        jq.UNREACHABLE();
    }
    
    public static void trap_handler(int code) {
        switch (code) {
            case 0: throw new NullPointerException();
            case 1: throw new ArrayIndexOutOfBoundsException();
            case 2: throw new ArithmeticException();
            case 3: throw new StackOverflowError();
            default: throw new InternalError("unknown hardware exception type: "+code);
        }
    }
    
    public abstract void deliverToStackFrame(jq_CompiledCode cc, Throwable x, CodeAddress ip, StackAddress fp);
    public abstract Object getThisPointer(jq_CompiledCode cc, CodeAddress ip, StackAddress fp);
    
    public static void deliverToCurrentThread(Throwable x, CodeAddress ip, StackAddress fp) {
        jq_Class x_type = (jq_Class) jq_Reference.getTypeOf(x);
        if (TRACE) SystemInterface.debugmsg("Delivering exception of type "+x_type+" to ip="+ip.stringRep()+" fp="+fp.stringRep());
        for (;;) {
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
            if (TRACE) SystemInterface.debugmsg("Checking compiled code "+cc);
            if ((cc == null) || (fp.isNull())) {
                // reached the top!
                System.out.println("Exception in thread \""+Unsafe.getThreadBlock()+"\" "+x);
                x.printStackTrace(System.out);
                SystemInterface.die(-1);
                jq.UNREACHABLE();
                return;
            } else {
                CodeAddress address = cc.findCatchBlock(ip, x_type);
                if (!address.isNull()) {
                    // TODO: analyze the catch block to see if the backtrace is necessary.
                    if (true) {
                        //StackFrame sf = (StackFrame)Reflection.getfield_A(x, ClassLib.sun13.java.lang.Throwable._backtrace);
                        //sf.fillInStackTrace();
                    }
                    
                    // go to this catch block!
                    if (TRACE) SystemInterface.debugmsg("Jumping to catch block at "+address.stringRep());
                    cc.deliverException(address, fp, x);
                    jq.UNREACHABLE();
                    return;
                }
                if (cc.getMethod() != null && cc.getMethod().isSynchronized()) {
                    // need to perform monitorexit here.
                    Object o;
                    if (cc.getMethod().isStatic()) {
                        o = Reflection.getJDKType(cc.getMethod().getDeclaringClass());
                        if (TRACE) SystemInterface.debugmsg("Performing monitorexit on static method "+cc.getMethod()+": object "+o);
                    } else {
                        o = cc.getThisPointer(ip, fp);
                        if (TRACE) SystemInterface.debugmsg("Performing monitorexit on instance method "+cc.getMethod()+": object "+o.getClass()+"@"+jq.hex(System.identityHashCode(o)));
                    }
                    Monitor.monitorexit(o);
                }
                ip = (CodeAddress) fp.offset(StackAddress.size()).peek();
                fp = (StackAddress) fp.peek();
            }
        }
    }
    
    public static void printStackTrace(Object backtrace, java.io.PrintWriter pw) {
        StackFrame sf = (StackFrame)backtrace;
        while (sf.next != null) {
            CodeAddress ip = sf.ip;
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
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
            pw.println(s.toCharArray());
            sf = sf.next;
        }
    }

    public static void printStackTrace(Object backtrace, java.io.PrintStream pw) {
        StackFrame sf = (StackFrame)backtrace;
        while (sf.next != null) {
            CodeAddress ip = sf.ip;
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
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
            pw.println(s.toCharArray());
            sf = sf.next;
        }
    }
    
    public static void printStackTrace(Object backtrace) {
        StackFrame sf = (StackFrame)backtrace;
        while (sf.next != null) {
            CodeAddress ip = sf.ip;
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
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
            sf = sf.next;
        }
    }
    
    public static Object getStackTrace() {
        // stack traces are a linked list.
        CodeAddress ip = (CodeAddress) StackAddress.getBasePointer().offset(StackAddress.size()).peek();
        StackAddress fp = (StackAddress) StackAddress.getBasePointer().peek();
        StackFrame sf = new StackFrame(fp, ip);
        sf.fillInStackTrace();
        return sf;
    }
    
    public static class StackFrame {
        protected StackAddress fp; // location of this stack frame
        protected CodeAddress ip;  // ip address
        protected StackFrame next; // next frame in linked list
        
        public StackFrame(StackAddress fp, CodeAddress ip) {
            this.fp = fp; this.ip = ip;
        }
        
        public void fillInStackTrace() {
            StackFrame dis = this;
            while (!dis.fp.isNull()) {
                CodeAddress ip2 = (CodeAddress) dis.fp.offset(StackAddress.size()).peek();
                StackAddress fp2 = (StackAddress) dis.fp.peek();
                dis.next = new StackFrame(fp2, ip2);
                dis = dis.next;
            }
        }

        public StackFrame getNext() { return next; }
        public StackAddress getFP() { return fp; }
        public CodeAddress getIP() { return ip; }
    }
    
    public static final jq_StaticMethod _athrow;
    public static final jq_StaticMethod _trap_handler;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/ExceptionDeliverer;");
        _athrow = k.getOrCreateStaticMethod("athrow", "(Ljava/lang/Throwable;)V");
        _trap_handler = k.getOrCreateStaticMethod("trap_handler", "(I)V");
    }
}

/*
 * ExceptionDeliverer.java
 *
 * Created on January 11, 2001, 10:34 PM
 *
 * @author  jwhaley
 * @version 
 */

package Run_Time;

import Allocator.CodeAllocator;
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_StaticMethod;
import Clazz.jq_Method;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
import UTF.Utf8;
import jq;

public abstract class ExceptionDeliverer {

    public static /*final*/ boolean TRACE = false;
    
    public static void athrow(Throwable k) {
        int ip = Unsafe.peek(Unsafe.EBP()+4);
        int fp = Unsafe.peek(Unsafe.EBP());
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
    
    public abstract void deliverToStackFrame(jq_CompiledCode cc, Throwable x, int ip, int fp);
    public abstract Object getThisPointer(jq_CompiledCode cc, int ip, int fp);
    
    public static void deliverToCurrentThread(Throwable x, int ip, int fp) {
        jq_Class x_type = (jq_Class)Unsafe.getTypeOf(x);
        if (TRACE) SystemInterface.debugmsg("Delivering exception of type "+x_type+" to ip="+jq.hex8(ip)+" fp="+jq.hex8(fp));
        for (;;) {
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
            if (TRACE) SystemInterface.debugmsg("Checking compiled code "+cc);
            if ((cc == null) || (fp == 0)) {
                // reached the top!
                System.out.println("Exception in thread \""+Unsafe.getThreadBlock()+"\" "+x);
                x.printStackTrace(System.out);
                SystemInterface.die(-1);
                jq.UNREACHABLE();
                return;
            } else {
                int address = cc.findCatchBlock(ip, x_type);
                if (address != 0) {
                    // TODO: analyze the catch block to see if the backtrace is necessary.
                    if (true) {
                        //StackFrame sf = (StackFrame)Reflection.getfield_A(x, ClassLib.sun13.java.lang.Throwable._backtrace);
                        //sf.fillInStackTrace();
                    }
                    
                    // go to this catch block!
                    if (TRACE) SystemInterface.debugmsg("Jumping to catch block at "+jq.hex8(address));
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
                ip = Unsafe.peek(fp+4);
                fp = Unsafe.peek(fp);
            }
        }
    }
    
    public static void printStackTrace(Object backtrace, java.io.PrintWriter pw) {
        StackFrame sf = (StackFrame)backtrace;
        while (sf.next != null) {
            int/*CodeAddress*/ ip = sf.ip;
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
            String s;
            if (cc != null) {
                jq_Method m = cc.getMethod();
                int code_offset = ip - cc.getEntrypoint();
                if (m != null) {
                    Utf8 sourcefile = m.getDeclaringClass().getSourceFile();
                    int bc_index = cc.getBytecodeIndex(ip);
                    int line_num = m.getLineNumber(bc_index);
                    s = "\tat "+m+" ("+sourcefile+":"+line_num+" bc:"+bc_index+" off:"+jq.hex(code_offset)+")";
                } else {
                    s = "\tat <unknown cc> (start:"+jq.hex8(ip-code_offset)+" off:"+jq.hex(code_offset)+")";
                }
            } else {
                s = "\tat <unknown addr> (ip:"+jq.hex8(ip)+")";
            }
            pw.println(s.toCharArray());
            sf = sf.next;
        }
    }

    public static void printStackTrace(Object backtrace, java.io.PrintStream pw) {
        StackFrame sf = (StackFrame)backtrace;
        while (sf.next != null) {
            int/*CodeAddress*/ ip = sf.ip;
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
            String s;
            if (cc != null) {
                jq_Method m = cc.getMethod();
                int code_offset = ip - cc.getEntrypoint();
                if (m != null) {
                    Utf8 sourcefile = m.getDeclaringClass().getSourceFile();
                    int bc_index = cc.getBytecodeIndex(ip);
                    int line_num = m.getLineNumber(bc_index);
                    s = "\tat "+m+" ("+sourcefile+":"+line_num+" bc:"+bc_index+" off:"+jq.hex(code_offset)+")";
                } else {
                    s = "\tat <unknown cc> (start:"+jq.hex8(ip-code_offset)+" off:"+jq.hex(code_offset)+")";
                }
            } else {
                s = "\tat <unknown addr> (ip:"+jq.hex8(ip)+")";
            }
            pw.println(s.toCharArray());
            sf = sf.next;
        }
    }
    
    public static Object getStackTrace() {
        // stack traces are a linked list.
        int/*CodeAddress*/ ip = Unsafe.peek(Unsafe.EBP()+4);
        int/*StackAddress*/ fp = Unsafe.peek(Unsafe.EBP());
        StackFrame sf = new StackFrame(fp, ip);
        sf.fillInStackTrace();
        return sf;
    }
    
    static class StackFrame {
        int fp; // location of this stack frame
        int ip; // ip address
        StackFrame next; // next frame in linked list
        
        StackFrame(int/*StackAddress*/ fp, int/*CodeAddress*/ ip) {
            this.fp = fp; this.ip = ip;
        }
        
        void fillInStackTrace() {
            StackFrame dis = this;
            while (dis.fp != 0) {
                int/*CodeAddress*/ ip2 = Unsafe.peek(dis.fp+4);
                int/*StackAddress*/ fp2 = Unsafe.peek(dis.fp);
                dis.next = new StackFrame(fp2, ip2);
                dis = dis.next;
            }
        }
    }
    
    public static final jq_StaticMethod _athrow;
    public static final jq_StaticMethod _trap_handler;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/ExceptionDeliverer;");
        _athrow = k.getOrCreateStaticMethod("athrow", "(Ljava/lang/Throwable;)V");
        _trap_handler = k.getOrCreateStaticMethod("trap_handler", "(I)V");
    }
}

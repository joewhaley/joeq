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
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_StaticMethod;
import Clazz.jq_Method;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
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
            default: throw new InternalError("unknown hardware exception type: "+code);
        }
    }
    
    public abstract void deliverToStackFrame(jq_CompiledCode cc, Throwable x, int ip, int fp);
    
    public static void deliverToCurrentThread(Throwable x, int ip, int fp) {
        jq_Class x_type = (jq_Class)Unsafe.getTypeOf(x);
        if (TRACE) SystemInterface.debugmsg("Delivering exception of type "+x_type+" to ip="+jq.hex8(ip)+" fp="+jq.hex8(fp));
        for (;;) {
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ip);
            if (TRACE) SystemInterface.debugmsg("Checking compiled code "+cc);
            if (cc == null) {
                // reached the top!
                System.out.println("Exception in thread \""+Unsafe.getThreadBlock()+"\" "+x);
                x.printStackTrace(System.out);
                SystemInterface.die();
                jq.UNREACHABLE();
                return;
            } else {
                int address = cc.findCatchBlock(ip, x_type);
                if (address != 0) {
                    // go to this catch block!
                    if (TRACE) SystemInterface.debugmsg("Jumping to catch block at "+jq.hex8(address));
                    cc.deliverException(address, fp, x);
                    jq.UNREACHABLE();
                    return;
                }
                if (cc.getMethod().isSynchronized()) {
                    // TODO: need to perform monitorexit here
                }
                ip = Unsafe.peek(fp+4);
                fp = Unsafe.peek(fp);
            }
        }
    }
    
    public static void printStackTrace(Object backtrace, java.io.PrintWriter pw) {
        int/*CodeAddress*/[] ips = (int[])backtrace;
        for (int i=0; i<ips.length; ++i) {
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ips[i]);
            if (cc != null) {
                jq_Method m = cc.getMethod();
                String sourcefile = m.getDeclaringClass().getSourceFile();
                int code_offset = ips[i] - cc.getEntrypoint();
                int bc_index = cc.getBytecodeIndex(code_offset);
                int line_num = m.getLineNumber(bc_index);
                String s = "\tat "+m+" ("+sourcefile+":"+line_num+" bc:"+bc_index+" off:"+jq.hex(code_offset)+")";
                pw.println(s.toCharArray());
            } else {
                String s = "\tat <unknown> (ip:"+jq.hex8(ips[i])+")";
                pw.println(s.toCharArray());
            }
        }
    }

    public static void printStackTrace(Object backtrace, java.io.PrintStream pw) {
        int/*Address*/[] ips = (int[])backtrace;
        for (int i=0; i<ips.length; ++i) {
            jq_CompiledCode cc = CodeAllocator.getCodeContaining(ips[i]);
            if (cc != null) {
                jq_Method m = cc.getMethod();
                String sourcefile = m.getDeclaringClass().getSourceFile();
                int code_offset = ips[i] - cc.getEntrypoint();
                int bc_index = cc.getBytecodeIndex(ips[i]);
                int line_num = m.getLineNumber(bc_index);
                String s = "\tat "+m+" ("+sourcefile+":"+line_num+" bc:"+bc_index+" off:"+jq.hex(code_offset)+")";
                pw.println(s.toCharArray());
            } else {
                String s = "\tat <unknown> (ip:"+jq.hex8(ips[i])+")";
                pw.println(s.toCharArray());
            }
        }
    }
    
    public static Object getStackTrace() {
        int/*CodeAddress*/ ip = Unsafe.peek(Unsafe.EBP()+4);
        int/*StackAddress*/ fp = Unsafe.peek(Unsafe.EBP());
        StackWalker sw = new StackWalker(ip, fp);
        // once to count
        int size = 0;
        while (sw.hasNext()) {
            sw.next();
            ++size;
        }
        int/*CodeAddress*/[] ips = new int[size];
        sw = new StackWalker(ip, fp);
        for (int i=0; i<ips.length; ++i) {
            ips[i] = sw.getIP();
            sw.next();
        }
        return ips;
    }
    
    public static final jq_StaticMethod _athrow;
    public static final jq_StaticMethod _trap_handler;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/ExceptionDeliverer;");
        _athrow = k.getOrCreateStaticMethod("athrow", "(Ljava/lang/Throwable;)V");
        _trap_handler = k.getOrCreateStaticMethod("trap_handler", "(I)V");
    }
}

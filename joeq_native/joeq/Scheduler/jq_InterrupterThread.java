/*
 * jq_InterrupterThread.java
 *
 * Created on April 3, 2001, 8:07 PM
 *
 */

package Scheduler;

import Allocator.CodeAllocator;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_InstanceMethod;
import Main.jq;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class jq_InterrupterThread extends Thread {

    public static /*final*/ boolean TRACE = false;
    
    jq_InterrupterThread(jq_NativeThread other_nt) {
        this.other_nt = other_nt;
        if (TRACE) SystemInterface.debugmsg("Initialized timer interrupt for native thread "+other_nt);
        myself = ClassLibInterface.DEFAULT.getJQThread(this);
        myself.disableThreadSwitch();
        this.tid = SystemInterface.create_thread(_run.getDefaultCompiledVersion().getEntrypoint(), HeapAddress.addressOf(this));
        jq_NativeThread my_nt = new jq_NativeThread(myself);
        my_nt.getCodeAllocator().init();
        my_nt.getHeapAllocator().init();
        // start it up
        SystemInterface.resume_thread(this.tid);
    }
    
    // C thread handles
    private int tid, pid;
    private jq_NativeThread other_nt;
    private jq_Thread myself; // for convenience, so we don't have to call Reflection.getfield_A
    
    public static final int QUANTA = 50;
    
    public void run() {
        this.pid = SystemInterface.init_thread();
        Unsafe.setThreadBlock(this.myself);
        for (;;) {
            SystemInterface.msleep(QUANTA);
            other_nt.suspend();
            // The other thread may hold a system lock, so outputting any debug info here may lead to deadlock
            jq_Thread javaThread = other_nt.getCurrentJavaThread();
            if (javaThread.isThreadSwitchEnabled()) {
                if (TRACE) SystemInterface.debugmsg("TICK! "+other_nt+" Java Thread = "+javaThread);
                javaThread.disableThreadSwitch();
                jq_RegisterState regs = javaThread.getRegisterState();
                regs.ContextFlags = jq_RegisterState.CONTEXT_CONTROL |
                                    jq_RegisterState.CONTEXT_INTEGER |
                                    jq_RegisterState.CONTEXT_FLOATING_POINT;
                boolean b = other_nt.getContext(regs);
                if (!b) {
                    if (TRACE) SystemInterface.debugmsg("Failed to get thread context for "+other_nt);
                } else {
                    if (TRACE) SystemInterface.debugmsg(other_nt+" : "+javaThread+" ip="+regs.Eip.stringRep()+" sp="+regs.Esp.stringRep()+" cc="+CodeAllocator.getCodeContaining(regs.Eip));
                    // simulate a call to threadSwitch method
                    regs.Esp = (StackAddress) regs.Esp.offset(-4);
                    regs.Esp.poke(HeapAddress.addressOf(other_nt));
                    regs.Esp = (StackAddress) regs.Esp.offset(-4);
                    regs.Esp.poke(regs.Eip);
                    regs.Eip = jq_NativeThread._threadSwitch.getDefaultCompiledVersion().getEntrypoint();
                    regs.ContextFlags = jq_RegisterState.CONTEXT_CONTROL;
                    b = other_nt.setContext(regs);
                    if (!b) {
                        if (TRACE) SystemInterface.debugmsg("Failed to set thread context for "+other_nt);
                    } else {
                        if (TRACE) SystemInterface.debugmsg(other_nt+" : simulating a call to threadSwitch");
                    }
                }
            } else {
                // the current Java thread does not have thread switching enabled.
                //if (TRACE) SystemInterface.debugmsg(other_nt+" : "+javaThread+" Thread switch not enabled");
            }
            other_nt.resume();
        }
    }

    public static final jq_Class _class;
    public static final jq_InstanceMethod _run;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LScheduler/jq_InterrupterThread;");
        _run = _class.getOrCreateInstanceMethod("run", "()V");
    }
}

/*
 * jq_Thread.java
 *
 * Created on January 12, 2001, 1:07 AM
 *
 * @author  jwhaley
 * @version 
 */

package Scheduler;

import Allocator.ObjectLayout;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_CompiledCode;
import Clazz.jq_Reference;
import Clazz.jq_InstanceMethod;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Class;
import Clazz.jq_StaticMethod;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import UTF.Utf8;
import Util.AtomicCounter;
import jq;

public class jq_Thread implements ObjectLayout {

    private final jq_RegisterState registers;
    private volatile int thread_switch_enabled;
    private jq_NativeThread native_thread;
    private Throwable exception_object;
    private final Thread thread_object;
    jq_Thread next;
    private jq_CompiledCode entry_point;
    private boolean isDaemon;
    private boolean isDead;
    private final int thread_id;
    
    public static final int INITIAL_STACK_SIZE = 65536;

    public static AtomicCounter thread_id_factory = new AtomicCounter(1);
    
    public jq_Thread(Thread t) {
        this.thread_object = t;
        this.registers = new jq_RegisterState();
        this.thread_id = thread_id_factory.get() << THREAD_ID_SHIFT;
        jq.assert(this.thread_id > 0);
        jq.assert(this.thread_id < THREAD_ID_MASK);
    }

    public Thread getJavaLangThreadObject() { return thread_object; }
    public String toString() { return thread_object+" (sus: "+thread_switch_enabled+")"; }
    public jq_RegisterState getRegisterState() { return registers; }
    public jq_NativeThread getNativeThread() { return native_thread; }
    void setNativeThread(jq_NativeThread nt) { native_thread = nt; }
    public boolean isThreadSwitchEnabled() { return thread_switch_enabled == 0; }
    public void disableThreadSwitch() {
        ++thread_switch_enabled;
    }
    public void enableThreadSwitch() {
        --thread_switch_enabled;
    }

    public void init() {
        Thread t = thread_object;
        jq_Reference z = (jq_Reference)Reflection.getJQType(t.getClass());
        jq_InstanceMethod m = z.getVirtualMethod(new jq_NameAndDesc(Utf8.get("run"), Utf8.get("()V")));
        entry_point = m.getDefaultCompiledVersion();
        // initialize register state to start at start function
        this.registers.Esp = SystemInterface.allocate_stack(INITIAL_STACK_SIZE);
        this.registers.Eip = entry_point.getEntrypoint();
        // bogus return address
        this.registers.Esp -= 4;
        // arg to run(): t
        Unsafe.poke4(this.registers.Esp -= 4, Unsafe.addressOf(t));
        // return from run() directly to destroy()
        Unsafe.poke4(this.registers.Esp -= 4, _destroyCurrentThread.getDefaultCompiledVersion().getEntrypoint());
    }
    public void start() {
        jq_NativeThread.startJavaThread(this);
    }
    public void sleep(long millis) {
        // TODO:  for now, sleep just yields
        yield();
    }
    public void yield() {
        if (this != Unsafe.getThreadBlock()) {
            SystemInterface.debugmsg("Yield called on "+this+" from thread "+Unsafe.getThreadBlock());
            jq.UNREACHABLE();
        }
        // act like we received a timer tick
        this.disableThreadSwitch();
        // store the register state to make it look like we received a timer tick.
        ////registers.Ebp = Unsafe.peek(Unsafe.EBP());
        ////registers.Eip = Unsafe.peek(Unsafe.EBP()+4);
        ////registers.Esp = Unsafe.EBP() + 12; // fp + ret addr + 1 param
        int esp = Unsafe.ESP();
        registers.Esp = esp - 8; // room for object pointer and return address
        registers.Ebp = Unsafe.EBP();
        registers.ControlWord = 0x027f;
        registers.StatusWord = 0x4000;
        registers.TagWord = 0xffff;
        // other registers don't matter.
        this.getNativeThread().threadSwitch();
    }
    public void setPriority(int newPriority) { }
    public void stop(Object o) { }
    public void suspend() { }
    public void resume() { }
    public void interrupt() { }
    public boolean isInterrupted(boolean clear) { return false; }
    public boolean isAlive() { return !isDead; }
    public int countStackFrames() { return 0; }
    public int getThreadId() { return thread_id; }

    public static void destroyCurrentThread() {
        jq_Thread t = Unsafe.getThreadBlock();
        t.isDead = true;
        jq_NativeThread.endCurrentJavaThread();
        jq.UNREACHABLE();
    }
    
    public static final jq_Class _class;
    public static final jq_StaticMethod _destroyCurrentThread;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LScheduler/jq_Thread;");
        _destroyCurrentThread = _class.getOrCreateStaticMethod("destroyCurrentThread", "()V");
    }
}

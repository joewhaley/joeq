// jq_Thread.java, created Mon Feb  5 23:23:21 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Scheduler;

import Allocator.ObjectLayout;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Clazz.jq_DontAlign;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Main.jq;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import UTF.Utf8;
import Util.Assert;
import Util.AtomicCounter;

/**
 * A jq_Thread corresponds to a Java (lightweight) thread.
 * 
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class jq_Thread implements jq_DontAlign {

    // C code relies on this field being first.
    private final jq_RegisterState registers;
    // C code relies on this field being second.
    private volatile int thread_switch_enabled;
    // C code relies on this field being third.
    private jq_NativeThread native_thread;
    private Throwable exception_object;
    private final Thread thread_object;
    jq_Thread next;
    private jq_CompiledCode entry_point;
    private boolean isDaemon;
    private boolean hasStarted;
    private boolean isDead;
    boolean wasPreempted;
    private int priority;
    private volatile int isInterrupted;
    private final int thread_id;

    public static final int INITIAL_STACK_SIZE = 65536;

    public static AtomicCounter thread_id_factory = new AtomicCounter(1);

    public jq_Thread(Thread t) {
        this.thread_object = t;
        this.registers = new jq_RegisterState();
        this.thread_id = thread_id_factory.increment() << ObjectLayout.THREAD_ID_SHIFT;
        Assert._assert(this.thread_id > 0);
        Assert._assert(this.thread_id < ObjectLayout.THREAD_ID_MASK);
        this.isDead = true; // threads start as dead.
        this.priority = 5;
    }

    public Thread getJavaLangThreadObject() { return thread_object; }
    public String toString() { return thread_object + " (sus: " + thread_switch_enabled+")"; }
    public jq_RegisterState getRegisterState() { return registers; }
    public jq_NativeThread getNativeThread() { return native_thread; }
    void setNativeThread(jq_NativeThread nt) { native_thread = nt; }
    public boolean isThreadSwitchEnabled() { return thread_switch_enabled == 0; }
    public void disableThreadSwitch() {
        if (!jq.RunningNative) {
            ++thread_switch_enabled;
        } else {
            ((HeapAddress)HeapAddress.addressOf(this).offset(_thread_switch_enabled.getOffset())).atomicAdd(1);
        }
    }
    public void enableThreadSwitch() {
        if (!jq.RunningNative) {
            --thread_switch_enabled;
        } else {
            ((HeapAddress)HeapAddress.addressOf(this).offset(_thread_switch_enabled.getOffset())).atomicSub(1);
        }
    }
    
    public void init() {
        Thread t = thread_object;
        jq_Reference z = jq_Reference.getTypeOf(t);
        jq_InstanceMethod m = z.getVirtualMethod(new jq_NameAndDesc(Utf8.get("run"), Utf8.get("()V")));
        entry_point = m.getDefaultCompiledVersion();
        // initialize register state to start at start function
        this.registers.Esp = SystemInterface.allocate_stack(INITIAL_STACK_SIZE);
        this.registers.Eip = entry_point.getEntrypoint();
        // bogus return address
        this.registers.Esp = (StackAddress) this.registers.getEsp().offset(-CodeAddress.size());
        // arg to run()
        this.registers.Esp = (StackAddress) this.registers.getEsp().offset(-HeapAddress.size());
        this.registers.getEsp().poke(HeapAddress.addressOf(t));
        // return from run() directly to destroy()
        this.registers.Esp = (StackAddress) this.registers.getEsp().offset(-CodeAddress.size());
        this.registers.getEsp().poke(_destroyCurrentThread.getDefaultCompiledVersion().getEntrypoint());
    }
    public void start() {
        if (entry_point == null) {
            // java.lang.Thread objects in the boot image may not be initialized.
            this.init();
        }
        if (this.hasStarted)
            throw new IllegalThreadStateException();
        this.isDead = false;
        this.hasStarted = true;
        jq_NativeThread.startJavaThread(this);
    }
    long sleepUntil;
    public void sleep(long millis) throws InterruptedException {
        sleepUntil = System.currentTimeMillis() + millis;
        for (;;) {
            if (this.isInterrupted(true)) {
                throw new InterruptedException();
            }
            yield();
            if (System.currentTimeMillis() >= sleepUntil) {
                break;
            }
        }
    }
    public void yield() {
        if (this != Unsafe.getThreadBlock()) {
            SystemInterface.debugwriteln("Yield called on " + this + " from thread " + Unsafe.getThreadBlock());
            Assert.UNREACHABLE();
        }
        // act like we received a timer tick
        this.disableThreadSwitch();
        // store the register state to make it look like we received a timer tick.
        StackAddress esp = StackAddress.getStackPointer();
        // leave room for object pointer and return address
        registers.Esp = (StackAddress) esp.offset(-CodeAddress.size()-HeapAddress.size());
        registers.Ebp = StackAddress.getBasePointer();
        registers.ControlWord = 0x027f;
        registers.StatusWord = 0x4000;
        registers.TagWord = 0xffff;
        // other registers don't matter.
        this.getNativeThread().yieldCurrentThread();
    }
    public void yieldTo(jq_Thread t) {
        Assert._assert(this == Unsafe.getThreadBlock());
        // if that thread is in the thread queue for the current native
        // thread, we can yield to him easily.
        this.disableThreadSwitch();
        // thread switching for this native thread is disabled, so
        // Java threads cannot move from our local thread queue.
        if (t.getNativeThread() != this.getNativeThread()) {
            // TODO: temporarily increase priority of t (?)
            return;
        }

        // act like we received a timer tick.
        // store the register state to make it look like we received a timer tick.
        StackAddress esp = StackAddress.getStackPointer();
        // leave room for object pointer, arg, return address
        registers.Esp = (StackAddress) esp.offset(-CodeAddress.size()-HeapAddress.size()-HeapAddress.size());
        registers.Ebp = StackAddress.getBasePointer();
        registers.ControlWord = 0x027f;
        registers.StatusWord = 0x4000;
        registers.TagWord = 0xffff;
        // other registers don't matter.
        this.getNativeThread().yieldCurrentThreadTo(t);
    }
    public void setPriority(int newPriority) {
        this.priority = newPriority;
    }
    public int getPriority() {
        return this.priority;
    }
    public void stop(Object o) { }
    public void suspend() { }
    public void resume() { }
    public void interrupt() { this.isInterrupted = 1; }
    public boolean isInterrupted(boolean clear) {
        boolean isInt = this.isInterrupted != 0;
        if (clear && isInt) {
            //int res = Unsafe.atomicCas4(_isInterrupted.getAddress(), 1, 0);
            this.isInterrupted = 0;
        }
        return isInt;
    }
    public boolean isAlive() { return !isDead; }
    public boolean isDaemon() { return isDaemon; }
    public void setDaemon(boolean b) { isDaemon = b; }
    public int countStackFrames() { return 0; }
    public int getThreadId() { return thread_id; }

    public static void destroyCurrentThread() {
        jq_Thread t = Unsafe.getThreadBlock();
        t.isDead = true;
        jq_NativeThread.endCurrentJavaThread();
        Assert.UNREACHABLE();
    }

    public static final jq_Class _class;
    public static final jq_StaticMethod _destroyCurrentThread;
    public static final jq_InstanceField _thread_switch_enabled;
    public static final jq_InstanceField _isInterrupted;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LScheduler/jq_Thread;");
        _destroyCurrentThread = _class.getOrCreateStaticMethod("destroyCurrentThread", "()V");
        _thread_switch_enabled = _class.getOrCreateInstanceField("thread_switch_enabled", "I");
        _isInterrupted = _class.getOrCreateInstanceField("isInterrupted", "I");
    }
}

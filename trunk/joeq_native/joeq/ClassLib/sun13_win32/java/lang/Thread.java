/*
 * Thread.java
 *
 * Created on January 29, 2001, 10:21 AM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.sun13_win32.java.lang;

import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticMethod;
import Clazz.jq_Class;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Unsafe;
import Run_Time.Reflection;
import Scheduler.jq_Thread;
import jq;

public abstract class Thread {

    // additional fields
    public final jq_Thread jq_thread = null;
    
    // overridden constructors
    public static void __init__(java.lang.Thread dis) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            int n = Reflection.invokestatic_I(_nextThreadNum);
            Reflection.invokeinstance_V(_init, dis, null, null, "Thread-"+n);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.Runnable target) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            int n = Reflection.invokestatic_I(_nextThreadNum);
            Reflection.invokeinstance_V(_init, dis, null, target, "Thread-"+n);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.ThreadGroup group, java.lang.Runnable target) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            int n = Reflection.invokestatic_I(_nextThreadNum);
            Reflection.invokeinstance_V(_init, dis, group, target, "Thread-"+n);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.String name) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            Reflection.invokeinstance_V(_init, dis, null, null, name);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.ThreadGroup group, java.lang.String name) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            Reflection.invokeinstance_V(_init, dis, group, null, name);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.Runnable target, java.lang.String name) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            Reflection.invokeinstance_V(_init, dis, null, target, name);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    public static void __init__(java.lang.Thread dis, java.lang.ThreadGroup group, java.lang.Runnable target, java.lang.String name) {
        jq_Thread t = new jq_Thread(dis);
        Reflection.putfield_A(dis, _jq_thread, t);
        jq.assert(_class.isClsInitialized());
        try {
            Reflection.invokeinstance_V(_init, dis, group, target, name);
        } catch (java.lang.Error x) {
            throw x;
        } catch (java.lang.Throwable x) {
            jq.UNREACHABLE();
        }
        t.init();
    }
    
    // native method implementations
    private static void registerNatives(jq_Class clazz) {}
    public static java.lang.Thread currentThread(jq_Class clazz) { return Unsafe.getThreadBlock().getJavaLangThreadObject(); }
    public static void yield(jq_Class clazz) { Unsafe.getThreadBlock().yield(); }
    public static void sleep(jq_Class clazz, long millis) throws InterruptedException { Unsafe.getThreadBlock().sleep(millis); }
    public static /*synchronized*/ void start(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.start();
    }
    private static boolean isInterrupted(java.lang.Thread dis, boolean ClearInterrupted) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        return jq_thread.isInterrupted(ClearInterrupted);
    }
    public static final boolean isAlive(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        return jq_thread.isAlive();
    }
    public static int countStackFrames(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        return jq_thread.countStackFrames();
    }
    private static void setPriority0(java.lang.Thread dis, int newPriority) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.setPriority(newPriority);
    }
    private static void stop0(java.lang.Thread dis, java.lang.Object o) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.stop(o);
    }
    private static void suspend0(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.suspend();
    }
    private static void resume0(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.resume();
    }
    private static void interrupt0(java.lang.Thread dis) {
        jq_Thread jq_thread = (jq_Thread)Reflection.getfield_A(dis, _jq_thread);
        jq_thread.interrupt();
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Thread;");
    public static final jq_InstanceField _jq_thread = _class.getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
    public static final jq_InstanceField _target = _class.getOrCreateInstanceField("target", "Ljava/lang/Runnable;");
    public static final jq_StaticMethod _nextThreadNum = _class.getOrCreateStaticMethod("nextThreadNum", "()I");
    public static final jq_InstanceMethod _init = _class.getOrCreateInstanceMethod("init", "(Ljava/lang/ThreadGroup;Ljava/lang/Runnable;Ljava/lang/String;)V");
}

package Run_Time;

import Bootstrap.MethodInvocation;
import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Main.jq;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.SystemInterface.ExternalLink;
import Run_Time.SystemInterface.Library;

/**
 * @author John Whaley
 */
public class HighResolutionTimer {

    private static long counter_frequency = 0L;

    public static final void init() {
        if (QueryPerformanceFrequency != null) {
            counter_frequency = query_performance_frequency();
        }
    }

    public static final double now() {
        if (QueryPerformanceCounter != null) {
            if (counter_frequency == 0L) init();
            long l = query_performance_counter();
            return (double)l / counter_frequency;
        } else if (gettimeofday != null) {
            long l = get_time_of_day();
            int sec = (int) (l >> 32);
            int usec = (int) l;
            return ((double)sec) * 1000000 + usec;
        } else {
            return 0.;
        }
    }

    private static long query_performance_frequency() {
        try {
            CodeAddress a = QueryPerformanceFrequency.resolve();
            StackAddress b = StackAddress.alloca(8);
            Unsafe.pushArgA(b);
            Unsafe.getThreadBlock().disableThreadSwitch();
            byte rc = (byte) Unsafe.invoke(a);
            Unsafe.getThreadBlock().enableThreadSwitch();
            if (rc != 0) {
                // error occurred (?)
            }
            long v = b.peek8();
            return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }

    private static long query_performance_counter() {
        try {
            CodeAddress a = QueryPerformanceCounter.resolve();
            StackAddress b = StackAddress.alloca(8);
            Unsafe.pushArgA(b);
            Unsafe.getThreadBlock().disableThreadSwitch();
            byte rc = (byte) Unsafe.invoke(a);
            Unsafe.getThreadBlock().enableThreadSwitch();
            if (rc != 0) {
                // error occurred (?)
            }
            long v = b.peek8();
            return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    
    private static long get_time_of_day() {
        try {
            CodeAddress a = gettimeofday.resolve();
            StackAddress b = StackAddress.alloca(8);
            Unsafe.pushArgA(HeapAddress.getNull());
            Unsafe.pushArgA(b);
            Unsafe.getThreadBlock().disableThreadSwitch();
            int rc = (int) Unsafe.invoke(a);
            Unsafe.getThreadBlock().enableThreadSwitch();
            if (rc != 0) {
                // error occurred (?)
            }
            long v = b.peek8();
            return v;
        } catch (Throwable t) { jq.UNREACHABLE(); }
        return 0L;
    }
    
    static ExternalLink QueryPerformanceFrequency;
    static ExternalLink QueryPerformanceCounter;
    static ExternalLink gettimeofday;
    
    static {
        if (jq.RunningNative) boot();
        else if (jq.on_vm_startup != null) {
            jq_Class c = (jq_Class) Reflection.getJQType(HighResolutionTimer.class);
            jq_Method m = c.getDeclaredStaticMethod(new jq_NameAndDesc("boot", "()V"));
            MethodInvocation mi = new MethodInvocation(m, null);
            jq.on_vm_startup.add(mi);
        }
    }
    
    public static void boot() {
        Library kernel32 = SystemInterface.registerLibrary("kernel32");
        Library c = SystemInterface.registerLibrary("c");

        if (kernel32 != null) {
            QueryPerformanceFrequency = kernel32.resolve("QueryPerformanceFrequency");
            QueryPerformanceCounter = kernel32.resolve("QueryPerformanceCounter");
        } else {
            QueryPerformanceFrequency = QueryPerformanceCounter = null;
        }

        if (c != null) {
            gettimeofday = c.resolve("gettimeofday");
        } else {
            gettimeofday = null;
        }
    }
}

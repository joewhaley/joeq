package Run_Time;

/**
 * @author John Whaley
 */
public class HighResolutionTimer {

    private static long counter_frequency;

    public static final void init() {
        counter_frequency = SystemInterface.query_performance_frequency();
    }

    public static final double now() {
        long l = SystemInterface.query_performance_counter();
        return (double)l / counter_frequency;
    }

}

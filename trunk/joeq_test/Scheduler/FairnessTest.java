/*
 * Created on Oct 18, 2003
 */
package Scheduler;

/**
 * @author jwhaley
 */
public class FairnessTest {

    public static class BusyThread extends Thread {
        
        public static volatile boolean start = false;
        public static volatile boolean stop = false;
        
        BusyThread(int n) {
            this.n = n; // from 0 to (number of threads -1) on each priority
        }
        
        int n;
        volatile int counter;
        
        public void run() {
            while (!start) {
                Thread.yield();
            }
            while (!stop) {
                ++counter;//CPU time for each thread
            }
        }
    }
    
    // number of thread for each priority
    static int NUMBER = Integer.parseInt(System.getProperty("fairness.number", "5"));
    // amount of time for priming CPU to make CPU stable
    static int PRIME = Integer.parseInt(System.getProperty("fairness.prime", "0"));
    // amount of time to run experiment
    static int TIME = Integer.parseInt(System.getProperty("fairness.time", "20000"));
    // minimum priority to test
    static int min = Integer.parseInt(System.getProperty("fairness.min", "1"));
    // maximum priority to test
    static int max = Integer.parseInt(System.getProperty("fairness.max", "8"));
    
    public static void main(String[] args) throws Exception {
        
        // prime the JIT
        if (PRIME > 0) {
            BusyThread prime = new BusyThread(-1); //make one thread to prime the CPU
            prime.start();
            BusyThread.start = true;
            Thread.sleep(PRIME);
            BusyThread.stop = true;
            prime.stop();
            BusyThread.start = false;
        }
        
        Thread.currentThread().setPriority(Math.min(max + 1, Thread.MAX_PRIORITY)); //current thread (initial thread) priority becomes 9 because want to execute this most often
        
        BusyThread[][] threads = new BusyThread[max+1][];
        for (int i = min; i <= max; ++i) {
            threads[i] = new BusyThread[NUMBER];//NUMBER is a size of array
            for (int j = 0; j < NUMBER; ++j) {
                BusyThread t = new BusyThread(j);
                threads[i][j] = t;
                t.setPriority(i);
            }
        }
        int n = max - min + 1; //number of priority levels (=8)
        System.out.println("Created "+n+"*"+NUMBER+"="+(n*NUMBER)+" threads");
        
        for (int i = min; i <= max; ++i) {
            for (int j = 0; j < NUMBER; ++j) {
                Thread t = threads[i][j];
                t.start(); //start run method (yield)
            }
        }
        
        System.out.println("All threads running");
        
        BusyThread.start = false; //make sure it's yielding
        BusyThread.stop = false;
        
        BusyThread.start = true;  //start running
        Thread.sleep(TIME);  //let it run for 20 sec
        BusyThread.stop = true;
        
        System.out.println("All threads stopped");
        
        long[] total = new long[max+1];
        long overall = 0L; //sum of counter numbers for all the threads
        for (int i = min; i <= max; ++i) {
            for (int j = 0; j < NUMBER; ++j) {
                BusyThread t = threads[i][j];
                total[i] += t.counter;
            }
            overall += total[i];
        }
        System.out.println(overall+" ticks recorded");
        
        for (int i = min; i <= max; ++i) {
            double average = (double) total[i] / overall; //to give percentage of how much the priority got executed
            System.out.println("Total for priority "+i+": "+s2(100.*average)+"%");
            System.out.println("Breakdown within priority: ");
            double stddev = 0.;
            double average2 = (double)total[i] / NUMBER; //average count each thread suppose to have in this priority
            for (int j = 0; j < NUMBER; ++j) {
                BusyThread t = threads[i][j];
                System.out.print(s2(t.counter / average2)); //ideal is 1.0
                System.out.print("x\t");
                double diff = t.counter / average2 - 1.;
                stddev += diff * diff;
            }
            System.out.println();//goto next line
            stddev /= (max-min);  //computation for standard deviation
            stddev = Math.sqrt(stddev);
            System.out.println("Standard deviation: "+s2(stddev));  //0 is the best
            System.out.println();
        }
        
        for (int i = 0; i < jq_NativeThread.native_threads.length; ++i) {
            jq_InterrupterThread it = jq_NativeThread.native_threads[i].it;
            System.err.println("Native thread #"+i+": ");
            it.dumpStatistics();
        }
            
    }
    /** print numbers with 2 decimal places */
    static String s2(double d) {
        String s = Double.toString(d);
        int index = s.indexOf('.');
        if (index >= 0) {
            s = s.substring(0, Math.min(s.length(), index+3));
        }
        return s;
    }
}

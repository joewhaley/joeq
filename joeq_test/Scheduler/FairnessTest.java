/*
 * Created on Oct 18, 2003
 */
package Scheduler;

import joeq.Scheduler.jq_InterrupterThread;
import joeq.Scheduler.jq_NativeThread;

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
        
        //int nNative = args.length > 0 ? Integer.parseInt(args[0]) : 1;
        //calculateTheoretical(nNative, TIME);
        
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
        
        long startTime = System.currentTimeMillis();
        
        BusyThread.start = false; //make sure it's yielding
        BusyThread.stop = false;
        
        BusyThread.start = true;  //start counting; before that, threads are all waiting
        Thread.sleep(TIME);  //let it run for 20 sec
        BusyThread.stop = true;
        
        long endTime = System.currentTimeMillis();

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
        System.out.println(overall+" ticks recorded, time spent "+(endTime-startTime)+" ms");
        double efficiency = (double)overall / (double)(endTime-startTime);
        System.out.println("Efficiency: "+efficiency+" ticks per ms");
        
        for (int i = min; i <= max; ++i) {
            double average = (double) total[i] / overall; //to give percentage of how much the priority got executed
            System.out.println("Total for priority "+i+": "+s2(100.*average)+"%");  //. double 
            System.out.print("Breakdown within priority: ");
            double stddev = 0.;
            double average2 = (double)total[i] / NUMBER; //average count each thread suppose to have in this priority
            for (int j = 0; j < NUMBER; ++j) {
                BusyThread t = threads[i][j];
                double d = t.counter / average2;
                System.out.print(s2(d)); //ideal is 1.0
                System.out.print("\t");
                double diff = d - 1.;
                stddev = stddev + diff * diff;
            }
            System.out.println();//goto next line
            stddev = stddev / (NUMBER-1);  //computation for standard deviation
            stddev = Math.sqrt(stddev);
            System.out.println("Standard deviation: "+stddev);  //0 is the best
            System.out.println();
        }
        System.out.println("Overall priority distribution:"); 
        for (int i = min; i <= max; ++i) {
            double average = (double) total[i] / overall;
            System.out.println(s2(100.*average)+"%");
        }
        
        for (int i = 0; i < jq_NativeThread.native_threads.length; ++i) {
            jq_InterrupterThread it = jq_NativeThread.native_threads[i].it;
            if (it != null) {
                System.err.println("Native thread #"+i+": ");
                it.dumpStatistics();  // show if ticks were enabled (success) or disabled (fail)
            }
        }
            
    }
    /** print numbers with 2 decimal places */
    static String s2(double d) {
        String s = Double.toString(d);
        int index = s.indexOf('.');  //if 34.65, index=2  .3 then 0  -1.0 then 2
        if (index >= 0) {
            s = s.substring(0, Math.min(s.length(), index+3));  // . counts as 1 char
        }
        return s;
    }
    
    static long current = 0L;
    static long[] count;
    static long overall;
    static int quanta;
    
    static boolean handleQueue(int q, boolean hasMain) {
        if (q+1 >= min && q+1 <= max) {
            count[q+1]++;
            overall++;
            current += quanta;
            return true;
        }
        if (hasMain) {
            int mainPri = Math.min(max + 1, Thread.MAX_PRIORITY);
            if (q == mainPri-1) {
                current += 1L;
                return true;
            }
        }
        return false;
    }
    
    public static void calculateTheoretical(int nativeThreads, int milliseconds) {
        quanta = jq_InterrupterThread.QUANTA;
        int[] dist = jq_NativeThread.DISTRIBUTION;
        int relprime_value = jq_NativeThread.relatively_prime_value;
        count = new long[dist.length+1];
        for (int x = 0; x < nativeThreads; ++x) {
            System.out.println("Simulating native thread "+x+" for "+milliseconds+" ms");
            int distCounter = 0;
            current = 0L;
        outer:
            while (current < milliseconds) {
                distCounter += relprime_value;
                int max = dist[dist.length-1];
                while (distCounter >= max) {
                    distCounter -= max;
                }
                for (int i = 0; ; ++i) {
                    if (distCounter < dist[i]) {
                        if (handleQueue(i, x == 0)) continue outer;
                        int c = ((distCounter&1)==1) ? 1 : -1;
                        for (int j = i + c; j < dist.length && j >= 0; j += c) {
                            if (handleQueue(j, x == 0)) continue outer;
                        }
                        c = -c;
                        for (int j = i + c; j < dist.length && j >= 0; j += c) {
                            if (handleQueue(j, x == 0)) continue outer;
                        }
                        current += 1L;
                        continue outer;
                    }
                }
            }
        }
        
        for (int i = min; i <= max; ++i) {
            double average = (double) count[i] / overall; //to give percentage of how much the priority got executed
            System.out.println("Total for priority "+i+": "+s2(100.*average)+"%");  //. double 
        }
    }
}

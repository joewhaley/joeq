/*
 * Created on Dec 8, 2003
 */
package Compil3r.Analysis.IPSSA.Apps;

import java.util.Collection;
import java.util.Iterator;
import Clazz.jq_Method;
import Compil3r.Analysis.IPSSA.IPSSABuilder;
import Compil3r.Analysis.IPSSA.SSADefinition;
import Compil3r.Analysis.IPSSA.Utils.ReachabilityTrace;

/**
 * @author Vladimir Livshits
 * This is a sample application that prints all paths between two definitions.
 */
public abstract class ReachablityTracerApp extends IPSSABuilder.Application {
    private String _def1_str;
    private String _def2_str;
    private boolean _verbose = false;

    ReachablityTracerApp(String name, String[] args) {
        super(name, args);
    }
    
    /*
    public static void main(String[] argv) {
        if(argv.length < 2) usage(argv);
        //SimpleReachablityTracer tracer = new SimpleReachablityTracer();
        //tracer.printPath(argv[0], argv[1]);
    }*/
    
    private static void usage(String[] argv) {
        System.err.print("Invalid parameters: ");
        for(int i = 0; i < argv.length; i++) {
            System.err.print(argv[i] + " ");
        }
        System.err.println("");     
        
        System.exit(1);   
    }

    private void printPath(String def1_str, String def2_str) {
        SSADefinition def1 = SSADefinition.Helper.lookupDefinition(def1_str);
        SSADefinition def2 = SSADefinition.Helper.lookupDefinition(def2_str);
        
        if(def1 == null) {
            System.err.println("Can't find definition " + def1_str);
            System.exit(1);
        }
        if(def2 == null) {
            System.err.println("Can't find definition " + def2_str);
            System.exit(1);
        }
        jq_Method method1 = def1.getMethod();
        jq_Method method2 = def2.getMethod();
        if(_verbose) {
            System.err.println("Computing all paths between " + def1_str + " in " + method1 + " and " + def2_str + " in " + method2);
        }
        
        printPath(def1, def2);          
    }

    protected abstract void printPath(SSADefinition def1, SSADefinition def2);

    protected void parseParams(String[] argv) {
        if(argv == null) return;
        if(argv.length < 2) usage(argv);
        _def1_str = argv[0];
        _def2_str = argv[1];        
    }

    public void run() {
        printPath(_def1_str, _def2_str);        
    }
    
    /**
     * This one uses PA results directly.
     * */
    public static class PAReachablityTracerApp extends ReachablityTracerApp {
        PAReachablityTracerApp(String name, String[] args) {
            super(name, args);
        }

        protected void printPath(SSADefinition def1, SSADefinition def2) {
            // TODO: add this           
        }
    }
    
    /**
     * This is one that works on IPSSA.
     * */
    public static class IPSSAReachablityTracerApp extends ReachablityTracerApp {
        public IPSSAReachablityTracerApp(){
            this(null, null);
        }
        IPSSAReachablityTracerApp(String name, String[] args) {
            super(name, args);
        }

        protected void printPath(SSADefinition def1, SSADefinition def2) {
            Collection/*ReachabilityTrace*/ traces = ReachabilityTrace.Algorithms.collectReachabilityTraces(def1, def2);
            for(Iterator iter = traces.iterator(); iter.hasNext(); ) {
                ReachabilityTrace trace = (ReachabilityTrace)iter.next();
                
                System.out.println("\t" + trace.toString());
            }                       
        }
    }
}

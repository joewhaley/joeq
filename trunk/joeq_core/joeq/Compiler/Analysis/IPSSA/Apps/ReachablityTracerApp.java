/*
 * Created on Dec 8, 2003
 */
package Compil3r.Analysis.IPSSA.Apps;

import java.util.Collection;
import java.util.Iterator;

import org.sf.javabdd.BDD;

import Clazz.jq_Method;
import Compil3r.Analysis.IPSSA.IPSSABuilder;
import Compil3r.Analysis.IPSSA.SSADefinition;
import Compil3r.Analysis.IPSSA.SSALocation;
import Compil3r.Analysis.IPSSA.SSALocation.LocalLocation;
import Compil3r.Analysis.IPSSA.Utils.ReachabilityTrace;
import Util.Assert;

/**
 * @author Vladimir Livshits
 * This is a sample application that prints all paths between two definitions.
 * Use one of the subclasses that rely on different sources for def-use data.
 */
public abstract class ReachablityTracerApp extends IPSSABuilder.Application {
    private String _def1_str;
    private String _def2_str;
    private boolean _verbose = false;

    ReachablityTracerApp(IPSSABuilder builder, String name, String[] args) {
        super(builder, name, args);
    }    
    
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
        public PAReachablityTracerApp(){
             this(null, null, null);
        }
     
        PAReachablityTracerApp(IPSSABuilder builder, String name, String[] args) {
            super(builder, name, args);
        }

        protected void printPath(SSADefinition def1, SSADefinition def2) {
            Assert._assert(def1.getLocation() instanceof SSALocation.LocalLocation);
            Assert._assert(def2.getLocation() instanceof SSALocation.LocalLocation);
                        
            SSALocation.LocalLocation loc1 = (LocalLocation)def1.getLocation();                        
            String name1 = loc1.getName(def1.getMethod(), def1.getQuad());
            
            SSALocation.LocalLocation loc2 = (LocalLocation)def2.getLocation();
            String name2 = loc1.getName(def2.getMethod(), def2.getQuad());
            
            System.err.println(def1 + " : " + name1 + ", " + def2 + " : " + name2);                        
            
            // create the bdd for the variable(s)?
            BDD defBDD = null;
                         
            //_builder.getPAResults().defUseGraph(defBDD, true, System.out);           
        }
    }
    
    /**
     * This is one that works on IPSSA.
     * */
    public static class IPSSAReachablityTracerApp extends ReachablityTracerApp {
        public IPSSAReachablityTracerApp(){
            this(null, null, null);
        }
        IPSSAReachablityTracerApp(IPSSABuilder builder, String name, String[] args) {
            super(builder, name, args);
        }

        protected void printPath(SSADefinition def1, SSADefinition def2) {
            System.err.println("Calculating paths between " + def1 + " and " + def2);
            Collection/*ReachabilityTrace*/ traces = ReachabilityTrace.Algorithms.collectReachabilityTraces(def1, def2);
            for(Iterator iter = traces.iterator(); iter.hasNext(); ) {
                ReachabilityTrace trace = (ReachabilityTrace)iter.next();
                
                System.out.println("\t" + trace.toString());
            }                       
        }
    }
}

package Compil3r.Analysis.IPA;
import java.util.Iterator;

import org.sf.javabdd.BDD;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Clazz.jq_Method;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.IPSSA.IPSSABuilder;

public class PAQuery {
    public static class ParamAliasFinder extends IPSSABuilder.Application {
        public ParamAliasFinder() {
            super(null, null, null);
        }
        public ParamAliasFinder(IPSSABuilder builder, String name, String[] args) {
            super(builder, name, args);
        }
    
        protected void parseParams(String[] args) {}
        
        void visitMethod(jq_Method m){
            MethodSummary ms = MethodSummary.getSummary(m);
            if(getBuilder().skipMethod(m)) return;
            if(ms == null) return;
            System.out.println("Processing method " + m + ":\t" + ms.getNumOfParams());
            
            PAResults paResults = getBuilder().getPAResults();
            PA r = paResults.getPAResults();
            /*
            TypedBDDFactory.TypedBDD bdd = (TypedBDD)r.bdd.zero();
            for(int i = 0; i < ms.getNumOfParams(); i++) {
                ParamNode paramNode = ms.getParamNode(i);
                
                System.out.println("\tParam #"+ i + ": " + 
                    ( paramNode == null ? "<null>" : paramNode.toString_long() ) );
                
                int index = paResults.getVariableIndex(paramNode);
                if(index == -1) continue;
                BDD varBDD = r.V1.ithVar(index);
                bdd.orWith(varBDD);
            }
            System.out.println("BDD: " + bdd.toStringWithDomains());
            */
            
            // get formal arguments for the method
            BDD methodBDD = r.M.ithVar(paResults.getMethodIndex(m));
            BDD params = r.formal.relprod(methodBDD, r.Mset);
            //System.out.println("params: " + params.toStringWithDomains());
            TypedBDD contexts = (TypedBDD)params.relprod(r.vP, 
                r.V1.set().andWith(r.H1c.set()).andWith(r.H1.set()).andWith(r.Z.set()) );
            //System.out.println("contexts: \n" + paResults.toString(contexts, -1));
            //TypedBDD pointsTo = (TypedBDD)params.relprod(r.vP, r.V1cH1cset);
            //System.out.println("pointsTo: \n" + paResults.toString(pointsTo, -1));
            int i = 0;
            for(Iterator contextIter = contexts.iterator(); contextIter.hasNext(); i++) {
                TypedBDD context = (TypedBDD)contextIter.next();
                
                //System.out.println("context #" + i + ": " + context.toStringWithDomains());
                
                TypedBDD t = (TypedBDD)params.relprod(r.vP, r.V1.set());
                TypedBDD pointsTo = (TypedBDD)context.relprod(t, r.V1c.set().andWith(r.H1c.set()));
                t.free();
                t = (TypedBDD)pointsTo.exist(r.Z.set());
                //System.out.println(t.satCount() + ", " + pointsTo.satCount());
                int pointsToSize = (int)pointsTo.satCount(r.H1.set().and(r.Zset));
                int projSize     = (int)t.satCount( r.H1.set() ); 
                if(projSize < pointsToSize) {                
                    System.out.println("Potential aliasing in context #" + i + ": " + "pointsTo: \n" + 
                        pointsToSize + ": " + paResults.toString(pointsTo, -1));
                                               
                    //System.out.println("context #" + i + ": " + 
                    //    projSize + ": " + t.toStringWithDomains() + "\n");
                }
                t.free();
            }      
        }
        
        public void run() {
            for(Iterator iter = getBuilder().getCallGraph().getAllMethods().iterator(); iter.hasNext();) {
                jq_Method m = (jq_Method)iter.next();
            
                visitMethod(m);
            }
        }
    }
}
package Compil3r.Analysis.IPA;
import java.util.Iterator;

import org.sf.javabdd.BDD;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Util.Assert;

import Clazz.jq_Method;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ParamNode;
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
            //if(getBuilder().skipMethod(m)) return;
            
            MethodSummary ms = MethodSummary.getSummary(m);
            if(ms == null) return;
            if(ms.getNumOfParams() < 2) return;
            
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
            boolean printedInfo = false;
            long contextSize = (long)contexts.satCount(r.V1c.set());
            for(Iterator contextIter = contexts.iterator(); contextIter.hasNext(); i++) {
                TypedBDD context = (TypedBDD)contextIter.next();
                
                //System.out.println("context #" + i + ": " + context.toStringWithDomains());
                
                Assert._assert(r.vPfilter != null);
                TypedBDD t = (TypedBDD)r.vP.and(r.vPfilter.id());   // restrict by the type filter
                TypedBDD t2 = (TypedBDD)params.relprod(t, r.V1.set());
                t.free();
                t = t2;
                
                //TypedBDD t = (TypedBDD)params.relprod(r.vP, r.V1.set());
                TypedBDD pointsTo = (TypedBDD)context.relprod(t, r.V1c.set().andWith(r.H1c.set()));                
                t.free();
                
                t = (TypedBDD)pointsTo.exist(r.Z.set());
                //System.out.println(t.satCount() + ", " + pointsTo.satCount());
                int pointsToSize = (int)pointsTo.satCount(r.H1.set().and(r.Zset));
                int projSize     = (int)t.satCount( r.H1.set() ); 
                if(projSize < pointsToSize) {
                    if(!printedInfo) {
                        printMethodInfo(m, ms);
                        printedInfo = true;
                    }                
                    ProgramLocation loc = new ProgramLocation.BCProgramLocation(m, 0);
                    System.out.println("\tPotential aliasing in context #" + i + " calling " + m.toString() + " at " + 
                        loc.getSourceFile() + ":" + loc.getLineNumber());
                    if(contextSize > 5) {
                        System.out.println("\t\t(A total of " + contextSize + " contexts) \n");  
                        break;
                    }
                    /*": " + "pointsTo: \n" + pointsToSize + ": " + paResults.toString(pointsTo, -1)*/
                                               
                    //System.out.println("context #" + i + ": " + 
                    //    projSize + ": " + t.toStringWithDomains() + "\n");
                }
                t.free();
            }
        }
        
        void printMethodInfo(jq_Method m, MethodSummary ms) {
            System.out.println("Processing method " + m + ":\t[" + ms.getNumOfParams() + "]");
            for(int i = 0; i < ms.getNumOfParams(); i++) {
                ParamNode paramNode = ms.getParamNode(i);
                System.out.print("\t\t");
                System.out.println(paramNode == null ? "<null>" : paramNode.toString_long());
            }
            System.out.print("\n");
        }
        public void run() {
            for(Iterator iter = getBuilder().getCallGraph().getAllMethods().iterator(); iter.hasNext();) {
                jq_Method m = (jq_Method)iter.next();
            
                visitMethod(m);
            }
        }
    }
    
    /**
     * Application for finding and printing const-parameters.
     * */
    public static class ConstParameterFinder extends IPSSABuilder.Application {
        static final String NON_CONST_QUALIFIER = "non_const ";
        static final String CONST_QUALIFIER     = "const ";
        PAResults _paResults;
        PA _r;

        public ConstParameterFinder() {
            this(null, "ConstParameterFinder", null);
        }
        public ConstParameterFinder(IPSSABuilder builder, String name, String[] args) {
            super(builder, name, args);
        }

        void visitMethod(jq_Method m){            
            if(getBuilder().skipMethod(m)) return;
    
            MethodSummary ms = MethodSummary.getSummary(m);
            if(ms == null) return;
    
            _paResults = getBuilder().getPAResults();
            _r = _paResults.getPAResults();
            
            TypedBDD params = (TypedBDD)_r.formal.relprod(_r.M.ithVar(_paResults.getMethodIndex(m)), _r.Mset);
            
            System.out.print(m.toString() + "( ");
            for(Iterator paramIter = params.iterator(); paramIter.hasNext();) {
                TypedBDD param = (TypedBDD)((TypedBDD)paramIter.next()).exist(_r.Zset);                
                
                boolean isConst = isConst(param, m, true);
                
                System.out.print(isConst ? CONST_QUALIFIER : NON_CONST_QUALIFIER);
                System.out.print(param.toStringWithDomains() /*_paResults.toString(param, -1) */ + " ");
            }
            System.out.print(") \n");
        }

        boolean isConst(BDD param, jq_Method m, boolean recursive) {                                   
            // pointsTo is what this particular parameter may point to
            BDD pointsTo = _r.vP.restrict(param);                               // H1xH1cxF
            BDD methodBDD = _r.M.ithVar(_paResults.getMethodIndex(m));          // M
            
            BDD mods = null;
            if (!recursive) {               
                // these are the modified heap locations
                mods = _r.S.relprod(param, _r.V1set);                           // H1xH1cxFxV2xV1cxV2c
            } else {
                BDD method_plus_context0 = methodBDD.andWith(_r.V1c.set());
                BDD reachableVars = _paResults.getReachableVars(method_plus_context0); // V1xV1c
                reachableVars = reachableVars.exist(_r.V1c.set());
                reachableVars.or(param);
                System.err.println("reachableVars: " + _paResults.toString((TypedBDD)reachableVars, -1));
                
                BDD stores = _r.S.relprod(reachableVars, _r.V2set);             // V1xV1c x V1xV1cxFxV2xV2c = V1xV1cxF
                mods = stores.relprod(_r.vP, _r.V1set);                         // V1xV1cxF x V1xV1cxH1xH1c = H1xH1cxF
            }
            
            boolean result = mods.isZero();
            mods.free();
            
            return result;
        }

        protected void parseParams(String[] args) {}

        public void run() {
            for(Iterator iter = getBuilder().getCallGraph().getAllMethods().iterator(); iter.hasNext();) {
                jq_Method m = (jq_Method)iter.next();

                visitMethod(m);
            }
        }
    }
}
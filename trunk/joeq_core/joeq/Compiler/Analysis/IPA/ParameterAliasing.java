package joeq.Compiler.Analysis.IPA;

import java.util.Iterator;

import joeq.Class.jq_Method;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.ParamNode;
import joeq.Compiler.Analysis.IPSSA.IPSSABuilder;
import joeq.Util.Assert;

import org.sf.javabdd.BDD;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

class ParameterAliasing {
    /**
     * Finds parameter aliases under different constexts.
     * */
    public static class ParamAliasFinder extends IPSSABuilder.Application {
        PAResults _paResults = null;
        PA _r = null;
         
        public ParamAliasFinder() {
            super(null, null, null);
        }
        public ParamAliasFinder(IPSSABuilder builder, String name, String[] args) {
            super(builder, name, args);
        }
    
        protected void parseParams(String[] args) {}
        
        class ModifiableBoolean {
            boolean _value;
            
            ModifiableBoolean(boolean value){
                this._value = value;
            }
            boolean getValue() {return _value;}
            void setValue(boolean value) {this._value = value;}
        }
        
        void visitMethod(jq_Method m){
            //if(getBuilder().skipMethod(m)) return;
            System.out.println("Processing method " + m.toString());
            MethodSummary ms = MethodSummary.getSummary(m);
            if(ms == null) return;
            if(ms.getNumOfParams() < 2) return;
            
            _paResults = getBuilder().getPAResults();
            _r = _paResults.getPAResults();
 
            // get formal arguments for the method
            BDD methodBDD = _r.M.ithVar(_paResults.getMethodIndex(m));
            BDD params = _r.formal.relprod(methodBDD, _r.Mset);
            //System.out.println("params: " + params.toStringWithDomains());
            Assert._assert(_paResults.r.H1cset != null);
            Assert._assert(_r.H1.set() != null);
            Assert._assert(_r.Z.set() != null);
            TypedBDD contexts = (TypedBDD)params.relprod(_r.vP, 
                _r.V1.set().andWith(_r.H1cset).andWith(_r.H1.set()).andWith(_r.Z.set()) );
            //System.out.println("contexts: \n" + paResults.toString(contexts, -1));
            //TypedBDD pointsTo = (TypedBDD)params.relprod(r.vP, r.V1cH1cset);
            //System.out.println("pointsTo: \n" + paResults.toString(pointsTo, -1));
            int i = 0;
            ModifiableBoolean printedInfo = new ModifiableBoolean(false);
            long contextSize = (long)contexts.satCount(_r.V1cset);
            for(Iterator contextIter = contexts.iterator(); contextIter.hasNext(); i++) {
                TypedBDD context = (TypedBDD)contextIter.next();

                processContext(m, ms, params, context, contextSize, printedInfo, i);
            }
        }
        
        void processContext(jq_Method m, MethodSummary ms, BDD params, TypedBDD context, long contextSize, ModifiableBoolean printedInfo, int i){
            //System.out.println("context #" + i + ": " + context.toStringWithDomains());
                
            Assert._assert(_r.vPfilter != null);
            TypedBDD t = (TypedBDD)_r.vP.and(_r.vPfilter.id());   // restrict by the type filter
            TypedBDD t2 = (TypedBDD)params.relprod(t, _r.V1.set());
            t.free();
            t = t2;
                
            //TypedBDD t = (TypedBDD)params.relprod(r.vP, r.V1.set());
            TypedBDD pointsTo = (TypedBDD)context.relprod(t, _r.V1cset.andWith(_r.H1cset));                
            t.free();
                
            t = (TypedBDD)pointsTo.exist(_r.Z.set());
            //System.out.println(t.satCount() + ", " + pointsTo.satCount());
            int pointsToSize = (int)pointsTo.satCount(_r.H1.set().and(_r.Zset));
            int projSize     = (int)t.satCount( _r.H1.set() ); 
            if(projSize < pointsToSize) {
                if(!printedInfo.getValue()) {
                    printMethodInfo(m, ms);
                    printedInfo.setValue(true);
                }                
                ProgramLocation loc = new ProgramLocation.BCProgramLocation(m, 0);
                System.out.println("\tPotential aliasing in context #" + i + " calling " + m.toString() + " at " + 
                    loc.getSourceFile() + ":" + loc.getLineNumber());
                if(contextSize > 5) {
                    System.out.println("\t\t(A total of " + contextSize + " contexts) \n");  
                    return;
                }
            }
            t.free();
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
}
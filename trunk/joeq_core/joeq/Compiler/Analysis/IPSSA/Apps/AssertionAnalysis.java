package Compil3r.Analysis.IPSSA.Apps;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.Operator;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;

public class AssertionAnalysis implements ControlFlowGraphVisitor {
    public void visitCFG(ControlFlowGraph cfg) {
        System.out.println("Processing method " + cfg.getMethod().toString());
        for(QuadIterator iter = new QuadIterator(cfg); iter.hasNext(); ) {
            Quad quad = (Quad)iter.next();
            
            if(quad.getOperator() instanceof Operator.Invoke) {
                processCall(cfg.getMethod(), quad);
            }
        }                   
    }

    private void processCall(jq_Method method, Quad quad) {
        ProgramLocation loc = new QuadProgramLocation(method, quad);
        
        if(!loc.isSingleTarget()) {
            //System.err.println("Skipping a potentially virtual call");
            return;
        }
        
        jq_Method callee = loc.getTargetMethod();
        System.out.println("\tProcessing a call to " + callee);
        if(!isAssert(callee)) {
            return;
        }
    }
    
    public final static String ASSERT_NAME = "assert";    
    private static boolean isAssert(jq_Method callee) {
        String name = callee.getName().toString();         
        return name.indexOf(ASSERT_NAME) != -1;
    }
}
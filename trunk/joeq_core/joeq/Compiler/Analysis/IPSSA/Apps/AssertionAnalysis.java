package Compil3r.Analysis.IPSSA.Apps;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.ProgramLocation;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.ControlFlowGraphVisitor;
import Compil3r.Quad.LoadedCallGraph;
import Compil3r.Quad.Quad;
import Compil3r.Quad.Operator;
import Compil3r.Quad.QuadIterator;

class AssertionDetectionPass implements ControlFlowGraphVisitor {
    public void visitCFG(ControlFlowGraph cfg) {
        for(QuadIterator iter = new QuadIterator(cfg); iter.hasNext(); ) {
            Quad quad = (Quad)iter.next();
            
            if(quad.getOperator() instanceof Operator.Invoke) {
                processCall(cfg.getMethod(), quad);
            }
        }                   
    }

    private void processCall(jq_Method method, Quad quad) {
        ProgramLocation loc = LoadedCallGraph.mapCall(new QuadProgramLocation(method, quad));
        
        if(!loc.isSingleTarget()) {
            System.err.println("Skipping a potentially virtual call");
            return;
        }
        jq_Method callee = loc.getTargetMethod();
        if(!isAssert(callee)) {
            return;
        }
        System.out.println("Processing " + quad);
        
    }
    
    public final static String ASSERT_NAME = "assert";    
    private static boolean isAssert(jq_Method callee) {
        String name = callee.getName().toString();         
        return name.indexOf(ASSERT_NAME) != -1;
    }
}
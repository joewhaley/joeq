/*
 * Created on Sep 19, 2003
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package joeq.Compil3r.Analysis.IPSSA;

import java.io.PrintStream;

import joeq.Clazz.jq_Method;
import joeq.Compil3r.Quad.ControlFlowGraph;

/**
 * Goes over all bindings in a method.
 * @author Vladimir Livshits
 * @version $Id$
 * */
public abstract class SSABindingVisitor {
	public abstract void visit(SSABinding b);
    
    /**
     * Applies itself to all bindings in the CFG.
     * */
	public void visitCFG(ControlFlowGraph cfg) {
		jq_Method method = cfg.getMethod();
				
		for (SSAIterator.BindingIterator j=SSAProcInfo.retrieveQuery(method).getBindingIterator(method); j.hasNext(); ) {
			SSABinding b = j.nextBinding();
			b.accept(this);
		}				
	}
	
	public static class EmptySSABindingVisitor extends SSABindingVisitor {
		public EmptySSABindingVisitor(){}
		
		public void visit(SSABinding b){
			// do nothing
		}
	}
	
	public static class SSABindingPrinter extends SSABindingVisitor {
		protected PrintStream _out;
		SSABindingPrinter(PrintStream out){
			this._out = out;
		}
		public void visit(SSABinding b){
			_out.println(b.toString());
		}
	}
}


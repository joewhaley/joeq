/*
 * Created on Sep 19, 2003
 *
 * To change the template for this generated file go to
 * Window>Preferences>Java>Code Generation>Code and Comments
 */
package Compil3r.Analysis.IPSSA;

import java.io.PrintStream;

import Clazz.jq_Method;
import Compil3r.Quad.ControlFlowGraph;

public abstract class SSABindingVisitor {
	public abstract void visit(SSABinding b);
	public void visitCFG(ControlFlowGraph _cfg) {
		jq_Method method = _cfg.getMethod();
				
		for (SSAIterator.BindingIterator j=SSAProcInfo.retrieveQuery(method).getBindingIterator(method); j.hasNext(); ) {
			SSABinding b = j.nextBinding();
			b.accept(this);
		}				
	}
	
	public class EmptySSABindingVisitor extends SSABindingVisitor {
		public EmptySSABindingVisitor(){}
		
		public void visit(SSABinding b){
			// do nothing
		}
	}
	
	public class SSABindingPrinter extends SSABindingVisitor {
		protected PrintStream _out;
		SSABindingPrinter(PrintStream out){
			this._out = out;
		}
		public void visit(SSABinding b){
			_out.println(b.toString());
		}
	}
}


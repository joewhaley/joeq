/*
 * PrintCFG.java
 *
 * Created on March 16, 2002, 12:16 PM
 */

package Compil3r.Quad;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */
public class PrintCFG implements ControlFlowGraphVisitor {

    public java.io.PrintStream out = System.out;
    
    /** Creates new PrintCFG */
    public PrintCFG() {}

    /** Sets output stream. */
    public void setOut(java.io.PrintStream out) { this.out = out; }
    
    /** Prints full dump of the given CFG to the output stream. */
    public void visitCFG(ControlFlowGraph cfg) { out.println(cfg.fullDump()); }
    
}

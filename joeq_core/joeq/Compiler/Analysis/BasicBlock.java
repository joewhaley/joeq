/*
 * BasicBlock.java
 *
 * Created on April 22, 2001, 1:11 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Analysis;

public class BasicBlock {

    public final int id;
    final int start;
    int end;
    BasicBlock[] predecessors;
    BasicBlock[] successors;
    
    BasicBlock(int id, int start) {
        this.id = id; this.start = start;
    }
    
    public int getStart() { return start; }
    public int getEnd() { return end; }
    
    public int getNumberOfPredecessors() { return predecessors.length; }
    public int getNumberOfSuccessors() { return successors.length; }
    public BasicBlock getPredecessor(int i) { return predecessors[i]; }
    public BasicBlock getSuccessor(int i) { return successors[i]; }
    
    public String toString() { return "BB"+id+" ("+start+"-"+end+")"; }
}

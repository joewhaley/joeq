/*
 * JSRInfo.java
 *
 * Created on January 31, 2002, 9:54 PM
 */

package Compil3r.BytecodeAnalysis;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class JSRInfo {

    public BasicBlock entry_block;
    public BasicBlock exit_block;
    public boolean[] changedLocals;
    
    public JSRInfo(BasicBlock entry, BasicBlock exit, boolean[] changed) {
        this.entry_block = entry;
        this.exit_block = exit;
        this.changedLocals = changed;
    }
    
}

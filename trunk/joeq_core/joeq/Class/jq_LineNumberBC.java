/*
 * jq_LineNumberBC.java
 *
 * Created on January 22, 2001, 11:33 PM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

public class jq_LineNumberBC implements Comparable {

    private char startPC;
    private char lineNum;

    public jq_LineNumberBC(char startPC, char lineNum) {
        this.startPC = startPC;
        this.lineNum = lineNum;
    }

    public char getStartPC() { return startPC; }
    public char getLineNum() { return lineNum; }
    
    public int compareTo(jq_LineNumberBC that) {
        if (this.startPC == that.startPC) return 0;
        if (this.startPC < that.startPC) return -1;
        return 1;
    }
    public int compareTo(Object that) {
        return compareTo((jq_LineNumberBC)that);
    }
    public boolean equals(jq_LineNumberBC that) {
        return this.startPC == that.startPC;
    }
    public boolean equals(Object that) {
        return equals((jq_LineNumberBC)that);
    }
    
    public String toString() {
        return "(startPC="+(int)startPC+",lineNum="+(int)lineNum+")";
    }

}

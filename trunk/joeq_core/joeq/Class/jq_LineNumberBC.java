/*
 * jq_LineNumberBC.java
 *
 * Created on January 22, 2001, 11:33 PM
 *
 */

package Clazz;

/**
 * This class matches bytecode indices to line numbers.
 * It implements the Comparable interface; objects are compared based on their
 * starting bytecode indices.
 *
 * @author  John Whaley
 * @version 
 */
public class jq_LineNumberBC implements Comparable {

    /** Starting bytecode index, inclusive. */
    private char startPC;
    /** Corresponding line number. */
    private char lineNum;

    /** Constructs a new jq_LineNumberBC object with the given starting bytecode index and line number.
     *  The starting bytecode index is inclusive.
     *
     * @param startPC  starting bytecode index, inclusive
     * @param lineNum  corresponding line number
     */
    public jq_LineNumberBC(char startPC, char lineNum) {
        this.startPC = startPC;
        this.lineNum = lineNum;
    }

    /** Returns the start bytecode index.
     * @return  start bytecode index
     */
    public char getStartPC() { return startPC; }
    /** Returns the line number.
     * @return  line number
     */
    public char getLineNum() { return lineNum; }
    
    /** Compares this jq_LineNumberBC object to the given jq_LineNumberBC object.
     *  Comparisons are based on the start bytecode index value.
     *
     * @param that  object to compare against
     * @return  -1 if this is less than given, 0 if same as given, 1 if greater than given
     */
    public int compareTo(jq_LineNumberBC that) {
        if (this.startPC == that.startPC) return 0;
        if (this.startPC < that.startPC) return -1;
        return 1;
    }
    public int compareTo(java.lang.Object that) {
        return compareTo((jq_LineNumberBC)that);
    }
    /** Compares this jq_LineNumberBC object to the given jq_LineNumberBC object.
     *  Comparisons are based on the start bytecode index value.
     *
     * @param that  object to compare against
     * @return  true if objects are equal, false otherwise
     */
    public boolean equals(jq_LineNumberBC that) {
        return this.startPC == that.startPC;
    }
    public boolean equals(Object that) {
        return equals((jq_LineNumberBC)that);
    }
    
    /** Returns a string representation of this jq_LineNumberBC object.
     * @return  string representation
     */
    public String toString() {
        return "(startPC="+(int)startPC+",lineNum="+(int)lineNum+")";
    }

}

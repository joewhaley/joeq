/*
 * Quad.java
 *
 * Created on April 21, 2001, 11:06 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import jq;

public class Quad {

    private Operator operator;
    private Operand operand1, operand2, operand3, operand4;
    
    /** Creates new Quad */
    Quad(Operator operator) {
        this.operator = operator;
    }
    Quad(Operator operator, Operand operand1) {
        this.operator = operator; this.operand1 = operand1;
    }
    Quad(Operator operator, Operand operand1, Operand operand2) {
        this.operator = operator; this.operand1 = operand1; this.operand2 = operand2;
    }
    Quad(Operator operator, Operand operand1, Operand operand2, Operand operand3) {
        this.operator = operator; this.operand1 = operand1; this.operand2 = operand2; this.operand3 = operand3;
    }
    Quad(Operator operator, Operand operand1, Operand operand2, Operand operand3, Operand operand4) {
        this.operator = operator; this.operand1 = operand1; this.operand2 = operand2; this.operand3 = operand3; this.operand4 = operand4;
    }
    Operand getOp1() { return operand1; }
    Operand getOp2() { return operand2; }
    Operand getOp3() { return operand3; }
    Operand getOp4() { return operand4; }
    
    void accept(QuadVisitor qv) { this.operator.accept(this, qv); }
    
    public String toString() {
        StringBuffer s = new StringBuffer();
        s.append(jq.left(operator.toString(), 30));
        if (operand1 != null) {
            s.append(operand1.toString());
        } else
            s.append("      ");
        if (operand2 != null) {
	    if (operand1 != null) s.append(',');
	    s.append('\t');
            s.append(operand2.toString());
        } else
            return s.toString();
        if (operand3 != null) {
	    s.append(",\t");
            s.append(operand3.toString());
        } else
            return s.toString();
        if (operand4 != null) {
	    s.append(",\t");
            s.append(operand4.toString());
	}
        return s.toString();
    }
}

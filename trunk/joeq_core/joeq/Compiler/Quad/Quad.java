/*
 * Quad.java
 *
 * Created on April 21, 2001, 11:06 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.Quad;

import Operand.RegisterOperand;
import Operator.Return;
import Util.Templates.List;
import Util.Templates.UnmodifiableList;
import Clazz.jq_Class;
import jq;

public class Quad {

    /** The operator.  Operator objects are shared across all quads. */
    private Operator operator;
    /** The four operands.  Operands are quad-specific. */
    private Operand operand1, operand2, operand3, operand4;
    /** Id number for this quad.  THIS NUMBER HOLDS NO MEANING WHATSOEVER.  It is just used for printing. */
    private int id_number;
    
    /** Creates new Quad */
    Quad(int id, Operator operator) {
        this.id_number = id; this.operator = operator;
    }
    Quad(int id, Operator operator, Operand operand1) {
        this.id_number = id; this.operator = operator; this.operand1 = operand1;
    }
    Quad(int id, Operator operator, Operand operand1, Operand operand2) {
        this.id_number = id; this.operator = operator; this.operand1 = operand1; this.operand2 = operand2;
    }
    Quad(int id, Operator operator, Operand operand1, Operand operand2, Operand operand3) {
        this.id_number = id; this.operator = operator; this.operand1 = operand1; this.operand2 = operand2; this.operand3 = operand3;
    }
    Quad(int id, Operator operator, Operand operand1, Operand operand2, Operand operand3, Operand operand4) {
        this.id_number = id; this.operator = operator; this.operand1 = operand1; this.operand2 = operand2; this.operand3 = operand3; this.operand4 = operand4;
    }
    /** These are not intended to be used outside of the Compil3r.Quad package.
     * Instead, use the static accessor methods for each operator, e.g. Move.getDest(quad).
     */
    Operand getOp1() { return operand1; }
    Operand getOp2() { return operand2; }
    Operand getOp3() { return operand3; }
    Operand getOp4() { return operand4; }
    void setOp1(Operand op) { operand1 = op; }
    void setOp2(Operand op) { operand2 = op; }
    void setOp3(Operand op) { operand3 = op; }
    void setOp4(Operand op) { operand4 = op; }
    
    /** Return the operator for this quad. */
    public Operator getOperator() { return operator; }
    
    /** Accepts a quad visitor to this quad.  For the visitor pattern. */
    public void accept(QuadVisitor qv) { this.operator.accept(this, qv); }

    /** Returns the id number of this quad.  THIS NUMBER HOLDS NO MEANING WHATSOEVER.  It is just used for printing. */
    public int getID() { return id_number; }
    
    /** Returns a list of the types of exceptions that this quad can throw.
     * Note that types in this list are not exact, therefore subtypes of the
     * returned types may also be thrown. */
    public List.jq_Class getThrownExceptions() {
	if (operator == Return.THROW_A.INSTANCE) {
	    Operand op = Return.getSrc(this);
	    if (op instanceof RegisterOperand) {
		// use the operand type.
		return new UnmodifiableList.jq_Class((jq_Class)((RegisterOperand)op).getType());
	    }
	}
	return this.operator.getThrownExceptions();
    }

    /** Returns a list of the registers defined by this quad. */
    public List.RegisterOperand getDefinedRegisters() { return this.operator.getDefinedRegisters(this); }
    /** Returns a list of the registers used by this quad. */
    public List.RegisterOperand getUsedRegisters() { return this.operator.getUsedRegisters(this); }
    
    /** Returns a string representation of this quad. */
    public String toString() {
        StringBuffer s = new StringBuffer();
        s.append(jq.left(Integer.toString(id_number), 4));
        s.append(jq.left(operator.toString(), 30));
        if (operand1 == null) return s.toString();
        s.append(operand1.toString());
        if (operand2 == null) return s.toString();
        s.append(",\t");
        s.append(operand2.toString());
        if (operand3 == null) return s.toString();
        s.append(",\t");
        s.append(operand3.toString());
        if (operand4 == null) return s.toString();
        s.append(",\t");
        s.append(operand4.toString());
        return s.toString();
    }
}

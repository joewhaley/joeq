// Quad.java, created Fri Jan 11 16:42:38 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import Clazz.jq_Class;
import Compil3r.Quad.Operand.RegisterOperand;
import Compil3r.Quad.Operator.Return;
import Interpreter.QuadInterpreter;
import Util.Strings;
import Util.Templates.List;
import Util.Templates.UnmodifiableList;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
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
    public void setOp1(Operand op) { operand1 = op; }
    public void setOp2(Operand op) { operand2 = op; }
    public void setOp3(Operand op) { operand3 = op; }
    public void setOp4(Operand op) { operand4 = op; }

    public Quad copy(int id_number) {
        Operand op1 = (operand1!=null)?operand1.copy():null;
        Operand op2 = (operand2!=null)?operand2.copy():null;
        Operand op3 = (operand3!=null)?operand3.copy():null;
        Operand op4 = (operand4!=null)?operand4.copy():null;
        return new Quad(id_number, operator, op1, op2, op3, op4);
    }
    
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
	return Compil3r.CompilationState.DEFAULT.getThrownExceptions(this);
    }

    /** Returns a list of the registers defined by this quad. */
    public List.RegisterOperand getDefinedRegisters() { return this.operator.getDefinedRegisters(this); }
    /** Returns a list of the registers used by this quad. */
    public List.RegisterOperand getUsedRegisters() { return this.operator.getUsedRegisters(this); }
    
    /** Interprets this quad, modifying the given interpreter state. */
    public void interpret(QuadInterpreter s) { this.operator.interpret(this, s); }

    /** Returns a string representation of this quad. */
    public String toString() {
        StringBuffer s = new StringBuffer();
        s.append(Strings.left(Integer.toString(id_number), 4));
        s.append(Strings.left(operator.toString(), 24));
        if (operand1 == null) {
            if (operand2 == null) return s.toString();
            s.append("    \t");
        } else {
            s.append(operand1.toString());
            if (operand2 == null) return s.toString();
            s.append(",\t");
        }
        s.append(operand2.toString());
        if (operand3 == null) return s.toString();
        s.append(",\t");
        s.append(operand3.toString());
        if (operand4 == null) return s.toString();
        s.append(",\t");
        s.append(operand4.toString());
        return s.toString();
    }
    
    /** Returns a short string representation of this quad, without any operands. */
    public String toString_short() {
        StringBuffer s = new StringBuffer();
        s.append(Strings.left(Integer.toString(id_number), 4));
        s.append(operator.toString());
        return s.toString();
    }
    
    public static final boolean DETERMINISTIC = true;
    
    public int hashCode() {
        if (DETERMINISTIC) return getID();
        else return System.identityHashCode(this);
    }
    
}

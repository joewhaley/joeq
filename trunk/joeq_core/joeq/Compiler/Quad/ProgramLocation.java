/*
 * ProgramLocation.java
 *
 * Created on August 31, 2002, 12:50 PM
 */

package Compil3r.Quad;

import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Reference;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.CallTargets;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.Operator.Invoke;
import Main.jq;
import Util.HashCodeComparator;
import Util.SortedArraySet;

/**
 * This class combines a jq_Method with a Quad to represent a location in the code.
 * This is useful for interprocedural analysis.
 *
 * @author  John Whaley
 * @version $Id$
 */
public class ProgramLocation {

    private final jq_Method m; private final Quad q;
    public ProgramLocation(jq_Method m, Quad q) {
        this.m = m; this.q = q;
    }
    public jq_Method getMethod() { return m; }
    public Quad getQuad() { return q; }
    
    public int hashCode() { return (q==null)?-1:q.hashCode(); }
    public boolean equals(ProgramLocation that) { return this.q == that.q; }
    public boolean equals(Object o) { if (o instanceof ProgramLocation) return equals((ProgramLocation)o); return false; }
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append(m.getName());
        sb.append("() quad ");
        sb.append((q==null)?-1:q.getID());
        if (q.getOperator() instanceof Invoke) {
            sb.append(" => ");
            sb.append(Invoke.getMethod(q).getMethod().getName());
            sb.append("()");
        }
        return sb.toString();
    }

    private byte getInvocationType() {
        if (q.getOperator() instanceof Invoke.InvokeVirtual) {
            return BytecodeVisitor.INVOKE_VIRTUAL;
        } else if (q.getOperator() instanceof Invoke.InvokeStatic) {
            jq_Method target = Invoke.getMethod(q).getMethod();
            if (target instanceof jq_InstanceMethod)
                return BytecodeVisitor.INVOKE_SPECIAL;
            else
                return BytecodeVisitor.INVOKE_STATIC;
        } else {
            jq.Assert(q.getOperator() instanceof Invoke.InvokeInterface);
            return BytecodeVisitor.INVOKE_INTERFACE;
        }
    }

    public jq_Method getTargetMethod() {
        if (!(q.getOperator() instanceof Invoke)) return null;
        return Invoke.getMethod(q).getMethod();
    }

    public CallTargets getCallTargets() {
        if (!(q.getOperator() instanceof Invoke)) return null;
        jq_Method target = Invoke.getMethod(q).getMethod();
        byte type = getInvocationType();
        return CallTargets.getTargets(target.getDeclaringClass(), target, type, true);
    }

    public CallTargets getCallTargets(jq_Method target, Node n) {
        return getCallTargets(n.getDeclaredType(), n instanceof ConcreteTypeNode);
    }

    public CallTargets getCallTargets(jq_Reference klass, boolean exact) {
        if (!(q.getOperator() instanceof Invoke)) return null;
        jq_Method target = Invoke.getMethod(q).getMethod();
        byte type = getInvocationType();
        return CallTargets.getTargets(target.getDeclaringClass(), target, type, klass, exact, true);
    }

    public CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact) {
        if (!(q.getOperator() instanceof Invoke)) return null;
        jq_Method target = Invoke.getMethod(q).getMethod();
        byte type = getInvocationType();
        return CallTargets.getTargets(target.getDeclaringClass(), target, type, receiverTypes, exact, true);
    }

    public CallTargets getCallTargets(java.util.Set nodes) {
        if (!(q.getOperator() instanceof Invoke)) return null;
        Set exact_types = new SortedArraySet(HashCodeComparator.INSTANCE);
        Set notexact_types = new SortedArraySet(HashCodeComparator.INSTANCE);
        for (Iterator i=nodes.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            Set s = (n instanceof ConcreteTypeNode)?exact_types:notexact_types;
            if (n.getDeclaredType() != null)
                s.add(n.getDeclaredType());
        }
        if (notexact_types.isEmpty()) return getCallTargets(exact_types, true);
        if (exact_types.isEmpty()) return getCallTargets(notexact_types, false);
        CallTargets ct = getCallTargets(exact_types, true);
        ct.union(getCallTargets(notexact_types, false));
        return ct;
    }
}

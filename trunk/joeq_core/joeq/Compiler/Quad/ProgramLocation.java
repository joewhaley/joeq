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

import Compil3r.Quad.AndersenInterface.AndersenMethod;
import Compil3r.Quad.AndersenInterface.AndersenType;
import Compil3r.Quad.AndersenInterface.AndersenReference;

import Compil3r.Quad.SSAReader.SSAMethod;
import Compil3r.Quad.SSAReader.SSAType;
import Compil3r.Quad.SSAReader.SSAClass;

/**
 * This class combines a jq_Method with a Quad to represent a location in the code.
 * This is useful for interprocedural analysis.
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ProgramLocation {
    private final AndersenMethod m;
    public ProgramLocation(AndersenMethod m) {
        this.m = m;
    }
    public AndersenMethod getMethod() { return m; }
    public abstract int getNumParams();
    public abstract AndersenType getParamType(int i);
    
//    protected abstract byte getInvocationType();
    
    public abstract AndersenMethod getTargetMethod();
    
    public abstract CallTargets getCallTargets();

    public abstract CallTargets getCallTargets(AndersenReference klass, boolean exact);
    public abstract CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact);
    
    public CallTargets getCallTargets(AndersenMethod target, Node n) {
        return getCallTargets((AndersenReference)n.getDeclaredType(), n instanceof ConcreteTypeNode);
    }
    
    public CallTargets getCallTargets(java.util.Set nodes) {
        Set exact_types = SortedArraySet.FACTORY.makeSet(HashCodeComparator.INSTANCE);
        Set notexact_types = SortedArraySet.FACTORY.makeSet(HashCodeComparator.INSTANCE);

        for (Iterator i=nodes.iterator(); i.hasNext(); ) {
            Node n = (Node)i.next();
            Set s = (n instanceof ConcreteTypeNode)?exact_types:notexact_types;
            if (n.getDeclaredType() != null)
                s.add(n.getDeclaredType());
        }
        if (notexact_types.isEmpty()) return getCallTargets(exact_types, true);
        if (exact_types.isEmpty()) return getCallTargets(notexact_types, false);
        CallTargets ct = getCallTargets(exact_types, true);
        if (ct==null) return null;
        // bugfix - added 'ct=' (Daniel Wright)
        ct = ct.union(getCallTargets(notexact_types, false));
        return ct;
    }
    
    public static class QuadProgramLocation extends ProgramLocation {
        private final Quad q;
        public QuadProgramLocation(AndersenMethod m, Quad q) {
            super(m);
            this.q = q;
        }
        
        public AndersenMethod getTargetMethod() {
            if (!(q.getOperator() instanceof Invoke)) return null;
            return Invoke.getMethod(q).getMethod();
        }
        
        public int getNumParams() { return Invoke.getParamList(q).length(); }
        public AndersenType getParamType(int i) { return Invoke.getParamList(q).get(i).getType(); }
        
        public int hashCode() { return (q==null)?-1:q.hashCode(); }
        public boolean equals(QuadProgramLocation that) { return this.q == that.q; }
        public boolean equals(Object o) { if (o instanceof QuadProgramLocation) return equals((QuadProgramLocation)o); return false; }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append(super.m.getName());
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
        
        public CallTargets getCallTargets() {
            if (!(q.getOperator() instanceof Invoke)) return null;
            jq_Method target = Invoke.getMethod(q).getMethod();
            byte type = getInvocationType();
            return CallTargets.getTargets(target.getDeclaringClass(), target, type, true);
        }
        
        public CallTargets getCallTargets(AndersenReference klass, boolean exact) {
            if (!(q.getOperator() instanceof Invoke)) return null;
            jq_Method target = Invoke.getMethod(q).getMethod();
            byte type = getInvocationType();
            return CallTargets.getTargets(target.getDeclaringClass(), target, type, (jq_Reference)klass, exact, true);
        }
        
        public CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact) {
            if (!(q.getOperator() instanceof Invoke)) return null;
            jq_Method target = Invoke.getMethod(q).getMethod();
            byte type = getInvocationType();
            return CallTargets.getTargets(target.getDeclaringClass(), target, type, receiverTypes, exact, true);
        }
    }
    
    public static class SSAProgramLocation extends ProgramLocation {
        final int identifier; // for .equals() comparison - should identify program location
        SSAMethod targetMethod;
        
        public SSAProgramLocation(int identifier, AndersenMethod m, SSAMethod targetMethod) {
            super(m);
            this.identifier = identifier;
            this.targetMethod = targetMethod;
        }
        
        public AndersenMethod getTargetMethod() { return targetMethod; }
        public int getNumParams() { return targetMethod.getNumParams(); }
        public AndersenType getParamType(int i) { return targetMethod.getParamType(i); }

        public int hashCode() { return identifier; }
        public boolean equals(SSAProgramLocation that) { return this.identifier == that.identifier; }
        public boolean equals(Object o) { if (o instanceof SSAProgramLocation) return equals((SSAProgramLocation)o); return false; }
        public String toString() {
            String s = super.m.getName()+"() invocation "+identifier;
            if (targetMethod != null)
                s += " => "+targetMethod.getName()+"()";
            return s;
        }
        
        
        public CallTargets getCallTargets() {
            return targetMethod.getCallTargets(targetMethod.getDeclaringClass(), false);
        }
        public CallTargets getCallTargets(AndersenReference klass, boolean exact) {
            SSAType ssaType = (SSAType)klass;

            // todo: handle other types (non-classes)
            return targetMethod.getCallTargets(ssaType.getSSAClass(), exact);
        }
        public CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact) {
            CallTargets ct = CallTargets.NoCallTarget.INSTANCE;
            
            for (Iterator i = receiverTypes.iterator(); i.hasNext(); ) {
                ct = ct.union(getCallTargets((AndersenReference)i.next(), exact));
            }
            
            return ct;
        }
    }
}

// ProgramLocation.java, created Sun Sep  1 17:38:25 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Map;
import java.util.StringTokenizer;

import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.Bytecodes;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.ControlFlowGraph;
import Compil3r.Quad.Quad;
import Compil3r.Quad.QuadIterator;
import Compil3r.Quad.Operator.Invoke;
import UTF.Utf8;
import Util.Assert;
import Util.IO.ByteSequence;

/**
 * This class provides a general mechanism to describe a location in the code,
 * independent of IR type.  It combines a method and a location within that
 * method.  This is useful for interprocedural analysis, among other things.
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class ProgramLocation {
    
    /** The method of this location. **/
    protected final jq_Method m;
    
    protected ProgramLocation(jq_Method m) {
        this.m = m;
    }
    
    public jq_Class getContainingClass() {
        return m.getDeclaringClass();
    }
    
    public jq_Method getMethod() {
        return m;
    }
    
    public Utf8 getSourceFile() {
        return getContainingClass().getSourceFile();
    }
    
    public int getLineNumber() {
        return m.getLineNumber(getBytecodeIndex());
    }
    
    public abstract int getID();
    public abstract int getBytecodeIndex();
    public abstract jq_Type getResultType();
    
    public abstract void write(DataOutput out) throws IOException;
    
    public abstract boolean isCall();
    public abstract jq_Method getTargetMethod();
    public abstract jq_Type[] getParamTypes();
    public abstract jq_Type getReturnType();
    public abstract boolean isSingleTarget();
    public abstract boolean isInterfaceCall();
    public abstract byte getInvocationType();
    
    public static class QuadProgramLocation extends ProgramLocation {
        private final Quad q;
        public QuadProgramLocation(jq_Method m, Quad q) {
            super(m);
            this.q = q;
        }
        
        public Quad getQuad() {
            return q;
        }
        
        public int getID() {
            return q.getID();
        }
        
        public int getBytecodeIndex() {
            Map map = CodeCache.getBCMap((jq_Method) super.m);
            Integer i = (Integer) map.get(q);
            return i.intValue();
        }
        
        public jq_Type getResultType() {
            return q.getDefinedRegisters().getRegisterOperand(0).getType();
        }
        
        public boolean isCall() {
            return q.getOperator() instanceof Invoke;
        }
        
        public jq_Method getTargetMethod() {
            Assert._assert(isCall());
            return Invoke.getMethod(q).getMethod();
        }
        
        public jq_Type[] getParamTypes() {
            Assert._assert(isCall());
            jq_Type[] t = Invoke.getMethod(q).getMethod().getParamTypes();
            if (t.length != Invoke.getParamList(q).length()) {
                t = new jq_Type[Invoke.getParamList(q).length()];
                for (int i=0; i<t.length; ++i) {
                    t[i] = Invoke.getParamList(q).get(i).getType();
                }
            }
            return t;
        }
        
        public jq_Type getReturnType() {
            Assert._assert(isCall());
            return Invoke.getMethod(q).getMethod().getReturnType();
        }
        
        public boolean isSingleTarget() {
            if (isInterfaceCall()) return false;
            if (!((Invoke) q.getOperator()).isVirtual()) return true;
            jq_InstanceMethod target = (jq_InstanceMethod) Invoke.getMethod(q).getMethod();
            target.getDeclaringClass().load();
            if (target.getDeclaringClass().isFinal()) return true;
            target.getDeclaringClass().prepare();
            if (!target.isLoaded()) {
                target = target.resolve1();
                if (!target.isLoaded()) {
                    // bad target method!
                    return false;
                }
                Invoke.getMethod(q).setMethod(target);
            }
            if (!target.isVirtual()) return true;
            return false;
        }
        
        public boolean isInterfaceCall() {
            return q.getOperator() instanceof Invoke.InvokeInterface;
        }
        
        public int hashCode() {
            return (q==null)?-1:q.hashCode();
        }
        public boolean equals(QuadProgramLocation that) {
            return this.q == that.q;
        }
        public boolean equals(Object o) {
            if (o instanceof QuadProgramLocation)
                return equals((QuadProgramLocation)o);
            return false;
        }
        
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
        
        public byte getInvocationType() {
            if (q.getOperator() instanceof Invoke.InvokeVirtual) {
                return BytecodeVisitor.INVOKE_VIRTUAL;
            } else if (q.getOperator() instanceof Invoke.InvokeStatic) {
                jq_Method target = Invoke.getMethod(q).getMethod();
                if (target instanceof jq_InstanceMethod)
                    return BytecodeVisitor.INVOKE_SPECIAL;
                else
                    return BytecodeVisitor.INVOKE_STATIC;
            } else {
                Assert._assert(q.getOperator() instanceof Invoke.InvokeInterface);
                return BytecodeVisitor.INVOKE_INTERFACE;
            }
        }
        
        /*
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
        */
        
        public void write(DataOutput out) throws IOException {
            m.writeDesc(out);
            out.writeBytes(" quad "+q.getID());
        }
        
    }
    
    public static class BCProgramLocation extends ProgramLocation {
        final int bcIndex;
        
        public BCProgramLocation(jq_Method m, int bcIndex) {
            super(m);
            this.bcIndex = bcIndex;
        }
        
        public int getID() {
            return bcIndex;
        }
        
        public int getBytecodeIndex() {
            return bcIndex;
        }
        
        public byte getBytecode() {
            byte[] bc = ((jq_Method) super.m).getBytecode();
            return bc[bcIndex];
        }
        
        public jq_Type getResultType() {
            ByteSequence bs = new ByteSequence(m.getBytecode(), bcIndex, 8);
            try {
                Bytecodes.Instruction i = Bytecodes.Instruction.readInstruction(getContainingClass().getCP(), bs);
                if (!(i instanceof Bytecodes.TypedInstruction)) return null;
                return ((Bytecodes.TypedInstruction)i).getType();
            } catch (IOException x) {
                Assert.UNREACHABLE();
                return null;
            }
        }
        
        public boolean isCall() {
            switch (getBytecode()) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    return true;
                default:
                    return false;
            }
        }
        
        public jq_Method getTargetMethod() {
            jq_Class clazz = ((jq_Method) super.m).getDeclaringClass();
            byte[] bc = ((jq_Method) super.m).getBytecode();
            char cpi = Util.Convert.twoBytesToChar(bc, bcIndex+1);
            switch (bc[bcIndex]) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                    return clazz.getCPasInstanceMethod(cpi);
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                    return clazz.getCPasStaticMethod(cpi);
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    return Run_Time.Arrays._multinewarray;
                default:
                    return null;
            }
        }
        
        public jq_Type[] getParamTypes() {
            return getTargetMethod().getParamTypes();
        }
        
        public jq_Type getReturnType() {
            return getTargetMethod().getReturnType();
        }
        
        public boolean isSingleTarget() {
            switch (getBytecode()) {
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    return true;
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                default:
                    return false;
            }
        }
        
        public boolean isInterfaceCall() {
            return getBytecode() == jq_ClassFileConstants.jbc_INVOKEINTERFACE;
        }

        public int hashCode() {
            return super.m.hashCode() ^ bcIndex;
        }
        public boolean equals(BCProgramLocation that) {
            return this.bcIndex == that.bcIndex && super.m == that.m;
        }
        public boolean equals(Object o) {
            if (o instanceof BCProgramLocation)
                return equals((BCProgramLocation) o);
            return false;
        }
        public String toString() {
            String s = super.m.getName()+"() @ "+bcIndex;
            return s;
        }
        
        public byte getInvocationType() {
            switch (getBytecode()) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                    return BytecodeVisitor.INVOKE_VIRTUAL;
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                    return BytecodeVisitor.INVOKE_SPECIAL;
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                    return BytecodeVisitor.INVOKE_INTERFACE;
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    return BytecodeVisitor.INVOKE_STATIC;
                default:
                    return -1;
            }
        }
        /*
        public CallTargets getCallTargets() {
            jq_Class clazz = ((jq_Method) super.m).getDeclaringClass();
            byte[] bc = ((jq_Method) super.m).getBytecode();
            if (bc == null || bcIndex < 0 || bcIndex+2 >= bc.length) return null;
            char cpi = Util.Convert.twoBytesToChar(bc, bcIndex+1);
            byte type;
            jq_Method method;
            switch (bc[bcIndex]) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                    type = BytecodeVisitor.INVOKE_VIRTUAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                    type = BytecodeVisitor.INVOKE_SPECIAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                    method = clazz.getCPasInstanceMethod(cpi);
                    type = BytecodeVisitor.INVOKE_INTERFACE;
                    break;
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                    method = clazz.getCPasStaticMethod(cpi);
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    method = Run_Time.Arrays._multinewarray;
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                default:
                    return null;
            }
            return CallTargets.getTargets(clazz, method, type, true);
        }
        public CallTargets getCallTargets(AndersenReference klass, boolean exact) {
            jq_Class clazz = ((jq_Method) super.m).getDeclaringClass();
            byte[] bc = ((jq_Method) super.m).getBytecode();
            if (bc == null || bcIndex < 0 || bcIndex+2 >= bc.length) return null;
            char cpi = Util.Convert.twoBytesToChar(bc, bcIndex+1);
            byte type;
            jq_Method method;
            switch (bc[bcIndex]) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                    type = BytecodeVisitor.INVOKE_VIRTUAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                    type = BytecodeVisitor.INVOKE_SPECIAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                    method = clazz.getCPasInstanceMethod(cpi);
                    type = BytecodeVisitor.INVOKE_INTERFACE;
                    break;
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                    method = clazz.getCPasStaticMethod(cpi);
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    method = Run_Time.Arrays._multinewarray;
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                default:
                    return null;
            }
            return CallTargets.getTargets(clazz, method, type, (jq_Reference) klass, exact, true);
        }
        public CallTargets getCallTargets(java.util.Set receiverTypes, boolean exact) {
            jq_Class clazz = ((jq_Method) super.m).getDeclaringClass();
            byte[] bc = ((jq_Method) super.m).getBytecode();
            if (bc == null || bcIndex < 0 || bcIndex+2 >= bc.length) return null;
            char cpi = Util.Convert.twoBytesToChar(bc, bcIndex+1);
            byte type;
            jq_Method method;
            switch (bc[bcIndex]) {
                case (byte) jq_ClassFileConstants.jbc_INVOKEVIRTUAL:
                    type = BytecodeVisitor.INVOKE_VIRTUAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKESPECIAL:
                    type = BytecodeVisitor.INVOKE_SPECIAL;
                    // fallthrough
                case (byte) jq_ClassFileConstants.jbc_INVOKEINTERFACE:
                    method = clazz.getCPasInstanceMethod(cpi);
                    type = BytecodeVisitor.INVOKE_INTERFACE;
                    break;
                case (byte) jq_ClassFileConstants.jbc_INVOKESTATIC:
                    method = clazz.getCPasStaticMethod(cpi);
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                case (byte) jq_ClassFileConstants.jbc_MULTIANEWARRAY:
                    method = Run_Time.Arrays._multinewarray;
                    type = BytecodeVisitor.INVOKE_STATIC;
                    break;
                default:
                    return null;
            }
            return CallTargets.getTargets(clazz, method, type, receiverTypes, exact, true);
        }
        */
        
        public void write(DataOutput out) throws IOException {
            m.writeDesc(out);
            out.writeBytes(" bc "+bcIndex);
        }
        
    }
    
    public static ProgramLocation read(StringTokenizer st) {
        jq_Method m = (jq_Method) jq_Method.read(st);
        String s = st.nextToken();
        int id = Integer.parseInt(st.nextToken());
        if (s.equals("bc")) {
            return new BCProgramLocation(m, id);
        }
        if (s.equals("quad")) {
            if (m.getBytecode() == null) return null;
            ControlFlowGraph cfg = CodeCache.getCode(m);
            for (QuadIterator i = new QuadIterator(cfg); i.hasNext(); ) {
                Quad q = i.nextQuad();
                if (q.getID() == id) return new QuadProgramLocation(m, q);
            }
        }
        return null;
    }
    
}
/*
 * Interpreter.java
 *
 * Created on January 15, 2001, 8:34 PM
 *
 */

package Interpreter;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Clazz.jq_Method;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticField;
import Clazz.jq_StaticMethod;
import Clazz.jq_TryCatchBC;
import Clazz.jq_Type;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Main.jq;
import Memory.Address;
import Memory.HeapAddress;
import Run_Time.Reflection;
import Run_Time.Unsafe;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class BytecodeInterpreter {

    public static /*final*/ boolean ALWAYS_TRACE = false;
    
    /** Creates new Interpreter */
    public BytecodeInterpreter(VMInterface vm, State state) {
        this.vm = vm; this.state = state;
    }

    // create an Interpreter.State and call invokeMethod(m, state)
    public abstract Object invokeMethod(jq_Method m) throws Throwable;
    public abstract Object invokeUnsafeMethod(jq_Method m) throws Throwable;
    
    // callee == null -> call compiled version
    public Object invokeMethod(jq_Method m, State callee) throws Throwable {
        //Run_Time.SystemInterface.debugwriteln("Invoking method "+m);
        jq_Class k = m.getDeclaringClass();
        jq.Assert(k.isClsInitialized());
        jq.Assert(m.getBytecode() != null);
        jq_Type[] paramTypes = m.getParamTypes();
        Object[] params = new Object[paramTypes.length];
        for (int i=paramTypes.length-1; i>=0; --i) {
            jq_Type t = paramTypes[i];
            if (t.isPrimitiveType()) {
                if (t == jq_Primitive.LONG) {
                    params[i] = new Long(state.pop_L());
                } else if (t == jq_Primitive.FLOAT) {
                    params[i] = new Float(state.pop_F());
                } else if (t == jq_Primitive.DOUBLE) {
                    params[i] = new Double(state.pop_D());
                } else {
                    params[i] = new Integer(state.pop_I());
                }
            } else {
                params[i] = state.pop_A();
            }
            //System.out.println("Param "+i+": "+params[i]);
        }
        for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
            jq_Type t = paramTypes[i];
            if (t.isPrimitiveType()) {
                if (t == jq_Primitive.LONG) {
                    long v = ((Long)params[i]).longValue();
                    if (callee == null) {
                        Unsafe.pushArg((int)(v>>32)); // hi
                        Unsafe.pushArg((int)v);       // lo
                    } else callee.setLocal_L(j, v);
                    ++j;
                } else if (t == jq_Primitive.FLOAT) {
                    float v = ((Float)params[i]).floatValue();
                    if (callee == null) {
                        Unsafe.pushArg(Float.floatToRawIntBits(v));
                    } else callee.setLocal_F(j, v);
                } else if (t == jq_Primitive.DOUBLE) {
                    long v = Double.doubleToRawLongBits(((Double)params[i]).doubleValue());
                    if (callee == null) {
                        Unsafe.pushArg((int)(v>>32)); // hi
                        Unsafe.pushArg((int)v);       // lo
                    } else callee.setLocal_D(j, Double.longBitsToDouble(v));
                    ++j;
                } else {
                    int v = ((Integer)params[i]).intValue();
                    if (callee == null) {
                        Unsafe.pushArg(v);
                    } else callee.setLocal_I(j, v);
                }
            } else {
                Object v = params[i];
                if (callee == null) {
                    Unsafe.pushArgA(HeapAddress.addressOf(v));
                } else callee.setLocal_A(j, v);
            }
        }
        if (callee == null) {
            jq_Type returnType = m.getReturnType();
            if (returnType.isReferenceType()) {
                Address result = Unsafe.invokeA(m.getDefaultCompiledVersion().getEntrypoint());
                if (returnType.isAddressType()) return result;
                return ((HeapAddress) result).asObject();
            }
            long result = Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
            if (returnType == jq_Primitive.VOID)
                return null;
            else if (returnType == jq_Primitive.LONG)
                return new Long(result);
            else if (returnType == jq_Primitive.FLOAT)
                return new Float(Float.intBitsToFloat((int)(result)));
            else if (returnType == jq_Primitive.DOUBLE)
                return new Double(Double.longBitsToDouble(result));
            else
                return new Integer((int)(result));
        } else {
            State oldState = this.state;
            this.state = callee;
            MethodInterpreter mi = new MethodInterpreter(m);
            Object synchobj = null;
            try {
                if (m.isSynchronized()) {
                    if (!m.isStatic()) {
                        if (mi.getTraceFlag()) mi.getTraceOut().println("synchronized instance method, locking 'this' object");
                        vm.monitorenter(synchobj = state.getLocal_A(0), mi);
                    } else {
                        if (mi.getTraceFlag()) mi.getTraceOut().println("synchronized static method, locking class object");
                        vm.monitorenter(synchobj = Reflection.getJDKType(m.getDeclaringClass()), mi);
                    }
                }
                mi.forwardTraversal();
                this.state = oldState;
                if (m.isSynchronized()) {
                    if (mi.getTraceFlag()) mi.getTraceOut().println("exiting synchronized method, unlocking object");
                    vm.monitorexit(synchobj);
                }
                jq_Type returnType = m.getReturnType();
                Object retval;
                if (returnType.isReferenceType()) {
                    retval = callee.getReturnVal_A();
                } else if (returnType == jq_Primitive.VOID) {
                    retval = null;
                } else if (returnType == jq_Primitive.LONG) {
                    retval = new Long(callee.getReturnVal_L());
                } else if (returnType == jq_Primitive.FLOAT) {
                    retval = new Float(callee.getReturnVal_F());
                } else if (returnType == jq_Primitive.DOUBLE) {
                    retval = new Double(callee.getReturnVal_D());
                } else {
                    retval = new Integer(callee.getReturnVal_I());
                }
                if (mi.getTraceFlag())
                    mi.getTraceOut().println("Return value: "+retval);
                return retval;
            } catch (WrappedException ix) {
                this.state = oldState;
                if (m.isSynchronized()) {
                    if (mi.getTraceFlag()) mi.getTraceOut().println("exiting synchronized method, unlocking object");
                    vm.monitorexit(synchobj);
                }
                throw ix.t;
            }
        }
    }
        /*
        int j = m.getParamWords();
        int[] argVals = new int[j];
        for (int i=paramTypes.length-1; i>=0; --i, --j) {
            jq_Type t = paramTypes[i];
            if (t.isPrimitiveType()) {
                if (t == jq_Primitive.LONG) {
                    long v = state.pop_L();
                    argVals[  j] = (int)v;       // lo
                    argVals[--j] = (int)(v>>32); // hi
                } else if (t == jq_Primitive.FLOAT) {
                    argVals[j] = Float.floatToRawIntBits(state.pop_F());
                } else if (t == jq_Primitive.DOUBLE) {
                    long v = Double.doubleToRawLongBits(state.pop_D());
                    argVals[  j] = (int)v;       // lo
                    argVals[--j] = (int)(v>>32); // hi
                } else {
                    argVals[j] = state.pop_I();
                }
            } else {
                argVals[j] = Unsafe.addressOf(state.pop_A());
            }
        }
        jq.Assert(j==0);
        for (int i=0; i<argVals.length; ++i) {
            if (callee == null) Unsafe.pushArg(argVals[i]);
            else callee.push_I(argVals[i]);
        }
        if (callee == null) return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
        else interpret(m, callee);
        */
    
    protected State state;
    protected final VMInterface vm;

    public abstract static class State {
        public abstract void push_I(int v);
        public abstract void push_L(long v);
        public abstract void push_F(float v);
        public abstract void push_D(double v);
        public abstract void push_A(Object v);
        public abstract void push(Object v);
        public abstract int pop_I();
        public abstract long pop_L();
        public abstract float pop_F();
        public abstract double pop_D();
        public abstract Object pop_A();
        public abstract Object pop();
        public abstract void popAll();
        public abstract Object peek_A(int depth);
        public abstract void setLocal_I(int i, int v);
        public abstract void setLocal_L(int i, long v);
        public abstract void setLocal_F(int i, float v);
        public abstract void setLocal_D(int i, double v);
        public abstract void setLocal_A(int i, Object v);
        public abstract int getLocal_I(int i);
        public abstract long getLocal_L(int i);
        public abstract float getLocal_F(int i);
        public abstract double getLocal_D(int i);
        public abstract Object getLocal_A(int i);
        public abstract void return_I(int v);
        public abstract void return_L(long v);
        public abstract void return_F(float v);
        public abstract void return_D(double v);
        public abstract void return_A(Object v);
        public abstract void return_V();
        public abstract int getReturnVal_I();
        public abstract long getReturnVal_L();
        public abstract float getReturnVal_F();
        public abstract double getReturnVal_D();
        public abstract Object getReturnVal_A();
    }
    
    public abstract static class VMInterface {
        public abstract Object new_obj(jq_Type t);
        public abstract Object new_array(jq_Type t, int length);
        public abstract Object checkcast(Object o, jq_Type t);
        public abstract boolean instance_of(Object o, jq_Type t);
        public abstract int arraylength(Object o);
        public abstract void monitorenter(Object o, MethodInterpreter v);
        public abstract void monitorexit(Object o);
        public abstract Object multinewarray(int[] dims, jq_Type t);
    }

    public static class WrappedException extends RuntimeException {
        Throwable t;
        WrappedException(Throwable t) { this.t = t; }
        public String toString() { return "WrappedException: "+t; }
    }
    
    class MethodInterpreter extends BytecodeVisitor {
        
        MethodInterpreter(jq_Method method) {
            super(method);
            i_end = -1;
            String s = method.getDeclaringClass().getName().toString();
            int i = s.lastIndexOf('.');
            name = s.substring(i+1)+"/"+method.getName();
            TRACE = ALWAYS_TRACE;
            out = System.err;
        }

        final String name;
        public String toString() {
            return name;
        }
        
        // Workaround for javac bug -> cannot access protected members of inner classes.
        boolean getTraceFlag() { return TRACE; }
        java.io.PrintStream getTraceOut() { return out; }
        
        public void forwardTraversal() throws VerifyError, WrappedException {
            if (this.TRACE) this.out.println(this+": Starting traversal.");
            for (;;) {
                i_start = i_end+1;
                if (i_start >= bcs.length) break;
                try {
                    super.visitBytecode();
                } catch (WrappedException ix) {
                    if (this.TRACE) this.out.println(this+": Exception thrown! "+ix.t);
                    handleException(ix.t);
                } catch (Throwable x) {
                    if (this.TRACE) this.out.println(this+": RuntimeException/Error thrown! "+x);
                    handleException(x);
                }
            }
            if (this.TRACE) this.out.println(this+": Finished traversal.");
        }

        public void continueForwardTraversal() throws VerifyError, WrappedException, ReflectiveInterpreter.MonitorExit {
            for (;;) {
                i_start = i_end+1;
                if (i_start >= bcs.length) break;
                try {
                    super.visitBytecode();
                } catch (ReflectiveInterpreter.MonitorExit x) {
                    throw x;
                } catch (WrappedException ix) {
                    if (this.TRACE) this.out.println(this+": Exception thrown! "+ix.t);
                    handleException(ix.t);
                } catch (Throwable x) {
                    if (this.TRACE) this.out.println(this+": RuntimeException/Error thrown! "+x);
                    handleException(x);
                }
            }
        }
        
        public void visitBytecode() throws WrappedException {
            try {
                super.visitBytecode();
            } catch (WrappedException ix) {
                if (this.TRACE) this.out.println(this+": Exception thrown! "+ix.t);
                handleException(ix.t);
            } catch (Throwable x) {
                if (this.TRACE) this.out.println(this+": RuntimeException/Error thrown! "+x);
                handleException(x);
            }
        }
        
        private void handleException(Throwable x) throws WrappedException {
            jq_Class t = (jq_Class)jq_Reference.getTypeOf(x);
            t.prepare();
            jq_TryCatchBC[] tc = method.getExceptionTable();
            for (int i=0; i<tc.length; ++i) {
                if (tc[i].catches(i_start, t)) {
                    state.popAll(); state.push_A(x);
                    branchTo(tc[i].getHandlerPC());
                    if (this.TRACE) this.out.println(this+": Branching to exception handler "+tc[i]);
                    return;
                }
            }
            if (this.TRACE) this.out.println(this+": Uncaught exception, exiting method.");
            throw new WrappedException(x);
        }

        protected void branchTo(int target) {
            i_end = target-1;
        }
        
        public void visitNOP() {
            super.visitNOP();
        }
        public void visitACONST(Object s) {
            super.visitACONST(s);
            state.push_A(s);
        }
        public void visitICONST(int c) {
            super.visitICONST(c);
            state.push_I(c);
        }
        public void visitLCONST(long c) {
            super.visitLCONST(c);
            state.push_L(c);
        }
        public void visitFCONST(float c) {
            super.visitFCONST(c);
            state.push_F(c);
        }
        public void visitDCONST(double c) {
            super.visitDCONST(c);
            state.push_D(c);
        }
        public void visitILOAD(int i) {
            super.visitILOAD(i);
            state.push_I(state.getLocal_I(i));
        }
        public void visitLLOAD(int i) {
            super.visitLLOAD(i);
            state.push_L(state.getLocal_L(i));
        }
        public void visitFLOAD(int i) {
            super.visitFLOAD(i);
            state.push_F(state.getLocal_F(i));
        }
        public void visitDLOAD(int i) {
            super.visitDLOAD(i);
            state.push_D(state.getLocal_D(i));
        }
        public void visitALOAD(int i) {
            super.visitALOAD(i);
            state.push_A(state.getLocal_A(i));
        }
        public void visitISTORE(int i) {
            super.visitISTORE(i);
            state.setLocal_I(i, state.pop_I());
        }
        public void visitLSTORE(int i) {
            super.visitLSTORE(i);
            state.setLocal_L(i, state.pop_L());
        }
        public void visitFSTORE(int i) {
            super.visitFSTORE(i);
            state.setLocal_F(i, state.pop_F());
        }
        public void visitDSTORE(int i) {
            super.visitDSTORE(i);
            state.setLocal_D(i, state.pop_D());
        }
        public void visitASTORE(int i) {
            super.visitASTORE(i);
            state.setLocal_A(i, state.pop_A());
        }
        public void visitIALOAD() {
            super.visitIALOAD();
            int index = state.pop_I();
            int[] array = (int[])state.pop_A();
            state.push_I(array[index]);
        }
        public void visitLALOAD() {
            super.visitLALOAD();
            int index = state.pop_I();
            long[] array = (long[])state.pop_A();
            state.push_L(array[index]);
        }
        public void visitFALOAD() {
            super.visitFALOAD();
            int index = state.pop_I();
            float[] array = (float[])state.pop_A();
            state.push_F(array[index]);
        }
        public void visitDALOAD() {
            super.visitDALOAD();
            int index = state.pop_I();
            double[] array = (double[])state.pop_A();
            state.push_D(array[index]);
        }
        public void visitAALOAD() {
            super.visitAALOAD();
            int index = state.pop_I();
            Object[] array = (Object[])state.pop_A();
            state.push_A(array[index]);
        }
        public void visitBALOAD() {
            super.visitBALOAD();
            int index = state.pop_I();
            Object array = (Object)state.pop_A();
            int val;
            try {
                if (array.getClass() == Class.forName("[Z")) val = ((boolean[])array)[index]?1:0;
                else val = ((byte[])array)[index];
            } catch (ClassNotFoundException x) { jq.UNREACHABLE(); return; }
            state.push_I(val);
        }
        public void visitCALOAD() {
            super.visitCALOAD();
            int index = state.pop_I();
            char[] array = (char[])state.pop_A();
            state.push_I(array[index]);
        }
        public void visitSALOAD() {
            super.visitSALOAD();
            int index = state.pop_I();
            short[] array = (short[])state.pop_A();
            state.push_I(array[index]);
        }
        public void visitIASTORE() {
            super.visitIASTORE();
            int val = state.pop_I();
            int index = state.pop_I();
            int[] array = (int[])state.pop_A();
            array[index] = val;
        }
        public void visitLASTORE() {
            super.visitLASTORE();
            long val = state.pop_L();
            int index = state.pop_I();
            long[] array = (long[])state.pop_A();
            array[index] = val;
        }
        public void visitFASTORE() {
            super.visitFASTORE();
            float val = state.pop_F();
            int index = state.pop_I();
            float[] array = (float[])state.pop_A();
            array[index] = val;
        }
        public void visitDASTORE() {
            super.visitDASTORE();
            double val = state.pop_D();
            int index = state.pop_I();
            double[] array = (double[])state.pop_A();
            array[index] = val;
        }
        public void visitAASTORE() {
            super.visitAASTORE();
            Object val = state.pop_A();
            int index = state.pop_I();
            Object[] array = (Object[])state.pop_A();
            array[index] = val;
        }
        public void visitBASTORE() {
            super.visitBASTORE();
            int val = state.pop_I();
            int index = state.pop_I();
            Object array = (Object)state.pop_A();
            try {
                if (array.getClass() == Class.forName("[Z")) ((boolean[])array)[index] = val!=0;
                else ((byte[])array)[index] = (byte)val;
            } catch (ClassNotFoundException x) { jq.UNREACHABLE(); }
        }
        public void visitCASTORE() {
            super.visitCASTORE();
            int val = state.pop_I();
            int index = state.pop_I();
            char[] array = (char[])state.pop_A();
            array[index] = (char)val;
        }
        public void visitSASTORE() {
            super.visitSASTORE();
            int val = state.pop_I();
            int index = state.pop_I();
            short[] array = (short[])state.pop_A();
            array[index] = (short)val;
        }
        public void visitPOP() {
            super.visitPOP();
            state.pop();
        }
        public void visitPOP2() {
            super.visitPOP2();
            state.pop();
            state.pop();
        }
        public void visitDUP() {
            super.visitDUP();
            Object o = state.pop();
            state.push(o);
            state.push(o);
        }
        public void visitDUP_x1() {
            super.visitDUP_x1();
            Object o1 = state.pop();
            Object o2 = state.pop();
            state.push(o1);
            state.push(o2);
            state.push(o1);
        }
        public void visitDUP_x2() {
            super.visitDUP_x2();
            Object o1 = state.pop();
            Object o2 = state.pop();
            Object o3 = state.pop();
            state.push(o1);
            state.push(o3);
            state.push(o2);
            state.push(o1);
        }
        public void visitDUP2() {
            super.visitDUP2();
            Object o1 = state.pop();
            Object o2 = state.pop();
            state.push(o2);
            state.push(o1);
            state.push(o2);
            state.push(o1);
        }
        public void visitDUP2_x1() {
            super.visitDUP2_x1();
            Object o1 = state.pop();
            Object o2 = state.pop();
            Object o3 = state.pop();
            state.push(o2);
            state.push(o1);
            state.push(o3);
            state.push(o2);
            state.push(o1);
        }
        public void visitDUP2_x2() {
            super.visitDUP2_x2();
            Object o1 = state.pop();
            Object o2 = state.pop();
            Object o3 = state.pop();
            Object o4 = state.pop();
            state.push(o2);
            state.push(o1);
            state.push(o4);
            state.push(o3);
            state.push(o2);
            state.push(o1);
        }
        public void visitSWAP() {
            super.visitSWAP();
            Object o1 = state.pop();
            Object o2 = state.pop();
            state.push(o1);
            state.push(o2);
        }
        public void visitIBINOP(byte op) {
            super.visitIBINOP(op);
            int v1 = state.pop_I();
            int v2 = state.pop_I();
            switch(op) {
                case BINOP_ADD:
                    state.push_I(v2+v1);
                    break;
                case BINOP_SUB:
                    state.push_I(v2-v1);
                    break;
                case BINOP_MUL:
                    state.push_I(v2*v1);
                    break;
                case BINOP_DIV:
                    state.push_I(v2/v1);
                    break;
                case BINOP_REM:
                    state.push_I(v2%v1);
                    break;
                case BINOP_AND:
                    state.push_I(v2&v1);
                    break;
                case BINOP_OR:
                    state.push_I(v2|v1);
                    break;
                case BINOP_XOR:
                    state.push_I(v2^v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitLBINOP(byte op) {
            super.visitLBINOP(op);
            long v1 = state.pop_L();
            long v2 = state.pop_L();
            switch(op) {
                case BINOP_ADD:
                    state.push_L(v2+v1);
                    break;
                case BINOP_SUB:
                    state.push_L(v2-v1);
                    break;
                case BINOP_MUL:
                    state.push_L(v2*v1);
                    break;
                case BINOP_DIV:
                    state.push_L(v2/v1);
                    break;
                case BINOP_REM:
                    state.push_L(v2%v1);
                    break;
                case BINOP_AND:
                    state.push_L(v2&v1);
                    break;
                case BINOP_OR:
                    state.push_L(v2|v1);
                    break;
                case BINOP_XOR:
                    state.push_L(v2^v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitFBINOP(byte op) {
            super.visitFBINOP(op);
            float v1 = state.pop_F();
            float v2 = state.pop_F();
            switch(op) {
                case BINOP_ADD:
                    state.push_F(v2+v1);
                    break;
                case BINOP_SUB:
                    state.push_F(v2-v1);
                    break;
                case BINOP_MUL:
                    state.push_F(v2*v1);
                    break;
                case BINOP_DIV:
                    state.push_F(v2/v1);
                    break;
                case BINOP_REM:
                    state.push_F(v2%v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitDBINOP(byte op) {
            super.visitDBINOP(op);
            double v1 = state.pop_D();
            double v2 = state.pop_D();
            switch(op) {
                case BINOP_ADD:
                    state.push_D(v2+v1);
                    break;
                case BINOP_SUB:
                    state.push_D(v2-v1);
                    break;
                case BINOP_MUL:
                    state.push_D(v2*v1);
                    break;
                case BINOP_DIV:
                    state.push_D(v2/v1);
                    break;
                case BINOP_REM:
                    state.push_D(v2%v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitIUNOP(byte op) {
            super.visitIUNOP(op);
            jq.Assert(op == UNOP_NEG);
            state.push_I(-state.pop_I());
        }
        public void visitLUNOP(byte op) {
            super.visitLUNOP(op);
            jq.Assert(op == UNOP_NEG);
            state.push_L(-state.pop_L());
        }
        public void visitFUNOP(byte op) {
            super.visitFUNOP(op);
            jq.Assert(op == UNOP_NEG);
            state.push_F(-state.pop_F());
        }
        public void visitDUNOP(byte op) {
            super.visitDUNOP(op);
            jq.Assert(op == UNOP_NEG);
            state.push_D(-state.pop_D());
        }
        public void visitISHIFT(byte op) {
            super.visitISHIFT(op);
            int v1 = state.pop_I();
            int v2 = state.pop_I();
            switch(op) {
                case SHIFT_LEFT:
                    state.push_I(v2 << v1);
                    break;
                case SHIFT_RIGHT:
                    state.push_I(v2 >> v1);
                    break;
                case SHIFT_URIGHT:
                    state.push_I(v2 >>> v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitLSHIFT(byte op) {
            super.visitLSHIFT(op);
            int v1 = state.pop_I();
            long v2 = state.pop_L();
            switch(op) {
                case SHIFT_LEFT:
                    state.push_L(v2 << v1);
                    break;
                case SHIFT_RIGHT:
                    state.push_L(v2 >> v1);
                    break;
                case SHIFT_URIGHT:
                    state.push_L(v2 >>> v1);
                    break;
                default:
                    jq.UNREACHABLE();
            }
        }
        public void visitIINC(int i, int v) {
            super.visitIINC(i, v);
            state.setLocal_I(i, state.getLocal_I(i)+v);
        }
        public void visitI2L() {
            super.visitI2L();
            state.push_L((long)state.pop_I());
        }
        public void visitI2F() {
            super.visitI2F();
            state.push_F((float)state.pop_I());
        }
        public void visitI2D() {
            super.visitI2D();
            state.push_D((double)state.pop_I());
        }
        public void visitL2I() {
            super.visitL2I();
            state.push_I((int)state.pop_L());
        }
        public void visitL2F() {
            super.visitL2F();
            state.push_F((float)state.pop_L());
        }
        public void visitL2D() {
            super.visitL2D();
            state.push_D((double)state.pop_L());
        }
        public void visitF2I() {
            super.visitF2I();
            state.push_I((int)state.pop_F());
        }
        public void visitF2L() {
            super.visitF2L();
            state.push_L((long)state.pop_F());
        }
        public void visitF2D() {
            super.visitF2D();
            state.push_D((double)state.pop_F());
        }
        public void visitD2I() {
            super.visitD2I();
            state.push_I((int)state.pop_D());
        }
        public void visitD2L() {
            super.visitD2L();
            state.push_L((long)state.pop_D());
        }
        public void visitD2F() {
            super.visitD2F();
            state.push_F((float)state.pop_D());
        }
        public void visitI2B() {
            super.visitI2B();
            state.push_I((byte)state.pop_I());
        }
        public void visitI2C() {
            super.visitI2C();
            state.push_I((char)state.pop_I());
        }
        public void visitI2S() {
            super.visitI2S();
            state.push_I((short)state.pop_I());
        }
        public void visitLCMP2() {
            super.visitLCMP2();
            long v1 = state.pop_L();
            long v2 = state.pop_L();
            state.push_I((v2>v1)?1:((v2==v1)?0:-1));
        }
        public void visitFCMP2(byte op) {
            super.visitFCMP2(op);
            float v1 = state.pop_F();
            float v2 = state.pop_F();
            int val;
            if (op == CMP_L)
                val = ((v2>v1)?1:((v2==v1)?0:-1));
            else
                val = ((v2<v1)?-1:((v2==v1)?0:1));
            state.push_I(val);
        }
        public void visitDCMP2(byte op) {
            super.visitDCMP2(op);
            double v1 = state.pop_D();
            double v2 = state.pop_D();
            int val;
            if (op == CMP_L)
                val = ((v2>v1)?1:((v2==v1)?0:-1));
            else
                val = ((v2<v1)?-1:((v2==v1)?0:1));
            state.push_I(val);
        }
        public void visitIF(byte op, int target) {
            super.visitIF(op, target);
            int v = state.pop_I();
            switch(op) {
                case CMP_EQ: if (v==0) branchTo(target); break;
                case CMP_NE: if (v!=0) branchTo(target); break;
                case CMP_LT: if (v<0) branchTo(target); break;
                case CMP_GE: if (v>=0) branchTo(target); break;
                case CMP_LE: if (v<=0) branchTo(target); break;
                case CMP_GT: if (v>0) branchTo(target); break;
                default: jq.UNREACHABLE();
            }
        }
        public void visitIFREF(byte op, int target) {
            super.visitIFREF(op, target);
            Object v = state.pop_A();
            switch(op) {
                case CMP_EQ: if (v==null) branchTo(target); break;
                case CMP_NE: if (v!=null) branchTo(target); break;
                default: jq.UNREACHABLE();
            }
        }
        public void visitIFCMP(byte op, int target) {
            super.visitIFCMP(op, target);
            int v1 = state.pop_I();
            int v2 = state.pop_I();
            switch(op) {
                case CMP_EQ: if (v2==v1) branchTo(target); break;
                case CMP_NE: if (v2!=v1) branchTo(target); break;
                case CMP_LT: if (v2<v1) branchTo(target); break;
                case CMP_GE: if (v2>=v1) branchTo(target); break;
                case CMP_LE: if (v2<=v1) branchTo(target); break;
                case CMP_GT: if (v2>v1) branchTo(target); break;
                default: jq.UNREACHABLE();
            }
        }
        public void visitIFREFCMP(byte op, int target) {
            super.visitIFREFCMP(op, target);
            Object v1 = state.pop_A();
            Object v2 = state.pop_A();
            switch(op) {
                case CMP_EQ: if (v2==v1) branchTo(target); break;
                case CMP_NE: if (v2!=v1) branchTo(target); break;
                default: jq.UNREACHABLE();
            }
        }
        public void visitGOTO(int target) {
            super.visitGOTO(target);
            branchTo(target);
        }
        public void visitJSR(int target) {
            super.visitJSR(target);
            state.push_I(i_end+1);
            branchTo(target);
        }
        public void visitRET(int i) {
            super.visitRET(i);
            branchTo(state.getLocal_I(i));
        }
        public void visitTABLESWITCH(int default_target, int low, int high, int[] targets) {
            super.visitTABLESWITCH(default_target, low, high, targets);
            int v = state.pop_I();
            if ((v < low) || (v > high)) branchTo(default_target);
            else branchTo(targets[v-low]);
        }
        public void visitLOOKUPSWITCH(int default_target, int[] values, int[] targets) {
            super.visitLOOKUPSWITCH(default_target, values, targets);
            int v = state.pop_I();
            for (int i=0; i<values.length; ++i) {
                if (v == values[i]) {
                    branchTo(targets[i]);
                    return;
                }
            }
            branchTo(default_target);
        }
        public void visitIRETURN() {
            super.visitIRETURN();
            state.return_I(state.pop_I());
            i_end = bcs.length;
        }
        public void visitLRETURN() {
            super.visitLRETURN();
            state.return_L(state.pop_L());
            i_end = bcs.length;
        }
        public void visitFRETURN() {
            super.visitFRETURN();
            state.return_F(state.pop_F());
            i_end = bcs.length;
        }
        public void visitDRETURN() {
            super.visitDRETURN();
            state.return_D(state.pop_D());
            i_end = bcs.length;
        }
        public void visitARETURN() {
            super.visitARETURN();
            state.return_A(state.pop_A());
            i_end = bcs.length;
        }
        public void visitVRETURN() {
            super.visitVRETURN();
            state.return_V();
            i_end = bcs.length;
        }
        public void visitIGETSTATIC(jq_StaticField f) {
            super.visitIGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_I(Reflection.getstatic_I(f));
        }
        public void visitLGETSTATIC(jq_StaticField f) {
            super.visitLGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_L(Reflection.getstatic_L(f));
        }
        public void visitFGETSTATIC(jq_StaticField f) {
            super.visitFGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_F(Reflection.getstatic_F(f));
        }
        public void visitDGETSTATIC(jq_StaticField f) {
            super.visitDGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_D(Reflection.getstatic_D(f));
        }
        public void visitAGETSTATIC(jq_StaticField f) {
            super.visitAGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_A(Reflection.getstatic_A(f));
        }
        public void visitZGETSTATIC(jq_StaticField f) {
            super.visitZGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_I(Reflection.getstatic_Z(f)?1:0);
        }
        public void visitBGETSTATIC(jq_StaticField f) {
            super.visitBGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_I(Reflection.getstatic_B(f));
        }
        public void visitCGETSTATIC(jq_StaticField f) {
            super.visitCGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_I(Reflection.getstatic_C(f));
        }
        public void visitSGETSTATIC(jq_StaticField f) {
            super.visitSGETSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            state.push_I(Reflection.getstatic_S(f));
        }
        public void visitIPUTSTATIC(jq_StaticField f) {
            super.visitIPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_I(f, state.pop_I());
        }
        public void visitLPUTSTATIC(jq_StaticField f) {
            super.visitLPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_L(f, state.pop_L());
        }
        public void visitFPUTSTATIC(jq_StaticField f) {
            super.visitFPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_F(f, state.pop_F());
        }
        public void visitDPUTSTATIC(jq_StaticField f) {
            super.visitDPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_D(f, state.pop_D());
        }
        public void visitAPUTSTATIC(jq_StaticField f) {
            super.visitAPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_A(f, state.pop_A());
        }
        public void visitZPUTSTATIC(jq_StaticField f) {
            super.visitZPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_Z(f, state.pop_I()!=0);
        }
        public void visitBPUTSTATIC(jq_StaticField f) {
            super.visitBPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_B(f, (byte)state.pop_I());
        }
        public void visitCPUTSTATIC(jq_StaticField f) {
            super.visitCPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_C(f, (char)state.pop_I());
        }
        public void visitSPUTSTATIC(jq_StaticField f) {
            super.visitSPUTSTATIC(f);
            f = resolve(f);
            f.getDeclaringClass().cls_initialize();
            Reflection.putstatic_S(f, (short)state.pop_I());
        }
        public void visitIGETFIELD(jq_InstanceField f) {
            super.visitIGETFIELD(f);
            f = resolve(f);
            state.push_I(Reflection.getfield_I(state.pop_A(), f));
        }
        public void visitLGETFIELD(jq_InstanceField f) {
            super.visitLGETFIELD(f);
            f = resolve(f);
            state.push_L(Reflection.getfield_L(state.pop_A(), f));
        }
        public void visitFGETFIELD(jq_InstanceField f) {
            super.visitFGETFIELD(f);
            f = resolve(f);
            state.push_F(Reflection.getfield_F(state.pop_A(), f));
        }
        public void visitDGETFIELD(jq_InstanceField f) {
            super.visitDGETFIELD(f);
            f = resolve(f);
            state.push_D(Reflection.getfield_D(state.pop_A(), f));
        }
        public void visitAGETFIELD(jq_InstanceField f) {
            super.visitAGETFIELD(f);
            f = resolve(f);
            state.push_A(Reflection.getfield_A(state.pop_A(), f));
        }
        public void visitBGETFIELD(jq_InstanceField f) {
            super.visitBGETFIELD(f);
            f = resolve(f);
            state.push_I(Reflection.getfield_B(state.pop_A(), f));
        }
        public void visitCGETFIELD(jq_InstanceField f) {
            super.visitCGETFIELD(f);
            f = resolve(f);
            state.push_I(Reflection.getfield_C(state.pop_A(), f));
        }
        public void visitSGETFIELD(jq_InstanceField f) {
            super.visitSGETFIELD(f);
            f = resolve(f);
            state.push_I(Reflection.getfield_S(state.pop_A(), f));
        }
        public void visitZGETFIELD(jq_InstanceField f) {
            super.visitZGETFIELD(f);
            f = resolve(f);
            state.push_I(Reflection.getfield_Z(state.pop_A(), f)?1:0);
        }
        public void visitIPUTFIELD(jq_InstanceField f) {
            super.visitIPUTFIELD(f);
            f = resolve(f);
            int v = state.pop_I();
            Reflection.putfield_I(state.pop_A(), f, v);
        }
        public void visitLPUTFIELD(jq_InstanceField f) {
            super.visitLPUTFIELD(f);
            f = resolve(f);
            long v = state.pop_L();
            Reflection.putfield_L(state.pop_A(), f, v);
        }
        public void visitFPUTFIELD(jq_InstanceField f) {
            super.visitFPUTFIELD(f);
            f = resolve(f);
            float v = state.pop_F();
            Reflection.putfield_F(state.pop_A(), f, v);
        }
        public void visitDPUTFIELD(jq_InstanceField f) {
            super.visitDPUTFIELD(f);
            f = resolve(f);
            double v = state.pop_D();
            Reflection.putfield_D(state.pop_A(), f, v);
        }
        public void visitAPUTFIELD(jq_InstanceField f) {
            super.visitAPUTFIELD(f);
            f = resolve(f);
            Object v = state.pop_A();
            Reflection.putfield_A(state.pop_A(), f, v);
        }
        public void visitBPUTFIELD(jq_InstanceField f) {
            super.visitBPUTFIELD(f);
            f = resolve(f);
            byte v = (byte)state.pop_I();
            Reflection.putfield_B(state.pop_A(), f, v);
        }
        public void visitCPUTFIELD(jq_InstanceField f) {
            super.visitCPUTFIELD(f);
            f = resolve(f);
            char v = (char)state.pop_I();
            Reflection.putfield_C(state.pop_A(), f, v);
        }
        public void visitSPUTFIELD(jq_InstanceField f) {
            super.visitSPUTFIELD(f);
            f = resolve(f);
            short v = (short)state.pop_I();
            Reflection.putfield_S(state.pop_A(), f, v);
        }
        public void visitZPUTFIELD(jq_InstanceField f) {
            super.visitZPUTFIELD(f);
            f = resolve(f);
            boolean v = state.pop_I()!=0;
            Reflection.putfield_Z(state.pop_A(), f, v);
        }
        protected Object INVOKEhelper(byte op, jq_Method f) {
            f = (jq_Method)resolve(f);
            jq_Class k = f.getDeclaringClass();
            k.cls_initialize();
            if (k == Unsafe._class || k.isAddressType()) {
                try {
                    // redirect call
                    return invokeUnsafeMethod(f);
                } catch (Throwable t) {
                    if (this.TRACE) this.out.println(this+": "+f+" threw "+t);
                    throw new WrappedException(t);
                }
            }
            if (op == INVOKE_SPECIAL) {
                f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
            } else if (op != INVOKE_STATIC) {
                Object o = state.peek_A(f.getParamWords()-1);
                jq_Reference t = jq_Reference.getTypeOf(o);
                t.cls_initialize();
                if (op == INVOKE_INTERFACE) {
                    if (!t.implementsInterface(f.getDeclaringClass()))
                        throw new IncompatibleClassChangeError();
                    if (t.isArrayType()) t = PrimordialClassLoader.loader.getJavaLangObject();
                } else {
                    jq.Assert(op == INVOKE_VIRTUAL);
                }
                jq_Method f2 = f;
                f = t.getVirtualMethod(f.getNameAndDesc());
                if (this.TRACE) this.out.println(this+": virtual method target "+f);
                if (f == null)
                    throw new AbstractMethodError("no such method "+f2.toString()+" in type "+t);
                if (f.isAbstract())
                    throw new AbstractMethodError("method "+f2.toString()+" on type "+t+" is abstract");
            } else {
                // static call
            }
            try {
                return invokeMethod(f);
            } catch (Throwable t) {
                if (this.TRACE) this.out.println(this+": "+f+" threw "+t);
                throw new WrappedException(t);
            }
        }
        public void visitIINVOKE(byte op, jq_Method f) {
            super.visitIINVOKE(op, f);
            state.push_I(((Integer)INVOKEhelper(op, f)).intValue());
        }
        public void visitLINVOKE(byte op, jq_Method f) {
            super.visitLINVOKE(op, f);
            state.push_L(((Long)INVOKEhelper(op, f)).longValue());
        }
        public void visitFINVOKE(byte op, jq_Method f) {
            super.visitFINVOKE(op, f);
            state.push_F(((Float)INVOKEhelper(op, f)).floatValue());
        }
        public void visitDINVOKE(byte op, jq_Method f) {
            super.visitDINVOKE(op, f);
            state.push_D(((Double)INVOKEhelper(op, f)).doubleValue());
        }
        public void visitAINVOKE(byte op, jq_Method f) {
            super.visitAINVOKE(op, f);
            state.push_A(INVOKEhelper(op, f));
        }
        public void visitVINVOKE(byte op, jq_Method f) {
            super.visitVINVOKE(op, f);
            INVOKEhelper(op, f);
        }
        public void visitNEW(jq_Type f) {
            super.visitNEW(f);
            state.push_A(vm.new_obj(f));
        }
        public void visitNEWARRAY(jq_Array f) {
            super.visitNEWARRAY(f);
            state.push_A(vm.new_array(f, state.pop_I()));
        }
        public void visitCHECKCAST(jq_Type f) {
            super.visitCHECKCAST(f);
            state.push_A(vm.checkcast(state.pop_A(), f));
        }
        public void visitINSTANCEOF(jq_Type f) {
            super.visitINSTANCEOF(f);
            state.push_I(vm.instance_of(state.pop_A(), f)?1:0);
        }
        public void visitARRAYLENGTH() {
            super.visitARRAYLENGTH();
            state.push_I(vm.arraylength(state.pop_A()));
        }
        public void visitATHROW() {
            super.visitATHROW();
            throw new WrappedException((Throwable)state.pop_A());
        }
        public void visitMONITOR(byte op) {
            super.visitMONITOR(op);
            Object v = state.pop_A();
            if (op == MONITOR_ENTER) vm.monitorenter(v, this);
            else vm.monitorexit(v);
        }
        public void visitMULTINEWARRAY(jq_Type f, char dim) {
            super.visitMULTINEWARRAY(f, dim);
            int[] dims = new int[dim];
            //for (int i=0; i<dim; ++i) f = ((jq_Array)f).getElementType();
            for (int i=0; i<dim; ++i) dims[dim-i-1] = state.pop_I();
            state.push_A(vm.multinewarray(dims, f));
        }
    }
    
}

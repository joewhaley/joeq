/*
 * QuadInterpreter.java
 *
 * Created on February 8, 2002, 3:06 PM
 */

package Interpreter;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.lang.reflect.Method;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import Clazz.*;
import Compil3r.Quad.*;
import Compil3r.Quad.RegisterFactory.Register;
import Compil3r.Quad.Operand.ParamListOperand;
import Run_Time.Reflection;
import Util.Templates.ListIterator;
import ReflectiveInterpreter.ReflectiveVMInterface;
import Bootstrap.PrimordialClassLoader;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class QuadInterpreter {

    public static class State extends QuadVisitor.EmptyVisitor {
        Map/*<Register, Object>*/ registers;
        ControlFlowGraph cfg;
        RegisterFactory rf;
        BasicBlock current_bb;
        ListIterator.Quad current_iterator;
        Quad current_quad;
	Object return_value;
	Throwable thrown;

        public State() {
            registers = new HashMap();
        }
        
	public static boolean TRACE = false;

        public static long num_nullcheck = 0;
        
	public void visitNullCheck(Quad q) { ++num_nullcheck; }
            
	public void visitQuad(Quad q) {
	    if (TRACE) System.out.println("Registers: "+registers);
	    if (TRACE) System.out.println("Interpreting: "+q);
	    q.interpret(this);
	}

	public void setReturnValue(Object o) { return_value = o; }
	public Object getReturnValue() { return return_value; }
	public Throwable getThrown() { return thrown; }
	public void setThrown(Throwable t) { thrown = t; }

	public Register getExceptionRegister() { return rf.getStack(0, PrimordialClassLoader.getJavaLangObject()); }

	public State invokeReflective(jq_Method f, ParamListOperand plo) {
	    if (f instanceof jq_StaticMethod)
		return invokeStaticReflective((jq_StaticMethod)f, plo);
	    else
		return invokeInstanceReflective((jq_InstanceMethod)f, plo);
	}
	public State invokeInstanceReflective(jq_InstanceMethod f, ParamListOperand plo) {
	    State s = new State();
	    try {
		Object[] param = new Object[plo.length()-1];
		for (int i=1; i<plo.length(); ++i) {
		    param[i-1] = getReg(plo.get(i).getRegister());
		}
		if (f instanceof jq_Initializer) {
		    try {
			Constructor co = (Constructor)Reflection.getJDKMember(f);
			co.setAccessible(true);
			UninitializedReference u = (UninitializedReference)getReg_A(plo.get(0).getRegister());
			jq.assert(u.k == f.getDeclaringClass());
			Object inited = co.newInstance(param);
			replaceUninitializedReferences(inited, u);
		    } catch (InstantiationException x) {
			jq.UNREACHABLE();
		    } catch (IllegalAccessException x) {
			jq.UNREACHABLE();
		    } catch (IllegalArgumentException x) {
			jq.UNREACHABLE();
		    } catch (InvocationTargetException x) {
			handleException(x.getTargetException());
		    }
		    return s;
		}

		Method m = (Method)Reflection.getJDKMember(f);
		m.setAccessible(true);
		Object result = m.invoke(getReg(plo.get(0).getRegister()), param);
		s.setReturnValue(result);
	    } catch (IllegalAccessException x) {
		jq.UNREACHABLE();
	    } catch (IllegalArgumentException x) {
		jq.UNREACHABLE();
	    } catch (InvocationTargetException x) {
		s.setThrown(x.getTargetException());
	    }
	    return s;
	}
	public State invokeStaticReflective(jq_StaticMethod f, ParamListOperand plo) {
	    State s = new State();
	    try {
		Object[] param = new Object[plo.length()];
		for (int i=0; i<plo.length(); ++i) {
		    param[i] = getReg(plo.get(i).getRegister());
		}
		Method m = (Method)Reflection.getJDKMember(f);
		m.setAccessible(true);
		Object result = m.invoke(null, param);
		s.setReturnValue(result);
	    } catch (IllegalAccessException x) {
		jq.UNREACHABLE();
	    } catch (IllegalArgumentException x) {
		jq.UNREACHABLE();
	    } catch (InvocationTargetException x) {
		s.setThrown(x.getTargetException());
	    }
	    return s;
	}

	static HashSet cantInterpret = new HashSet();
	static {
	    jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/PrintStream;");
	    jq_Method m = k.getOrCreateInstanceMethod("write", "(Ljava/lang/String;)V");
	    cantInterpret.add(m);
	}

	public State invokeMethod(jq_Method f, ParamListOperand plo) {
	    if (TRACE) System.out.println("Invoking "+f);
	    jq_Class c = f.getDeclaringClass();
	    c.load(); c.verify(); c.prepare(); c.sf_initialize(); c.cls_initialize();
	    if (cantInterpret.contains(f) || f.isNative() || f instanceof jq_Initializer)
		return invokeReflective(f, plo);
	    ControlFlowGraph cfg = CodeCache.getCode(f);
	    State s = new State();
	    Object[] param = new Object[plo.length()];
	    for (int i=0; i<plo.length(); ++i) {
		param[i] = getReg(plo.get(i).getRegister());
	    }
	    s.interpretMethod(f, param, cfg.getRegisterFactory(), cfg);
	    if (TRACE) System.out.println("Finished interpreting "+f);
	    return s;
	}

        public static State interpretMethod(jq_Method f, Object[] params) {
	    State s = new State();
	    ControlFlowGraph cfg = CodeCache.getCode(f);
	    s.interpretMethod(f, params, cfg.getRegisterFactory(), cfg);
	    return s;
	}

        public void interpretMethod(jq_Method m, Object[] params, RegisterFactory rf, ControlFlowGraph cfg) {
            this.cfg = cfg; this.rf = rf;
	    // initialize parameters
	    jq_Type[] paramTypes = m.getParamTypes();
	    for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
		Register r = rf.getLocal(j, paramTypes[i]);
		registers.put(r, params[i]);
		if (paramTypes[i].getReferenceSize() == 8) ++j;
	    }
	    // start interpretation
	    current_bb = cfg.entry();
	    for (;;) {
		current_iterator = current_bb.iterator();
		while (current_iterator.hasNext()) {
		    Quad q = current_iterator.nextQuad();
		    q.accept(this);
		}
		if (current_bb.isExit()) break;
		current_bb = current_bb.getFallthroughSuccessor();
	    }
	}

	public void branchTo(BasicBlock bb) {
	    if (TRACE) System.out.println("Branching to: "+bb);
	    current_bb = bb;
	    current_iterator = bb.iterator();
	}

	public void handleException(Throwable x) {
	    jq_Class t = (jq_Class)ReflectiveVMInterface.INSTANCE.getJQTypeOf(x);
            t.load(); t.verify(); t.prepare();
	    ExceptionHandler eh = current_bb.getExceptionHandlers().mustCatch(t);
	    if (eh != null) {
		Register r = rf.getStack(0, t);
		registers.put(r, x);
		branchTo(eh.getEntry());
	    } else {
		thrown = x;
		branchTo(cfg.exit());
	    }
	}

        public int getReg_I(Register r) { return ((Integer)registers.get(r)).intValue(); }
        public float getReg_F(Register r) { return ((Float)registers.get(r)).floatValue(); }
        public long getReg_L(Register r) { return ((Long)registers.get(r)).longValue(); }
        public double getReg_D(Register r) { return ((Double)registers.get(r)).doubleValue(); }
        public Object getReg_A(Register r) { return registers.get(r); }
        public Object getReg(Register r) { return registers.get(r); }
        
        public void putReg_I(Register r, int i) { registers.put(r, new Integer(i)); }
        public void putReg_F(Register r, float i) { registers.put(r, new Float(i)); }
        public void putReg_L(Register r, long i) { registers.put(r, new Long(i)); }
        public void putReg_D(Register r, double i) { registers.put(r, new Double(i)); }
        public void putReg_A(Register r, Object i) { registers.put(r, i); }
        public void putReg(Register r, Object i) { registers.put(r, i); }
        
	public void replaceUninitializedReferences(Object o, UninitializedReference u) {
	    Iterator i = registers.entrySet().iterator();
	    while (i.hasNext()) {
		Map.Entry e = (Map.Entry)i.next();
		if (e.getValue() == u) e.setValue(o);
	    }
	}

	public String toString() {
	    if (thrown != null) return "Thrown exception: "+thrown+" (null checks: "+num_nullcheck+")";
	    return "Returned: "+return_value+" (null checks: "+num_nullcheck+")";
	}
    }
    
    public static class UninitializedReference {
        public jq_Class k;
        public UninitializedReference(jq_Class k) { this.k = k; }
	public String toString() { return k+" <uninit>"; }
    }
}

/*
 * QuadInterpreter.java
 *
 * Created on February 8, 2002, 3:06 PM
 */

package Interpreter;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
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
import Util.FilterIterator.Filter;
import jq;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class QuadInterpreter {

    public static class State extends QuadVisitor.EmptyVisitor {
	jq_Method method;
        Map/*<Register, Object>*/ registers;
        ControlFlowGraph cfg;
        RegisterFactory rf;
        BasicBlock current_bb;
        ListIterator.Quad current_iterator;
        Quad current_quad;
	Object return_value;
	Throwable thrown;

        public State(jq_Method m) {
            registers = new HashMap();
	    method = m;
        }
        
	public static boolean TRACE = false;

        public static long num_nullcheck = 0;
        
	public void visitNullCheck(Quad q) { ++num_nullcheck; }
            
	public void visitQuad(Quad q) {
	    if (TRACE) System.out.println("Registers: "+registers);
	    if (TRACE) System.out.println("Interpreting: "+q);
	    current_quad = q;
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
	public Object[] generateParamArray(jq_Method f, ParamListOperand plo) {
	    int offset = f.isStatic()?0:1;
	    jq_Type[] paramTypes = f.getParamTypes();
	    Object[] param = new Object[plo.length()-offset];
	    for (int i=offset; i<plo.length(); ++i) {
		if (paramTypes[i] == jq_Primitive.BYTE) {
		    param[i-offset] = new Byte((byte)getReg_I(plo.get(i).getRegister()));
		} else if (paramTypes[i] == jq_Primitive.CHAR) {
		    param[i-offset] = new Character((char)getReg_I(plo.get(i).getRegister()));
		} else if (paramTypes[i] == jq_Primitive.SHORT) {
		    param[i-offset] = new Short((short)getReg_I(plo.get(i).getRegister()));
		} else if (paramTypes[i] == jq_Primitive.BOOLEAN) {
		    param[i-offset] = new Boolean(getReg_I(plo.get(i).getRegister()) != 0);
		} else {
		    param[i-offset] = getReg(plo.get(i).getRegister());
		}
	    }
	    return param;
	}
	public State invokeInstanceReflective(jq_InstanceMethod f, ParamListOperand plo) {
	    State s = new State(f);
	    try {
		Object[] param = generateParamArray(f, plo);
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
	    State s = new State(f);
	    if (f == Allocator.HeapAllocator._multinewarray) {
		// special case
		int dim = getReg_I(plo.get(0).getRegister());
		jq_Type t = (jq_Type)getReg_A(plo.get(1).getRegister());
		int[] dims = new int[dim];
		for (int i=0; i<dim; ++i)
		    dims[dim-i-1] = getReg_I(plo.get(i+2).getRegister());
		for (int i=0; i<dims.length; ++i) {
		    t.load(); t.verify(); t.prepare(); t.sf_initialize(); t.cls_initialize();
		    t = ((jq_Array)t).getElementType();
		}
		try {
		    s.return_value = java.lang.reflect.Array.newInstance(Reflection.getJDKType(t), dims);
		} catch (Throwable x) {
		    s.setThrown(x);
		}
		return s;
	    }
	    Object[] param = generateParamArray(f, plo);
	    try {
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

        static Set bad_methods;
        static Set bad_classes;
        static Filter interpret_filter;
	static {
            bad_classes = new HashSet();
            bad_classes.add(Reflection._class);
            bad_methods = new HashSet();
	    jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/PrintStream;");
	    jq_Method m2 = k.getOrCreateInstanceMethod("write", "(Ljava/lang/String;)V");
	    bad_methods.add(m2);
	    bad_methods.add(Allocator.HeapAllocator._multinewarray);
            interpret_filter = new Filter() {
                public boolean isElement(Object o) {
                    jq_Method m = (jq_Method)o;
                    if (m.isNative()) return false;
                    if (m.getBytecode() == null) return false;
                    if (m instanceof jq_Initializer) return false;
                    if (bad_classes.contains(m.getDeclaringClass())) return false;
                    if (bad_methods.contains(m)) return false;
                    return true;
                }
            };
	}

	public State invokeMethod(jq_Method f, ParamListOperand plo) {
	    if (TRACE) System.out.println("Invoking "+f);
	    jq_Class c = f.getDeclaringClass();
	    c.load(); c.verify(); c.prepare(); c.sf_initialize(); c.cls_initialize();
	    if (!interpret_filter.isElement(f))
		return invokeReflective(f, plo);
	    ControlFlowGraph cfg = CodeCache.getCode(f);
	    State s = new State(f);
	    Object[] param = new Object[plo.length()];
	    for (int i=0; i<plo.length(); ++i) {
		param[i] = getReg(plo.get(i).getRegister());
	    }
	    s.interpretMethod(f, param, cfg.getRegisterFactory(), cfg);
	    if (TRACE) System.out.println("Finished interpreting "+f);
	    return s;
	}

        public static State interpretMethod(jq_Method f, Object[] params) {
	    State s = new State(f);
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
                if (TRACE) System.out.println("Method "+method+" handler "+eh+" catches "+x);
	    } else {
		thrown = x;
		branchTo(cfg.exit());
                if (TRACE)
		    System.out.println("Method "+method+" does not catch "+x);
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

	public String currentLocation() { return method+" "+current_bb+" quad#"+current_quad.getID(); }

	public String toString() {
            if (thrown != null)
                return "Thrown exception: "+thrown+" (null checks: "+num_nullcheck+")";
	    return "Returned: "+return_value+" (null checks: "+num_nullcheck+")";
	}
        
    }
    
    public static class UninitializedReference {
        public jq_Class k;
        public UninitializedReference(jq_Class k) { this.k = k; }
	public String toString() { return k+" <uninit>"; }
    }
}

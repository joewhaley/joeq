package Clazz;

import java.util.Iterator;

import Allocator.DefaultHeapAllocator;
import Assembler.x86.DirectBindCall;
import Bootstrap.BootstrapCodeAddress;
import Bootstrap.BootstrapHeapAddress;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Compil3r.Reference.x86.x86ReferenceCompiler;
import Compil3r.Reference.x86.x86ReferenceLinker;
import ClassLib.ClassLibInterface;
import Compil3r.Compil3rInterface;
import Main.jq;
import Run_Time.DebugInterface;
import Run_Time.TypeCheck;
import Run_Time.StackCodeWalker;

class Delegates implements jq_ClassFileConstants {
    static class Field implements jq_Field.Delegate {
	public final boolean isCodeAddressType(jq_Field f) {
	    return f.getType() == CodeAddress._class ||
		f.getType() == BootstrapCodeAddress._class;
	}
	public final boolean isHeapAddressType(jq_Field f) {
	    return f.getType() == HeapAddress._class ||
		f.getType() == BootstrapHeapAddress._class;
	}
	public final boolean isStackAddressType(jq_Field f) {
	    return f.getType() == StackAddress._class;
	}
    }
    static class Method implements jq_Method.Delegate {
	public final jq_CompiledCode compile_stub (jq_Method m) {
	    return x86ReferenceCompiler.generate_compile_stub(m);
	}
	public final jq_CompiledCode compile (jq_Method m) {
	    jq_CompiledCode default_compiled_version;
            //System.out.println("Compiling: "+m);
            if (m.isNative() && m.getBytecode() == null) {
                System.out.println("Unimplemented native method! "+m);
                if (x86ReferenceLinker._nativeMethodError.getState() < STATE_CLSINITIALIZED) {
                    jq_Class k = x86ReferenceLinker._class;
                    k.verify(); //k.prepare();
                    if (x86ReferenceLinker._nativeMethodError.getState() != STATE_PREPARED)
                        x86ReferenceLinker._nativeMethodError.prepare();
                    default_compiled_version = x86ReferenceLinker._nativeMethodError.compile();
                    //if (k != getDeclaringClass() && getDeclaringClass().getSuperclass() != null) { k.cls_initialize(); }
                } else {
                    default_compiled_version = x86ReferenceLinker._nativeMethodError.getDefaultCompiledVersion();
                }
            } else if (m.isAbstract()) {
                if (x86ReferenceLinker._abstractMethodError.getState() < STATE_CLSINITIALIZED) {
                    jq_Class k = x86ReferenceLinker._class;
                    k.verify(); //k.prepare();
                    //default_compiled_version = x86ReferenceLinker._abstractMethodError.getDefaultCompiledVersion();
                    if (x86ReferenceLinker._abstractMethodError.getState() != STATE_PREPARED)
                        x86ReferenceLinker._abstractMethodError.prepare();
                    default_compiled_version = x86ReferenceLinker._abstractMethodError.compile();
                    //if (k != getDeclaringClass() && getDeclaringClass().getSuperclass() != null) { k.cls_initialize(); }
                } else {
                    default_compiled_version = x86ReferenceLinker._abstractMethodError.getDefaultCompiledVersion();
                }
            } else {
                Compil3rInterface c;
                if (true)
                    c = new x86ReferenceCompiler(m);
                //else
                //    c = new x86OpenJITCompiler(m);
                default_compiled_version = c.compile();
                if (jq.RunningNative)
                    default_compiled_version.patchDirectBindCalls();
            }
	    return default_compiled_version;
	}
    }
    static class CompiledCode implements jq_CompiledCode.Delegate {
	public void patchDirectBindCalls (Iterator i) {
	    while (i.hasNext()) {
		DirectBindCall r = (DirectBindCall) i.next();
		r.patch();
	    }
	}
	public void patchDirectBindCalls (Iterator i, jq_Method method, jq_CompiledCode cc) {
            while (i.hasNext()) {
                DirectBindCall r = (DirectBindCall) i.next();
                if (r.getTarget() == method) {
                    if (jq_CompiledCode.TRACE_REDIRECT) DebugInterface.debugwriteln("patching direct bind call in " + this + " at " + r.getSource().stringRep() + " to refer to " + cc);
                    r.patchTo(cc);
                }
            }
	}
    }
    static class Klass implements jq_Class.Delegate {
	public final Object newInstance(jq_Class c, int instance_size, Object vtable) {
	    c.cls_initialize();
	    return DefaultHeapAllocator.allocateObject(instance_size, vtable);
	}
    }
    static class Array implements jq_Array.Delegate {
	public final Object newInstance(jq_Array a, int length, Object vtable) {
	    return DefaultHeapAllocator.allocateArray(length, a.getInstanceSize(length), vtable);
	}
    }

    static class Member implements jq_Member.Delegate {
	public final void checkCallerAccess(jq_Member m, int depth) throws IllegalAccessException {
	    jq_Class field_class = m.getDeclaringClass();
	    if (m.isPublic() && field_class.isPublic()) {
		// completely public!
		return;
	    }
	    StackCodeWalker sw = new StackCodeWalker(null, StackAddress.getBasePointer());
	    while (--depth >= 0) sw.gotoNext();
	    jq_CompiledCode cc = sw.getCode();
	    if (cc != null) {
		jq_Class caller_class = cc.getMethod().getDeclaringClass();
		if (caller_class == field_class) {
		    // same class! access allowed!
		    return;
		}
		if (field_class.isPublic() || caller_class.isInSamePackage(field_class)) {
		    if (m.isPublic()) {
			// class is accessible and field is public!
			return;
		    }
		    if (m.isProtected()) {
			if (TypeCheck.isAssignable(caller_class, field_class)) {
			    // field is protected and field_class is supertype of caller_class!
			    return;
			}
		    }
		    if (!m.isPrivate()) {
			if (caller_class.isInSamePackage(field_class)) {
			    // field is package-private and field_class and caller_class are in the same package!
			    return;
			}
		    }
		}
	    }
	    throw new IllegalAccessException();
	}
    }
}

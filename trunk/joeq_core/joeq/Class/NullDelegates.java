package Clazz;

import java.util.Iterator;

import Main.jq;

abstract class NullDelegates {
    static class Field implements jq_Field.Delegate {
	public final boolean isCodeAddressType(jq_Field f) { return false; }
	public final boolean isHeapAddressType(jq_Field f) { return false; }
	public final boolean isStackAddressType(jq_Field f) { return false; }
    }

    static class Method implements jq_Method.Delegate {
	public final jq_CompiledCode compile_stub (jq_Method m) {
	    return null;
	}
	public final jq_CompiledCode compile (jq_Method m) {
	    return null;
	}
    }

    static class CompiledCode implements jq_CompiledCode.Delegate {
	public final void patchDirectBindCalls (Iterator i) { }
	public final void patchDirectBindCalls (Iterator i, jq_Method m, jq_CompiledCode cc) { }
    }

    static class Klass implements jq_Class.Delegate {
	public final Object newInstance(jq_Class c, int instance_size, Object vtable) {
	    try {
		return Class.forName(c.getName()).newInstance();
	    } catch (Exception e) { return null; }
	}
    }

    static class Array implements jq_Array.Delegate {
	public final Object newInstance(jq_Array a, int length, Object vtable) {
	    jq.UNREACHABLE("Can't create new arrays!");
	    return null;
	}
    }

    static class Member implements jq_Member.Delegate {
	public final void checkCallerAccess(jq_Member m, int depth) throws IllegalAccessException {
	    jq_Class field_class = m.getDeclaringClass();
	    if (m.isPublic() && field_class.isPublic()) {
		// completely public!
		return;
	    }
	    /* TODO: Handle the other cases reflectively somehow */
	    throw new IllegalAccessException();
	}
    }
}

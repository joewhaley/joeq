package Clazz;

import java.util.Iterator;

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
}
